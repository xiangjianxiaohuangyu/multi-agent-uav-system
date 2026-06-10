"""FAISS 索引管理器。

封装 ``IndexIDMap2(IndexFlatL2)`` + ``id_map.json`` 重建缓冲：

- ``index.faiss`` — FAISS 二进制，主搜索路径
- ``id_map.json`` — ``{str(experience_id): [11 floats]}``，用于索引损坏时从
  MySQL/JSON 重建 FAISS（FAISS 内部虽然存了向量，但没有事务日志）
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

log = logging.getLogger(__name__)


class FaissIndexManager:
    """线程安全的 FAISS 索引包装器。

    所有变更操作（add / delete / update / rebuild）都在 ``_lock`` 内串行执行，
    避免并发写盘导致文件损坏。FAISS 搜索自身是线程安全的，但为了日志一致性
    我们把 search 也放进锁里（11 维小数据集，开销可忽略）。
    """

    def __init__(
        self,
        dimension: int,
        index_path: Path,
        id_map_path: Path,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        self.dimension: int = int(dimension)
        self.index_path: Path = Path(index_path)
        self.id_map_path: Path = Path(id_map_path)
        self._index: faiss.IndexIDMap2 | None = None
        self._id_to_vec: dict[int, list[float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #

    def load_or_create(self) -> None:
        """启动时调用：恢复索引；必要时从 id_map 重建。"""
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            if self.index_path.is_file():
                try:
                    self._index = faiss.read_index(str(self.index_path))
                    if self._index.d != self.dimension:
                        log.warning(
                            "faiss dim mismatch: file=%s expected=%s; rebuilding from id_map",
                            self._index.d, self.dimension,
                        )
                        self._index = None
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to read faiss index %s: %s", self.index_path, exc)
                    self._index = None

            # 加载或重建 id_map
            if self.id_map_path.is_file():
                try:
                    with self.id_map_path.open("r", encoding="utf-8") as f:
                        raw = json.load(f)
                    self._id_to_vec = {int(k): list(v) for k, v in raw.items()}
                except Exception as exc:  # noqa: BLE001
                    log.warning("failed to read id_map %s: %s", self.id_map_path, exc)
                    self._id_to_vec = {}
            else:
                self._id_to_vec = {}

            # 三种恢复策略
            if self._index is None and self._id_to_vec:
                log.info("faiss index missing, rebuilding from id_map (n=%d)", len(self._id_to_vec))
                self._rebuild_locked()
                self._save_locked()
            elif self._index is None:
                self._index = self._new_index()
                self._save_locked()
            else:
                # 索引已加载；尝试校验与 id_map 一致性
                if self._index.ntotal != len(self._id_to_vec):
                    log.warning(
                        "faiss ntotal=%d != id_map size=%d; rebuilding from id_map",
                        self._index.ntotal, len(self._id_to_vec),
                    )
                    self._rebuild_locked()
                    self._save_locked()
            log.info("faiss index ready (ntotal=%d, dim=%d)", self.ntotal, self.dimension)

    def save(self) -> None:
        """把索引和 id_map 写盘。"""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        if self._index is None:
            return
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to write faiss index: %s", exc)
            raise
        try:
            tmp = self.id_map_path.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(
                    {str(k): list(v) for k, v in self._id_to_vec.items()},
                    f, ensure_ascii=False,
                )
            tmp.replace(self.id_map_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("failed to write id_map: %s", exc)
            raise

    def _new_index(self) -> faiss.IndexIDMap2:
        flat = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIDMap2(flat)

    # ------------------------------------------------------------------ #
    # 读写
    # ------------------------------------------------------------------ #

    @property
    def ntotal(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def add_vector(self, vec_id: int, vector: np.ndarray) -> None:
        """添加一个向量。

        Raises:
            ValueError: ``vec_id`` 已存在。
            RuntimeError: 索引未初始化。
        """
        vec = self._as_vector(vector)
        with self._lock:
            if self._index is None:
                self._index = self._new_index()
            if vec_id in self._id_to_vec:
                raise ValueError(f"vec_id {vec_id} already exists; use update_vector")
            ids = np.asarray([vec_id], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
            self._id_to_vec[vec_id] = vec.flatten().tolist()
            self._save_locked()

    def delete_vector(self, vec_id: int) -> bool:
        """删除一个向量。返回是否真的删除了（id 不存在则返回 False，不抛）。"""
        with self._lock:
            if self._index is None or vec_id not in self._id_to_vec:
                return False
            ids = np.asarray([vec_id], dtype=np.int64)
            try:
                self._index.remove_ids(ids)
            except Exception as exc:  # noqa: BLE001
                log.warning("faiss remove_ids(%s) failed: %s", vec_id, exc)
            self._id_to_vec.pop(vec_id, None)
            self._save_locked()
            return True

    def update_vector(self, vec_id: int, vector: np.ndarray) -> bool:
        """更新一个向量：先删再加。返回是否更新成功。"""
        vec = self._as_vector(vector)
        with self._lock:
            if self._index is None or vec_id not in self._id_to_vec:
                return False
            ids = np.asarray([vec_id], dtype=np.int64)
            try:
                self._index.remove_ids(ids)
            except Exception as exc:  # noqa: BLE001
                log.warning("faiss remove_ids(%s) during update failed: %s", vec_id, exc)
            self._index.add_with_ids(vec, ids)
            self._id_to_vec[vec_id] = vec.flatten().tolist()
            self._save_locked()
            return True

    def search_topk(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-K L2 搜索。

        Returns:
            ``(distances, ids)`` 两个 ``np.ndarray``（长度 ``min(k, ntotal)``）。
            空索引时返回 ``(np.array([]), np.array([]))``。
        """
        q = self._as_vector(query)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                empty = np.empty((0,), dtype=np.float32)
                return empty.copy(), empty.astype(np.int64).copy()
            kk = max(1, min(int(k), int(self._index.ntotal)))
            distances, ids = self._index.search(q, kk)
            return distances[0], ids[0]

    # ------------------------------------------------------------------ #
    # 重建
    # ------------------------------------------------------------------ #

    def rebuild_from(self, id_to_vector: dict[int, list[float]]) -> None:
        """从 id → vector 字典重建整个索引（用于从 MySQL 全量回灌）。"""
        with self._lock:
            self._id_to_vec = {int(k): list(v) for k, v in id_to_vector.items()}
            self._rebuild_locked()
            self._save_locked()

    def _rebuild_locked(self) -> None:
        self._index = self._new_index()
        if not self._id_to_vec:
            return
        ids_arr = np.array(sorted(self._id_to_vec.keys()), dtype=np.int64)
        vecs = np.array(
            [self._id_to_vec[i] for i in ids_arr.tolist()],
            dtype=np.float32,
        )
        if vecs.ndim == 1:
            vecs = vecs.reshape(-1, self.dimension)
        self._index.add_with_ids(vecs, ids_arr)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    def _as_vector(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dimension:
            raise ValueError(
                f"vector dim {arr.shape[1]} != index dim {self.dimension}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("vector contains non-finite values (NaN/Inf)")
        return arr

    # 用于测试 / 调试
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ntotal": int(self._index.ntotal) if self._index else 0,
                "dimension": self.dimension,
                "id_count": len(self._id_to_vec),
                "index_path": str(self.index_path),
                "id_map_path": str(self.id_map_path),
            }
