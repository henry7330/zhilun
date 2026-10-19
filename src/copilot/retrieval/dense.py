"""语义向量召回（dense）—— 基于 model2vec 的静态嵌入。

为什么选 model2vec 而不是 sentence-transformers
---------------------------------------------
`sentence-transformers` 会拉入 **torch（数 GB）**，而本项目当前阶段
只需要一个可用的语义信号来做**消融对照**。`model2vec` 提供蒸馏后的
静态嵌入（256 维），依赖里没有 torch，下载量小一两个数量级。

代价要说清楚：
- 静态嵌入**没有上下文感知**（同一个词在任何语境下向量相同）；
- 表达力弱于 BERT 类编码器，绝对指标会偏低。

这在本项目里是可接受的：dense 层的角色是**与词面方法融合后的增量**，
而不是追求 SOTA。若日后需要更强语义，替换本模块的实现即可，
`Retriever` 接口不变。

可复现性注意
------------
模型权重会随上游更新而漂移，因此 :class:`DenseRetriever` 会**记录模型名**，
报告里也应写明，否则历史指标无法对照。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from copilot.retrieval.base import Chunk, ScoredChunk

__all__ = ["DenseRetriever", "DEFAULT_MODEL", "DEFAULT_DIM_NOTE"]

#: 多语言静态嵌入。英文模型（如 potion-base-8M）对中文无意义，必须用 multilingual。
DEFAULT_MODEL = "minishlab/potion-multilingual-128M"
DEFAULT_DIM_NOTE = "256 维静态嵌入（无上下文感知）"


class DenseRetriever:
    """用静态嵌入做余弦相似度召回。

    向量在构造时**一次性预算**（片段数量小，几十到几千，代价可忽略），
    查询只编码一条。全部向量做 L2 归一化，因此点积即余弦。
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        name: str = "dense",
    ) -> None:
        if not chunks:
            raise ValueError("chunks 不能为空")
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise ImportError(
                "需要 model2vec。请安装：pip install -e \".[retrieval]\""
            ) from exc

        self._chunks = list(chunks)
        self.model_name = model_name
        self.name = name
        self._model = StaticModel.from_pretrained(model_name)

        vecs = self._model.encode(
            [c.text for c in self._chunks], batch_size=batch_size
        )
        self._vecs = self._normalize(np.asarray(vecs, dtype=np.float32))
        self.dim = int(self._vecs.shape[1])

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def retrieve(self, query: str, k: int = 10) -> list[ScoredChunk]:
        if not query.strip():
            return []
        qv = np.asarray(self._model.encode([query]), dtype=np.float32)
        qv = self._normalize(qv)[0]
        sims = self._vecs @ qv
        top = np.argsort(sims)[::-1][:k]
        return [
            ScoredChunk(self._chunks[i], float(sims[i]))
            for i in top
            if sims[i] > 0
        ]
