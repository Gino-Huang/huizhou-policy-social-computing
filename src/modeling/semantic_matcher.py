"""语义匹配：sentence-transformers 句向量 + 余弦相似度 Top-K 检索（含阈值）。

设计要点：
- 模型延迟加载（lazy init）：import 本模块、实例化类都不会下载/加载模型，
  只有第一次调用 encode() 时才加载。测试与 CI 无需 GPU、无需下载。
- 每个热点只保留相似度 >= threshold 的政策段落；若无达标者则记为 None
  （无可靠匹配），避免"硬匹配"制造虚假关联。
- 附带 TF-IDF 余弦基线，便于消融对比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import jieba_tokenizer


class SemanticMatcher:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        threshold: float = 0.55,
        top_k: int = 5,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.top_k = top_k
        self.batch_size = batch_size
        self.device = device
        self._model = None  # 延迟加载

    def _ensure_model(self):
        """首次编码时加载模型；失败时抛出带提示的异常。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码为 L2 归一化句向量。"""
        model = self._ensure_model()
        return np.asarray(
            model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )

    def similarity_matrix(
        self,
        policy_texts: list[str],
        hotspot_texts: list[str],
    ) -> np.ndarray:
        """返回 (N 政策段落 x M 热点) 余弦相似度矩阵。

        因向量已 L2 归一化，余弦相似度 = 点积。
        """
        p = self.encode(policy_texts)
        h = self.encode(hotspot_texts)
        return (p @ h.T).astype(float)

    def match(
        self,
        policy_ids: list[str],
        hotspot_ids: list[str],
        sim_matrix: np.ndarray,
    ) -> pd.DataFrame:
        """Top-K 检索：每行 (hotspot_id, policy_id, similarity, rank)。

        低于 threshold 的候选被丢弃；某热点若无达标候选，则输出一行
        policy_id=None 的记录，标记"无可靠匹配"。
        """
        rows: list[dict] = []
        for j, hid in enumerate(hotspot_ids):
            sims = sim_matrix[:, j]
            order = np.argsort(sims)[::-1][: self.top_k]
            kept = [(int(i), float(sims[i])) for i in order if sims[i] >= self.threshold]
            if not kept:
                rows.append(
                    {"hotspot_id": hid, "policy_id": None, "similarity": 0.0, "rank": 0}
                )
                continue
            for rank, (i, s) in enumerate(kept, start=1):
                rows.append(
                    {
                        "hotspot_id": hid,
                        "policy_id": policy_ids[i],
                        "similarity": s,
                        "rank": rank,
                    }
                )
        return pd.DataFrame(rows)


def tfidf_cosine_baseline(
    policy_texts: list[str],
    hotspot_texts: list[str],
) -> np.ndarray:
    """TF-IDF 余弦相似度基线：返回 (N 政策 x M 热点) 相似度矩阵。

    用于与句向量方法的消融对比：若两者 Top-K 重合度高，说明词汇重叠
    已能解释大部分匹配；若差异大，说明语义泛化带来了增量价值。
    """
    vec = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None, lowercase=False)
    p = vec.fit_transform(policy_texts)
    h = vec.transform(hotspot_texts)
    return cosine_similarity(p, h)
