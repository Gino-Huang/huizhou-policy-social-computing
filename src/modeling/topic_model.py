"""主题建模基线：TF-IDF + 每文档关键词 + KMeans 聚类（含轮廓系数评估）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.preprocessing.text_cleaner import jieba_tokenizer


class TfidfTopicModel:
    """TF-IDF 主题建模器（可解释的强基线）。

    用法:
        model = TfidfTopicModel(max_features=5000)
        model.fit(texts)
        model.top_keywords_per_doc(doc_idx=0, n=10)
        labels = model.fit_kmeans(n_clusters=8, seed=42)
        model.evaluate_silhouette(labels)
    """

    def __init__(self, max_features: int = 5000, min_df: int = 1):
        self.vectorizer = TfidfVectorizer(
            tokenizer=jieba_tokenizer,
            token_pattern=None,
            lowercase=False,
            max_features=max_features,
            min_df=min_df,
        )
        self.tfidf_matrix = None
        self.feature_names: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "TfidfTopicModel":
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return self

    def top_keywords_per_doc(self, doc_idx: int, n: int = 10) -> list[tuple[str, float]]:
        """返回某文档 TF-IDF 权重最高的 n 个词 (词, 权重)。"""
        if self.tfidf_matrix is None:
            raise RuntimeError("先调用 fit()")
        row = self.tfidf_matrix[doc_idx].toarray().ravel()
        top_idx = row.argsort()[::-1][:n]
        return [(self.feature_names[i], float(row[i])) for i in top_idx if row[i] > 0]

    def cluster_keywords(self, labels: np.ndarray, n: int = 10) -> dict[int, list[str]]:
        """按聚类聚合 TF-IDF 权重，输出每簇的代表性关键词。"""
        if self.tfidf_matrix is None:
            raise RuntimeError("先调用 fit()")
        result: dict[int, list[str]] = {}
        for c in sorted(set(labels)):
            rows = self.tfidf_matrix[np.asarray(labels) == c]
            scores = np.asarray(rows.sum(axis=0)).ravel()
            top_idx = scores.argsort()[::-1][:n]
            result[int(c)] = [self.feature_names[i] for i in top_idx if scores[i] > 0]
        return result

    def fit_kmeans(self, n_clusters: int, seed: int = 42) -> np.ndarray:
        """KMeans 聚类，返回每文档的簇标签。"""
        if self.tfidf_matrix is None:
            raise RuntimeError("先调用 fit()")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(self.tfidf_matrix)

    def evaluate_silhouette(self, labels: np.ndarray) -> float:
        """轮廓系数（越接近 1 聚类结构越清晰；需 >=2 个簇）。"""
        if self.tfidf_matrix is None:
            raise RuntimeError("先调用 fit()")
        if len(set(labels)) < 2:
            return float("nan")
        return float(silhouette_score(self.tfidf_matrix, labels))

    def doc_topic_distribution(self, labels: np.ndarray) -> pd.Series:
        """文档在各主题上的分布（计数归一化）。"""
        s = pd.Series(labels).value_counts(normalize=True).sort_index()
        return s
