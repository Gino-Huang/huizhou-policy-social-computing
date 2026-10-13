"""去重：精确去重（归一化文本）+ 近似去重（TF-IDF 余弦相似度）。"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.text_cleaner import clean_text, jieba_tokenizer


def exact_dedup(df: pd.DataFrame, text_col: str = "text") -> tuple[pd.DataFrame, int]:
    """按清洗后的文本精确去重。返回 (去重后 df, 删除行数)。"""
    norm = df[text_col].fillna("").map(clean_text)
    mask = ~norm.duplicated(keep="first")
    return df.loc[mask].reset_index(drop=True), int((~mask).sum())


def near_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.9,
) -> tuple[pd.DataFrame, int]:
    """近似去重：TF-IDF 余弦相似度 >= threshold 的视为重复，保留第一条。

    注意：O(n^2) 内存，只适合中小规模数据；大规模请改用 MinHash/LSH。
    """
    texts = df[text_col].fillna("").map(clean_text).tolist()
    if len(texts) <= 1:
        return df.reset_index(drop=True), 0
    vec = TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None, lowercase=False)
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat)
    keep = [True] * len(texts)
    removed = 0
    for i in range(len(texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if keep[j] and sim[i, j] >= threshold:
                keep[j] = False
                removed += 1
    return df.loc[keep].reset_index(drop=True), removed
