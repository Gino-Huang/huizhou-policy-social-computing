"""政策协同网络分析：关键词共现网络 + 主题关联网络。

对应原计划选题3的"共现网络、社群发现、中心性分析"：
- 关键词共现图：节点=关键词，边权=同段落共现次数
- 主题关联图：节点=v2的15个政策主题，边权=TF-IDF空间主题中心向量的余弦相似度

重要边界：共现/相似度只描述文本层面的关联结构，不直接解释为因果关系或政策效果。
"""
from __future__ import annotations

import itertools
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.modeling.network_builder import detect_communities


def top_keywords_per_doc(vectorizer, tfidf_matrix, top_n: int = 8) -> list[list[str]]:
    """每篇文档取 TF-IDF 权重最高的 top_n 个词。"""
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    keywords: list[list[str]] = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(i).toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        keywords.append([str(feature_names[j]) for j in top_idx if row[j] > 0])
    return keywords


def build_keyword_cooccurrence(
    keywords_per_doc: list[list[str]],
    min_df: int = 3,
    min_weight: int = 2,
) -> nx.Graph:
    """构建关键词共现图。

    - 节点：至少出现在 min_df 个段落中的关键词；节点属性 df=段落文档频率
    - 边：两个关键词同现于一段落则计数；仅保留权重 >= min_weight 的边
    """
    df_count: Counter = Counter()
    for kws in keywords_per_doc:
        for kw in set(kws):
            df_count[kw] += 1
    vocab = {kw for kw, c in df_count.items() if c >= min_df}

    edge_weights: Counter = Counter()
    for kws in keywords_per_doc:
        kept = sorted(set(kws) & vocab)
        for a, b in itertools.combinations(kept, 2):
            edge_weights[(a, b)] += 1

    G = nx.Graph()
    for kw in vocab:
        G.add_node(kw, df=int(df_count[kw]))
    for (a, b), w in edge_weights.items():
        if w >= min_weight:
            G.add_edge(a, b, weight=int(w))
    # 去掉孤立节点
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def centrality_report(G: nx.Graph, top_n: int = 15) -> pd.DataFrame:
    """度中心性 / 加权度 / 中介中心性排名表。"""
    deg = nx.degree_centrality(G)
    wdeg = {n: float(G.degree(n, weight="weight")) for n in G.nodes()}
    btw = nx.betweenness_centrality(G, weight="weight", seed=42)
    df = pd.DataFrame({
        "node": list(G.nodes()),
        "degree_centrality": [deg[n] for n in G.nodes()],
        "weighted_degree": [wdeg[n] for n in G.nodes()],
        "betweenness": [btw[n] for n in G.nodes()],
    }).sort_values("betweenness", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


def build_topic_graph(
    centroids: np.ndarray,
    topic_names: list[str],
    keep: str = "top_k",
    top_k: int = 30,
    threshold: float = 0.15,
) -> nx.Graph:
    """主题关联图：节点=主题，边权=中心向量余弦相似度。

    选择余弦相似度而非关键词重叠度的理由：中心向量是在聚类所用的同一
    TF-IDF 特征空间中的分布表示，余弦相似度反映主题整体语义分布的接近
    程度；关键词重叠（Jaccard）只看 top-K 词表交集，对 K 的选择敏感，
    且会放大"发展/建设/推进"这类通用词的虚假关联。
    keep="top_k" 保留权重最高的 top_k 条边；keep="threshold" 保留 >= threshold 的边。
    """
    sim = cosine_similarity(centroids)
    n = len(topic_names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, float(sim[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    if keep == "top_k":
        chosen = pairs[:top_k]
    else:
        chosen = [p for p in pairs if p[2] >= threshold]

    G = nx.Graph()
    for i, name in enumerate(topic_names):
        G.add_node(i, name=name)
    for i, j, w in chosen:
        G.add_edge(i, j, weight=w)
    return G


def summarize_communities(G: nx.Graph, id_to_name: dict) -> list[dict]:
    """社群发现（贪心模块度），返回每个社群的成员名与规模。"""
    comms = detect_communities(G, method="greedy")
    out = []
    for ci, members in enumerate(comms):
        names = sorted(id_to_name[m] for m in members)
        out.append({"community": ci, "size": len(members), "members": names})
    return out
