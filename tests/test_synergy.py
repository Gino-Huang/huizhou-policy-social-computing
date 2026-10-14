"""协同网络模块测试：关键词共现图、主题关联图、中心性报告。"""
import numpy as np
import pandas as pd

from src.modeling.synergy import (
    build_keyword_cooccurrence,
    build_topic_graph,
    centrality_report,
)


def _toy_keywords():
    # 4 个"段落"的关键词表
    return [
        ["产业", "人才", "企业"],
        ["产业", "人才", "创新"],
        ["产业", "教育", "学校"],
        ["旅游", "海洋", "生态"],
    ]


def test_keyword_cooccurrence_structure():
    G = build_keyword_cooccurrence(_toy_keywords(), min_df=2, min_weight=1)
    # "产业"出现在3段，"人才"2段；"企业/创新/教育/学校/旅游/海洋/生态"只出现1段被过滤
    assert set(G.nodes()) == {"产业", "人才"}
    assert G.number_of_edges() == 1
    assert G["产业"]["人才"]["weight"] == 2  # 同现于前两段
    assert G.nodes["产业"]["df"] == 3


def test_keyword_cooccurrence_respects_min_weight():
    G = build_keyword_cooccurrence(_toy_keywords(), min_df=1, min_weight=2)
    # 权重>=2 的边只有 产业-人才
    assert G.number_of_edges() == 1
    assert ("产业", "人才") in G.edges() or ("人才", "产业") in G.edges()
    # 孤立节点应被移除
    import networkx as nx
    assert len(list(nx.isolates(G))) == 0


def test_topic_graph_top_k_and_centrality():
    rng = np.random.RandomState(0)
    centroids = rng.rand(5, 10)
    names = [f"主题{i}" for i in range(5)]
    G = build_topic_graph(centroids, names, keep="top_k", top_k=4)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 4
    # 边权应为余弦相似度（0~1）
    for _, _, d in G.edges(data=True):
        assert 0.0 <= d["weight"] <= 1.0
    rep = centrality_report(G, top_n=3)
    assert list(rep.columns) == ["node", "degree_centrality", "weighted_degree", "betweenness"]
    assert len(rep) == 3
    # 按 betweenness 降序
    assert rep["betweenness"].is_monotonic_decreasing
