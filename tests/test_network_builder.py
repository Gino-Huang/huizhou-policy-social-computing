"""network_builder 单元测试：用合成相似度矩阵建图并校验结构。"""
import numpy as np

from src.modeling.network_builder import (
    build_bipartite_graph,
    compare_time_slices,
    detect_communities,
    hotspot_projection,
    network_density,
    weighted_degree_ranking,
)

POLICIES = ["p1", "p2", "p3"]
HOTSPOTS = ["h1", "h2"]
# p1-h1 强相关，p2-h1/p2-h2 中等，p3 无达标边
SIM = np.array(
    [
        [0.90, 0.10],
        [0.60, 0.70],
        [0.20, 0.30],
    ]
)
TH = 0.55


def test_build_bipartite_graph_structure():
    G = build_bipartite_graph(POLICIES, HOTSPOTS, SIM, threshold=TH)
    assert G.number_of_nodes() == 5
    # 达标的边：p1-h1, p2-h1, p2-h2
    assert G.number_of_edges() == 3
    assert G["p1"]["h1"]["weight"] == 0.90
    kinds = {n: d["kind"] for n, d in G.nodes(data=True)}
    assert kinds["p1"] == "policy" and kinds["h1"] == "hotspot"


def test_weighted_degree_ranking():
    G = build_bipartite_graph(POLICIES, HOTSPOTS, SIM, threshold=TH)
    ranked = weighted_degree_ranking(G, kind="policy")
    # p2 加权度 = 0.6+0.7=1.3 最高；p3 无边 = 0
    assert ranked[0][0] == "p2"
    assert abs(ranked[0][1] - 1.3) < 1e-9
    assert ranked[-1] == ("p3", 0.0)


def test_hotspot_projection():
    G = build_bipartite_graph(POLICIES, HOTSPOTS, SIM, threshold=TH)
    P = hotspot_projection(G)
    # h1 与 h2 共享邻居 p2，应相连
    assert P.has_edge("h1", "h2")
    assert P["h1"]["h2"]["weight"] > 0


def test_network_density_and_communities():
    G = build_bipartite_graph(POLICIES, HOTSPOTS, SIM, threshold=TH)
    # 3 条边 / (3*2) 可能边 = 0.5
    assert abs(network_density(G) - 0.5) < 1e-9
    comms = detect_communities(G)
    assert sum(len(c) for c in comms) == 5  # 全覆盖


def test_compare_time_slices():
    g1 = build_bipartite_graph(POLICIES, HOTSPOTS, SIM, threshold=TH)
    g2 = build_bipartite_graph(POLICIES, HOTSPOTS, SIM * 0, threshold=TH)  # 空图
    summary = compare_time_slices({"w1": g1, "w2": g2})
    assert summary["w1"]["n_edges"] == 3
    assert summary["w2"]["n_edges"] == 0
    assert summary["w2"]["density"] == 0.0
