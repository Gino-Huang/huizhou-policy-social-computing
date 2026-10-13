"""语义网络构建：政策-热点二部图、加权度排名、投影、社区发现、时间切片对比。"""
from __future__ import annotations

import networkx as nx
from networkx.algorithms import bipartite, community


def build_bipartite_graph(
    policy_ids: list[str],
    hotspot_ids: list[str],
    sim_matrix,
    threshold: float = 0.55,
) -> nx.Graph:
    """构建二部图：政策节点 <-> 热点节点，边权 = 相似度（仅保留 >= 阈值的边）。

    节点属性: kind in {"policy", "hotspot"}；边属性: weight。
    """
    G = nx.Graph()
    for pid in policy_ids:
        G.add_node(pid, kind="policy")
    for hid in hotspot_ids:
        G.add_node(hid, kind="hotspot")
    for i, pid in enumerate(policy_ids):
        for j, hid in enumerate(hotspot_ids):
            w = float(sim_matrix[i, j])
            if w >= threshold:
                G.add_edge(pid, hid, weight=w)
    return G


def weighted_degree_ranking(G: nx.Graph, kind: str = "policy") -> list[tuple[str, float]]:
    """按加权度（strength）降序排列指定类型的节点。"""
    nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
    ranked = sorted(nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)
    return [(n, float(G.degree(n, weight="weight"))) for n in ranked]


def hotspot_projection(G: nx.Graph) -> nx.Graph:
    """热点投影网络：共享政策邻居的热点相连，边权 = 共享邻居权重之积求和。"""
    hotspots = {n for n, d in G.nodes(data=True) if d.get("kind") == "hotspot"}
    return bipartite.weighted_projected_graph(G, hotspots)


def detect_communities(G: nx.Graph, method: str = "greedy") -> list[set]:
    """社区发现。method="greedy" 用 networkx 内置贪心模块度；"louvain" 需 python-louvain。

    返回社区列表（节点集合），按规模降序。
    """
    if G.number_of_edges() == 0:
        return [set(G.nodes())] if G.number_of_nodes() else []
    if method == "louvain":
        try:
            import community as community_louvain  # python-louvain

            part = community_louvain.best_partition(G, weight="weight")
            comms: dict[int, set] = {}
            for node, cid in part.items():
                comms.setdefault(cid, set()).add(node)
            return sorted(comms.values(), key=len, reverse=True)
        except ImportError as e:
            raise ImportError("louvain 需要安装 python-louvain") from e
    comms = community.greedy_modularity_communities(G, weight="weight")
    return sorted([set(c) for c in comms], key=len, reverse=True)


def network_density(G: nx.Graph) -> float:
    """二部图密度（实际边数 / 可能的最大边数）。"""
    policies = [n for n, d in G.nodes(data=True) if d.get("kind") == "policy"]
    hotspots = [n for n, d in G.nodes(data=True) if d.get("kind") == "hotspot"]
    possible = len(policies) * len(hotspots)
    if possible == 0:
        return 0.0
    return G.number_of_edges() / possible


def compare_time_slices(graphs: dict[str, nx.Graph]) -> dict[str, dict]:
    """时间切片对比：输入 {切片标签: 图}，输出各切片的规模/密度/孤立节点数。"""
    summary: dict[str, dict] = {}
    for label, G in graphs.items():
        n_policy = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "policy")
        n_hot = sum(1 for _, d in G.nodes(data=True) if d.get("kind") == "hotspot")
        summary[label] = {
            "n_policy": n_policy,
            "n_hotspot": n_hot,
            "n_edges": G.number_of_edges(),
            "density": network_density(G),
            "isolated": len(list(nx.isolates(G))),
        }
    return summary
