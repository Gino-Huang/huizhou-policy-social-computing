"""Milestone-2 第三轮：BGE 模型重跑语义匹配链路。

背景（reports/model_shootout_20261014.md）：
  120 对人工标注证明 paraphrase-multilingual-MiniLM-L12-v2 在本任务上
  不可靠（所有 ≥0.60 候选全被标为 0），对决裁决将默认模型切换为
  BAAI/bge-small-zh-v1.5、阈值 0.55 → 0.45（configs/config.yaml）。
  但 v2 的全部语义产物仍是旧模型算的——本轮用 BGE 重跑，抹平这一缺口。

与 v2 的差异：
  - 语义模型：configs/config.yaml 的 model.name / matching.similarity_threshold
  - 热点快照：只用最新一份（snapshot_20261015T012727Z.csv），不再拼接全部快照
  - 主题建模不重跑：TF-IDF+KMeans 与句向量无关，直接复用 policy_topics_v2.parquet
    与 run_summary_v2.json 的簇关键词（保证主题命名一致、可比）

输入:
  data/processed/passages_v2.parquet
  data/processed/policy_topics_v2.parquet
  data/processed/run_summary_v2.json   (取簇关键词用于主题命名)
  data/raw/hotspots/snapshot_20261015T012727Z.csv
  configs/config.yaml

输出（均带 _v3 后缀，不覆盖 v1/v2）:
  data/processed/semantic_edges_v3.parquet
  data/processed/run_summary_v3.json
  reports/figures/fig2c_match_scores.svg
  reports/figures/fig3c_bipartite_network.svg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.semantic_matcher import SemanticMatcher, tfidf_cosine_baseline  # noqa: E402
from src.modeling import network_builder as nb  # noqa: E402
from src.analytics.topic_gap import build_distribution, js_divergence, gap_table  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    seed = cfg.get("seed", 42)
    threshold = float(cfg["matching"]["similarity_threshold"])
    top_k = int(cfg["matching"]["top_k"])
    model_name = cfg["model"]["name"]
    print(f"model={model_name}, threshold={threshold}, top_k={top_k}")

    out_dir = PROJECT_ROOT / "data" / "processed"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 段落与主题（复用 v2，TF-IDF 与句向量无关） ----------
    passages = pd.read_parquet(out_dir / "passages_v2.parquet")
    use = passages[passages["used_in_matching"]].copy().reset_index(drop=True)
    texts = use["text"].tolist()
    pids = use["passage_id"].tolist()
    print(f"passages used in matching: {len(use)}")

    topics = pd.read_parquet(out_dir / "policy_topics_v2.parquet")
    pid2cluster = dict(zip(topics["passage_id"], (int(x) for x in topics["cluster"])))
    labels = np.array([pid2cluster[p] for p in pids])
    v2sum = json.load(open(out_dir / "run_summary_v2.json", encoding="utf-8"))
    kw = v2sum["cluster_keywords"]
    best_k = v2sum["k"]
    topic_names = [f"主题{i}: {'/'.join(kw[str(i)][:3])}" for i in range(best_k)]
    print(f"topics reused: k={best_k}, silhouette(v2)={v2sum['silhouette'][str(best_k)]}")

    # ---------- 2. 热点快照（只用最新一份） ----------
    snap = sorted((PROJECT_ROOT / "data" / "raw" / "hotspots").glob("snapshot_*.csv"))[-1]
    print("snapshot:", snap.name)
    hot = pd.read_csv(snap)
    hot = hot.reset_index(drop=True)
    hot["hid"] = hot["platform"].astype(str) + "__" + hot.index.astype(str)
    hot_titles = hot["title"].astype(str).tolist()
    print(f"hotspots: {len(hot)} ({hot['platform'].value_counts().to_dict()})")

    # ---------- 3. BGE 语义匹配 ----------
    matcher = SemanticMatcher(model_name=model_name, threshold=threshold, top_k=top_k)
    print("encoding passages (BGE should be cached)...")
    sim = matcher.similarity_matrix(texts, hot_titles)
    print("sim matrix:", sim.shape, "max:", round(float(sim.max()), 3),
          "mean:", round(float(sim.mean()), 3))
    edges = matcher.match(pids, hot["hid"].tolist(), sim)
    edges["model_version"] = model_name
    edges.rename(columns={"policy_id": "passage_id", "hotspot_id": "hotspot_id"}, inplace=True)
    edges.to_parquet(out_dir / "semantic_edges_v3.parquet", index=False)

    matched_edges = edges[edges["passage_id"].notna()].copy()
    n_hot_matched = matched_edges["hotspot_id"].nunique()
    print(f"edges: {len(matched_edges)}, hotspots matched: {n_hot_matched}/{len(hot)}")

    base_sim = tfidf_cosine_baseline(texts, hot_titles)
    agree = sum(int(np.argmax(sim[:, j])) == int(np.argmax(base_sim[:, j]))
                for j in range(len(hot)))
    print(f"top-1 agreement (semantic vs tfidf): {agree}/{len(hot)} = {agree/len(hot):.2%}")
    print(f"mean best sim (semantic): {sim.max(axis=0).mean():.3f}, "
          f"(tfidf): {base_sim.max(axis=0).mean():.3f}")

    # ---------- 3b. 伪影核查 ----------
    best_scores = sim.max(axis=0)
    print(f"[artifact-check] max best score = {best_scores.max():.3f} "
          f"(v2 旧模型曾出现 0.839 离群高分；BGE 全矩阵最大值应 < 0.60)")
    print(f"[artifact-check] n hotspots with best >= 0.60: {(best_scores >= 0.60).sum()}")
    top_j = np.argsort(best_scores)[::-1][:10]
    print("[artifact-check] top-10 best matches:")
    for j in top_j:
        i = int(np.argmax(sim[:, j]))
        print(f"  {best_scores[j]:.3f} | {hot_titles[j][:30]} -> {texts[i][:45]}")

    # ---------- 4. 二部网络 ----------
    G = nb.build_bipartite_graph(pids, hot["hid"].tolist(), sim, threshold=threshold)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"density={nb.network_density(G):.5f}")
    comms = nb.detect_communities(G)
    print(f"communities: {len(comms)}, sizes: {sorted([len(c) for c in comms], reverse=True)[:8]}")

    topic_strength: dict[int, float] = {}
    for u, v, d in G.edges(data=True):
        for node in (u, v):
            if node in pid2cluster:
                topic_strength[pid2cluster[node]] = topic_strength.get(pid2cluster[node], 0.0) + float(d["weight"])
    topic_rank = sorted(topic_strength.items(), key=lambda x: x[1], reverse=True)

    # ---------- 5. 关注差异 (RQ3) ----------
    best = matched_edges.sort_values("similarity", ascending=False).drop_duplicates("hotspot_id")
    hid2cluster = {r["hotspot_id"]: int(pid2cluster[str(r["passage_id"])])
                   for _, r in best.iterrows()}
    hot_labels = [hid2cluster[h] for h in hot["hid"] if h in hid2cluster]
    n_hot_unmatched = len(hot) - len(hot_labels)

    p_policy = build_distribution(pd.Series(labels), best_k)
    p_hot = build_distribution(pd.Series(hot_labels), best_k)
    jsd = js_divergence(p_policy, p_hot)
    gap = gap_table(p_policy, p_hot, topic_names)
    print(f"JSD(policy || hotspot) = {jsd:.4f}; hotspots without match: {n_hot_unmatched}")

    # ---------- 6. 图表（SVG，文本型以便 GitHub 直接推送） ----------
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(best_scores, bins=30, edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--", label=f"阈值 {threshold}")
    ax.set_xlabel("最佳匹配相似度")
    ax.set_ylabel("热点数")
    ax.set_title(f"热点-政策最佳匹配分数分布 v3（BGE，{snap.stem}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2c_match_scores.svg")
    plt.close(fig)

    import networkx as nx
    H = G.copy()
    H.remove_nodes_from(list(nx.isolates(H)))
    # 图注：GitHub 单文件推送上限 ~115KB，完整图（945 条边）过大；
    # 此处仅绘制每个热点 Top-3 边用于展示，全部边见 semantic_edges_v3.parquet。
    top3_edges = set()
    for _, r in matched_edges.sort_values("similarity", ascending=False).groupby("hotspot_id").head(3).iterrows():
        top3_edges.add((str(r["passage_id"]), str(r["hotspot_id"])))
    Hviz = H.edge_subgraph([e for e in H.edges() if (e[0], e[1]) in top3_edges or (e[1], e[0]) in top3_edges]).copy()
    Hviz.add_nodes_from([n for n in H.nodes() if H.degree(n) > 0 and n not in Hviz.nodes()])
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(Hviz, seed=seed, k=0.6)
    pol_nodes = [n for n, d in Hviz.nodes(data=True) if d.get("kind") == "policy"]
    hot_nodes = [n for n, d in Hviz.nodes(data=True) if d.get("kind") == "hotspot"]
    colors = [pid2cluster[n] % 10 for n in pol_nodes]
    nx.draw_networkx_nodes(Hviz, pos, nodelist=pol_nodes, node_size=28, node_color=colors,
                           cmap="tab10", ax=ax, label="政策段落")
    nx.draw_networkx_nodes(Hviz, pos, nodelist=hot_nodes, node_size=60, node_color="red",
                           alpha=0.8, ax=ax, label="热点")
    nx.draw_networkx_edges(Hviz, pos, alpha=0.25, width=0.6, ax=ax)
    ax.set_title(f"政策-热点语义二部图 v3 (BGE, 展示用Top-3/热点, 边数={Hviz.number_of_edges()})")
    ax.axis("off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3c_bipartite_network.svg")
    plt.close(fig)

    # ---------- 7. 汇总 ----------
    top_matches = best.sort_values("similarity", ascending=False).head(10)
    examples = []
    for _, r in top_matches.iterrows():
        htitle = hot.loc[hot["hid"] == r["hotspot_id"], "title"].iloc[0]
        hplat = hot.loc[hot["hid"] == r["hotspot_id"], "platform"].iloc[0]
        ptext = use.loc[use["passage_id"] == str(r["passage_id"]), "text"].iloc[0]
        examples.append({
            "hotspot": htitle,
            "platform": hplat,
            "policy_passage": ptext[:100],
            "passage_len": len(ptext),
            "similarity": round(float(r["similarity"]), 3),
            "policy_topic": topic_names[int(pid2cluster[str(r["passage_id"])])],
        })

    summary = {
        "run": "v3",
        "snapshot": snap.name,
        "n_passages_used": int(len(use)),
        "n_hotspots": len(hot),
        "hotspot_platforms": hot["platform"].value_counts().to_dict(),
        "k": best_k,
        "topics_reused_from": "v2 (TF-IDF+KMeans, 与句向量无关)",
        "model": model_name,
        "threshold": threshold,
        "top_k": top_k,
        "n_edges": len(matched_edges),
        "n_hotspots_matched": int(n_hot_matched),
        "n_hotspots_unmatched": int(n_hot_unmatched),
        "max_best_similarity": round(float(best_scores.max()), 3),
        "mean_best_sim_semantic": round(float(best_scores.mean()), 3),
        "mean_best_sim_tfidf": round(float(base_sim.max(axis=0).mean()), 3),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "graph_density": round(nb.network_density(G), 5),
        "n_communities": len(comms),
        "community_sizes": sorted([len(c) for c in comms], reverse=True),
        "top1_agreement_semantic_vs_tfidf": round(agree / len(hot), 3),
        "jsd": round(jsd, 4),
        "topic_strength_ranking": [
            {"topic": topic_names[i], "strength": round(s, 3)} for i, s in topic_rank
        ],
        "gap_table": gap.round(4).to_dict(orient="records"),
        "top_matches": examples,
    }
    with open(out_dir / "run_summary_v3.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary saved:", out_dir / "run_summary_v3.json")
    print("figures saved:", fig_dir)


if __name__ == "__main__":
    main()
