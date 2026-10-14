"""Milestone-2: 在真实数据上运行完整算法链路（RQ1→RQ3）。

输入:
  data/processed/text_segments.parquet   政策文本段（4621）
  data/raw/hotspots/snapshot_*.csv       热点快照（60 条）
  configs/config.yaml                    阈值 / top_k / 模型名

输出:
  data/processed/semantic_edges.parquet
  data/processed/policy_topics.parquet   (segment_id, cluster)
  data/processed/run_summary.json
  reports/figures/fig1_topic_overview.png
  reports/figures/fig2_match_scores.png
  reports/figures/fig3_bipartite_network.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# huggingface_hub 经 httpx 读取代理环境变量时，会把 no_proxy 里
# 的括号 IPv6 条目（如 [::1]）解析崩掉；收敛为纯主机名写法。
# 该代理仅用于 localhost 直连，收紧不影响外网下载走代理。
import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.text_cleaner import clean_text  # noqa: E402
from src.preprocessing.deduplicator import exact_dedup  # noqa: E402
from src.modeling.topic_model import TfidfTopicModel  # noqa: E402
from src.modeling.semantic_matcher import SemanticMatcher, tfidf_cosine_baseline  # noqa: E402
from src.modeling import network_builder as nb  # noqa: E402
from src.analytics.topic_gap import build_distribution, js_divergence, gap_table  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# PDF 提取残留的版式噪音：公报刊头、页码、纯数字行
_BOILERPLATE_RES = [
    re.compile(r"惠州市人民政府公报"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^\d{1,4}$"),
    re.compile(r"^[—–-]{2,}$"),
]


def is_boilerplate(text: str) -> bool:
    return any(p.search(text) for p in _BOILERPLATE_RES)


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    seed = cfg.get("seed", 42)
    threshold = float(cfg["matching"]["similarity_threshold"])
    top_k = int(cfg["matching"]["top_k"])
    model_name = cfg["model"]["name"]

    out_dir = PROJECT_ROOT / "data" / "processed"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 政策文本清洗 ----------
    segs = pd.read_parquet(out_dir / "text_segments.parquet")
    print(f"raw segments: {len(segs)}")
    segs["clean"] = segs["text"].astype(str).map(clean_text)
    segs = segs[segs["clean"].str.len() >= 6].copy()
    segs = segs[~segs["clean"].map(is_boilerplate)].copy()
    segs, n_dup = exact_dedup(segs, text_col="clean")
    print(f"after clean+dedup: {len(segs)} (exact dups removed: {n_dup})")

    texts = segs["clean"].tolist()
    seg_ids = segs["segment_id"].astype(str).tolist()

    # ---------- 2. TF-IDF 主题建模 (RQ1) ----------
    tm = TfidfTopicModel(max_features=8000, min_df=2)
    tm.fit(texts)
    sil = {}
    for k in range(8, 16):
        labels = tm.fit_kmeans(n_clusters=k, seed=seed)
        sil[k] = tm.evaluate_silhouette(labels)
        print(f"k={k}: silhouette={sil[k]:.4f}")
    best_k = max(sil, key=sil.get)
    print(f"chosen k={best_k}")
    labels = tm.fit_kmeans(n_clusters=best_k, seed=seed)
    kw = tm.cluster_keywords(labels, n=10)
    dist = tm.doc_topic_distribution(labels)

    segs["cluster"] = labels
    segs[["segment_id", "cluster"]].to_parquet(out_dir / "policy_topics.parquet", index=False)

    topic_names = [f"主题{i}: {'/'.join(kw[i][:3])}" for i in range(best_k)]

    # ---------- 3. 热点快照 ----------
    snap_files = sorted((PROJECT_ROOT / "data" / "raw" / "hotspots").glob("snapshot_*.csv"))
    print("snapshots:", [p.name for p in snap_files])
    hot = pd.concat([pd.read_csv(p) for p in snap_files], ignore_index=True)
    hot["hid"] = hot["platform"].astype(str) + "__" + hot.index.astype(str)
    hot_titles = hot["title"].astype(str).tolist()
    print(f"hotspots: {len(hot)}")

    # ---------- 4. 语义匹配 ----------
    matcher = SemanticMatcher(model_name=model_name, threshold=threshold, top_k=top_k)
    print("encoding policy segments (triggers model download on first run)...")
    sim = matcher.similarity_matrix(texts, hot_titles)
    print("sim matrix:", sim.shape, "max:", float(sim.max()), "mean:", float(sim.mean()))
    edges = matcher.match(seg_ids, hot["hid"].tolist(), sim)
    edges["model_version"] = model_name
    edges.rename(columns={"policy_id": "policy_segment_id", "hotspot_id": "hotspot_id"},
                 inplace=True)
    edges.to_parquet(out_dir / "semantic_edges.parquet", index=False)

    matched_edges = edges[edges["policy_segment_id"].notna()].copy()
    n_hot_matched = matched_edges["hotspot_id"].nunique()
    print(f"edges: {len(matched_edges)}, hotspots matched: {n_hot_matched}/{len(hot)}")

    # TF-IDF 基线对比
    base_sim = tfidf_cosine_baseline(texts, hot_titles)
    agree = 0
    for j in range(len(hot)):
        top_sem = int(np.argmax(sim[:, j]))
        top_base = int(np.argmax(base_sim[:, j]))
        if top_sem == top_base:
            agree += 1
    print(f"top-1 agreement (semantic vs tfidf): {agree}/{len(hot)} = {agree/len(hot):.2%}")
    print(f"mean best sim (semantic): {sim.max(axis=0).mean():.3f}, "
          f"(tfidf): {base_sim.max(axis=0).mean():.3f}")

    # ---------- 5. 二部网络 ----------
    G = nb.build_bipartite_graph(seg_ids, hot["hid"].tolist(), sim, threshold=threshold)
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"density={nb.network_density(G):.5f}")
    comms = nb.detect_communities(G)
    print(f"communities: {len(comms)}, sizes: {sorted([len(c) for c in comms], reverse=True)[:8]}")

    # 按主题聚合加权度
    seg2cluster = dict(zip(seg_ids, labels))
    topic_strength: dict[int, float] = {}
    for u, v, d in G.edges(data=True):
        # u/v 其中之一是 policy segment
        for node in (u, v):
            if node in seg2cluster:
                topic_strength[seg2cluster[node]] = topic_strength.get(seg2cluster[node], 0.0) + float(d["weight"])
    topic_rank = sorted(topic_strength.items(), key=lambda x: x[1], reverse=True)

    # ---------- 6. 关注差异 (RQ3) ----------
    # 热点 → 其最佳匹配段落的主题
    best = matched_edges.sort_values("similarity", ascending=False).drop_duplicates("hotspot_id")
    hid2cluster = {}
    for _, r in best.iterrows():
        hid2cluster[r["hotspot_id"]] = int(seg2cluster[str(r["policy_segment_id"])])
    hot_labels = [hid2cluster[h] for h in hot["hid"] if h in hid2cluster]
    n_hot_unmatched = len(hot) - len(hot_labels)

    p_policy = build_distribution(pd.Series(labels), best_k)
    p_hot = build_distribution(pd.Series(hot_labels), best_k)
    jsd = js_divergence(p_policy, p_hot)
    gap = gap_table(p_policy, p_hot, topic_names)
    print(f"JSD(policy || hotspot) = {jsd:.4f}; hotspots without match: {n_hot_unmatched}")

    # ---------- 7. 图表 ----------
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # fig1: 主题规模 + 关键词
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = pd.Series(labels).value_counts().sort_index()
    ax.barh([topic_names[i] for i in counts.index], counts.values)
    ax.set_xlabel("段落数")
    ax.set_title(f"政策主题分布 (k={best_k}, silhouette={sil[best_k]:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_topic_overview.png", dpi=150)
    plt.close(fig)

    # fig2: 每个热点最佳匹配分数直方图
    fig, ax = plt.subplots(figsize=(9, 5))
    best_scores = sim.max(axis=0)
    ax.hist(best_scores, bins=30, edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--", label=f"阈值 {threshold}")
    ax.set_xlabel("最佳匹配相似度")
    ax.set_ylabel("热点数")
    ax.set_title("热点-政策最佳匹配分数分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_match_scores.png", dpi=150)
    plt.close(fig)

    # fig3: 二部网络（仅含匹配节点）
    import networkx as nx
    H = G.copy()
    isolates = list(nx.isolates(H))
    H.remove_nodes_from(isolates)
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(H, seed=seed, k=0.6)
    pol_nodes = [n for n, d in H.nodes(data=True) if d.get("kind") == "policy"]
    hot_nodes = [n for n, d in H.nodes(data=True) if d.get("kind") == "hotspot"]
    colors = [seg2cluster[n] % 10 for n in pol_nodes]
    nx.draw_networkx_nodes(H, pos, nodelist=pol_nodes, node_size=28, node_color=colors,
                           cmap="tab10", ax=ax, label="政策段落")
    nx.draw_networkx_nodes(H, pos, nodelist=hot_nodes, node_size=60, node_color="red",
                           alpha=0.8, ax=ax, label="热点")
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.6, ax=ax)
    ax.set_title(f"政策-热点语义二部图 (边数={H.number_of_edges()}, 节点数={H.number_of_nodes()})")
    ax.axis("off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_bipartite_network.png", dpi=150)
    plt.close(fig)

    # ---------- 8. 汇总 ----------
    top_matches = best.sort_values("similarity", ascending=False).head(10)
    examples = []
    for _, r in top_matches.iterrows():
        htitle = hot.loc[hot["hid"] == r["hotspot_id"], "title"].iloc[0]
        ptext = segs.loc[segs["segment_id"].astype(str) == str(r["policy_segment_id"]), "clean"].iloc[0]
        examples.append({
            "hotspot": htitle,
            "policy_segment": ptext[:80],
            "similarity": round(float(r["similarity"]), 3),
            "policy_topic": topic_names[int(seg2cluster[str(r["policy_segment_id"])])],
        })

    summary = {
        "n_segments_raw": 4621,
        "n_segments_used": len(segs),
        "n_hotspots": len(hot),
        "k": best_k,
        "silhouette": {str(k): round(v, 4) for k, v in sil.items()},
        "cluster_keywords": {str(i): kw[i] for i in range(best_k)},
        "cluster_sizes": {str(i): int((labels == i).sum()) for i in range(best_k)},
        "model": model_name,
        "threshold": threshold,
        "top_k": top_k,
        "n_edges": len(matched_edges),
        "n_hotspots_matched": int(n_hot_matched),
        "n_hotspots_unmatched": int(n_hot_unmatched),
        "graph_density": round(nb.network_density(G), 5),
        "n_communities": len(comms),
        "community_sizes": sorted([len(c) for c in comms], reverse=True),
        "top1_agreement_semantic_vs_tfidf": round(agree / len(hot), 3),
        "mean_best_sim_semantic": round(float(sim.max(axis=0).mean()), 3),
        "mean_best_sim_tfidf": round(float(base_sim.max(axis=0).mean()), 3),
        "jsd": round(jsd, 4),
        "topic_strength_ranking": [
            {"topic": topic_names[i], "strength": round(s, 3)} for i, s in topic_rank
        ],
        "gap_table": gap.round(4).to_dict(orient="records"),
        "top_matches": examples,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary saved:", out_dir / "run_summary.json")
    print("figures saved:", fig_dir)


if __name__ == "__main__":
    main()
