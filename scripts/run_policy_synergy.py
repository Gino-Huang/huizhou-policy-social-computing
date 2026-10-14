"""政策协同网络分析（原计划选题3主线）：关键词共现网络 + 主题关联网络。

输入：data/processed/passages_v2.parquet（863段落）、policy_topics_v2.parquet（15主题标签）
输出：
  data/processed/synergy_keyword_edges.csv / synergy_keyword_nodes.csv
  data/processed/synergy_topic_edges.csv / synergy_summary.json
  reports/figures/fig4_keyword_network.svg / fig5_topic_network.svg

运行：.venv/bin/python scripts/run_policy_synergy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.synergy import (  # noqa: E402
    build_keyword_cooccurrence,
    build_topic_graph,
    centrality_report,
    summarize_communities,
    top_keywords_per_doc,
)
from src.modeling.topic_model import TfidfTopicModel  # noqa: E402
from src.utils.config import load_config  # noqa: E402

TOPIC_NAMES = [
    "企业人才", "教育就业", "科技创新", "海洋旅游", "新材料智能",
    "工程规划", "城乡建设", "养老医疗", "投资改革", "法治思想",
    "安全涉外", "经济增长", "绿色消费", "行政执法", "市场立法",
]

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    seed = cfg.get("seed", 42)
    proc = PROJECT_ROOT / "data" / "processed"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    passages = pd.read_parquet(proc / "passages_v2.parquet")
    topics = pd.read_parquet(proc / "policy_topics_v2.parquet")
    labels = topics.set_index("passage_id")["cluster"]
    texts = passages.set_index("passage_id").loc[topics["passage_id"]]["text"].tolist()
    y = topics["cluster"].to_numpy()
    print(f"passages={len(texts)}, topics={len(set(y))}")

    # ---------- TF-IDF ----------
    tm = TfidfTopicModel(max_features=5000)
    tm.fit(texts)
    X = tm.tfidf_matrix
    keywords_per_doc = top_keywords_per_doc(tm.vectorizer, X, top_n=8)

    # ---------- 1. 关键词共现网络 ----------
    Gkw = build_keyword_cooccurrence(keywords_per_doc, min_df=3, min_weight=2)
    print(f"keyword graph: nodes={Gkw.number_of_nodes()}, edges={Gkw.number_of_edges()}")
    kw_cent = centrality_report(Gkw, top_n=15)
    kw_comms = summarize_communities(Gkw, {n: n for n in Gkw.nodes()})

    kw_nodes = pd.DataFrame([
        {"keyword": n, "df": d["df"],
         "degree_centrality": kw_cent.set_index("node").loc[n, "degree_centrality"]
         if n in set(kw_cent["node"]) else 0.0,
         "betweenness": kw_cent.set_index("node").loc[n, "betweenness"]
         if n in set(kw_cent["node"]) else 0.0}
        for n, d in Gkw.nodes(data=True)
    ]).sort_values("betweenness", ascending=False)
    kw_edges = pd.DataFrame(
        [{"source": a, "target": b, "weight": d["weight"]} for a, b, d in Gkw.edges(data=True)]
    ).sort_values("weight", ascending=False)
    kw_nodes.to_csv(proc / "synergy_keyword_nodes.csv", index=False, encoding="utf-8-sig")
    kw_edges.to_csv(proc / "synergy_keyword_edges.csv", index=False, encoding="utf-8-sig")

    # ---------- 2. 主题关联网络 ----------
    centroids = np.vstack([
        np.asarray(X[y == c].mean(axis=0)).ravel() for c in range(15)
    ])
    Gt = build_topic_graph(centroids, TOPIC_NAMES, keep="top_k", top_k=30)
    print(f"topic graph: nodes={Gt.number_of_nodes()}, edges={Gt.number_of_edges()}")
    t_cent = centrality_report(Gt, top_n=15)
    t_comms = summarize_communities(Gt, {i: TOPIC_NAMES[i] for i in range(15)})

    t_edges = pd.DataFrame(
        [{"source": TOPIC_NAMES[a], "target": TOPIC_NAMES[b], "weight": round(d["weight"], 4)}
         for a, b, d in Gt.edges(data=True)]
    ).sort_values("weight", ascending=False)
    t_edges.to_csv(proc / "synergy_topic_edges.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_passages": len(texts),
        "n_topics": 15,
        "topic_names": TOPIC_NAMES,
        "keyword_graph": {
            "nodes": Gkw.number_of_nodes(), "edges": Gkw.number_of_edges(),
            "top_betweenness": kw_cent[["node", "betweenness"]].round(4).to_dict("records"),
            "communities": kw_comms,
        },
        "topic_graph": {
            "nodes": Gt.number_of_nodes(), "edges": Gt.number_of_edges(),
            "betweenness_ranking": [
                {"topic": TOPIC_NAMES[int(r["node"])], "betweenness": round(float(r["betweenness"]), 4)}
                for _, r in t_cent.iterrows()
            ],
            "strongest_links": t_edges.head(10).to_dict("records"),
            "communities": t_comms,
        },
    }
    (proc / "synergy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 3. 图 ----------
    # 关键词网络：取 betweenness top 60 节点绘制
    top_nodes = kw_cent["node"].tolist()[:60]
    H = Gkw.subgraph(top_nodes).copy()
    pos = nx.spring_layout(H, seed=seed, k=0.9)
    btw = nx.betweenness_centrality(H, weight="weight", seed=seed)
    sizes = [300 + 3000 * btw[n] for n in H.nodes()]
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.7, ax=ax)
    nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color="#4C9BE8",
                           alpha=0.85, ax=ax)
    nx.draw_networkx_labels(H, pos, font_size=9, ax=ax)
    ax.set_title("政策关键词共现网络（节点大小=中介中心性，top 60 关键词）")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_keyword_network.svg")
    plt.close(fig)

    # 主题关联网络
    pos2 = nx.spring_layout(Gt, seed=seed, k=1.2)
    btw2 = nx.betweenness_centrality(Gt, weight="weight", seed=seed)
    sizes2 = [800 + 6000 * btw2[n] for n in Gt.nodes()]
    widths = [0.5 + 4 * d["weight"] for _, _, d in Gt.edges(data=True)]
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(Gt, pos2, alpha=0.35, width=widths, ax=ax)
    nx.draw_networkx_nodes(Gt, pos2, node_size=sizes2, node_color="#E8844C",
                           alpha=0.9, ax=ax)
    nx.draw_networkx_labels(Gt, pos2, labels={i: TOPIC_NAMES[i] for i in Gt.nodes()},
                            font_size=10, ax=ax)
    ax.set_title("政策主题关联网络（边权=TF-IDF中心向量余弦相似度，节点大小=中介中心性）")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_topic_network.svg")
    plt.close(fig)

    print("saved: synergy_keyword_nodes/edges.csv, synergy_topic_edges.csv, synergy_summary.json")
    print("figures: fig4_keyword_network.svg, fig5_topic_network.svg")
    print("\n主题中介中心性 top5（潜在协同关键节点）:")
    for r in summary["topic_graph"]["betweenness_ranking"][:5]:
        print(f"  {r['topic']}: {r['betweenness']}")


if __name__ == "__main__":
    main()
