"""惠州市"十五五"政策语义网络与社会热点分析平台 — Streamlit 交互仪表盘。

运行：streamlit run app.py
数据：默认读取 data/samples/ 下的合成演示数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

from src.analytics.temporal_analysis import (
    burst_detection,
    daily_counts,
    moving_average,
    normalize_heat_per_platform,
    rank_percentiles,
)
from src.analytics.topic_gap import build_distribution, gap_table, js_divergence
from src.modeling.network_builder import (
    build_bipartite_graph,
    detect_communities,
    network_density,
    weighted_degree_ranking,
)
from src.modeling.semantic_matcher import SemanticMatcher, tfidf_cosine_baseline
from src.modeling.topic_model import TfidfTopicModel
from src.preprocessing.text_cleaner import tokenize
from src.utils.config import load_config, project_root

st.set_page_config(page_title="惠州政策 · 社会热点实验室", layout="wide")
st.title('惠州市"十五五"政策语义网络与社会热点分析平台')
st.caption("Huizhou Policy & Public Attention Lab · 演示数据为合成样本，仅展示分析流程")

cfg = load_config()
ROOT = project_root()
TH = cfg["matching"]["similarity_threshold"]
TOPK = cfg["matching"]["top_k"]
SEED = cfg["seed"]
N_CLUSTERS = cfg["clustering"]["n_clusters"]


@st.cache_data
def load_samples():
    policy = pd.read_csv(ROOT / "data/samples/sample_policy_segments.csv")
    hot = pd.read_csv(ROOT / "data/samples/sample_hotspot_snapshots.csv")
    hot["captured_at"] = pd.to_datetime(hot["captured_at"])
    return policy, hot


@st.cache_data
def topic_modeling(texts, n_clusters):
    model = TfidfTopicModel(max_features=3000)
    model.fit(texts)
    labels = model.fit_kmeans(n_clusters=n_clusters, seed=SEED)
    sil = model.evaluate_silhouette(labels)
    kw = model.cluster_keywords(labels, n=8)
    return labels, sil, kw


@st.cache_data
def compute_similarity(_policy_texts, _hotspot_titles, use_embeddings: bool):
    """返回 (N_policy x M_hotspot) 相似度矩阵。句向量失败时回退 TF-IDF 基线。"""
    if use_embeddings:
        try:
            m = SemanticMatcher(
                model_name=cfg["model"]["name"],
                threshold=TH,
                top_k=TOPK,
                device=cfg["model"]["device"],
            )
            return m.similarity_matrix(_policy_texts, _hotspot_titles), "sentence-transformers"
        except Exception as e:  # noqa: BLE001 — 演示环境无模型时优雅降级
            st.warning(f"句向量模型加载失败（{e}），已回退到 TF-IDF 基线。")
    return tfidf_cosine_baseline(_policy_texts, _hotspot_titles), "tfidf-baseline"


policy_df, hot_df = load_samples()
policy_texts = policy_df["text"].tolist()
hot_titles = hot_df["title"].tolist()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["主题概览", "语义网络", "时间演化", "关注差异", "数据导出"]
)

# ---------------- 主题概览 ----------------
with tab1:
    st.subheader("政策文本主题结构（TF-IDF + KMeans）")
    labels, sil, keywords = topic_modeling(policy_texts, N_CLUSTERS)
    policy_df["topic"] = labels
    col1, col2 = st.columns(2)
    with col1:
        dist = policy_df["topic"].value_counts().sort_index().reset_index()
        dist.columns = ["topic", "count"]
        fig = px.bar(dist, x="topic", y="count", title="各主题政策段落数",
                     labels={"topic": "主题", "count": "段落数"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("轮廓系数 Silhouette", f"{sil:.3f}")
        st.write("每主题代表性关键词：")
        for t in sorted(keywords):
            st.write(f"**主题 {t}**：{'、'.join(keywords[t][:6])}")

# ---------------- 语义网络 ----------------
with tab2:
    st.subheader("政策—热点语义二部网络")
    use_emb = st.checkbox("使用句向量（需下载模型；取消则用 TF-IDF 基线）", value=False)
    sim, method = compute_similarity(policy_texts, hot_titles, use_emb)
    st.caption(f"相似度方法：{method} · 阈值 {TH} · Top-K {TOPK}")

    pids = policy_df["segment_id"].tolist()
    hids = [f"{r.snapshot_id}" for r in hot_df.itertuples()]
    G = build_bipartite_graph(pids, hids, sim, threshold=TH)
    c1, c2, c3 = st.columns(3)
    c1.metric("政策节点", sum(1 for _, d in G.nodes(data=True) if d["kind"] == "policy"))
    c2.metric("热点节点", sum(1 for _, d in G.nodes(data=True) if d["kind"] == "hotspot"))
    c3.metric("网络密度", f"{network_density(G):.3f}")

    # Plotly 交互网络
    pos = nx.spring_layout(G, seed=SEED, k=0.6)
    edge_x, edge_y, edge_w = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(d["weight"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="rgba(150,150,150,0.5)"),
                             hoverinfo="none", showlegend=False))
    for kind, color, name in [("policy", "#1f77b4", "政策段落"), ("hotspot", "#ff7f0e", "社会热点")]:
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers", name=name,
            marker=dict(size=12, color=color),
            text=nodes, hoverinfo="text"))
    fig.update_layout(title="政策—热点语义网络（边 = 相似度 ≥ 阈值）",
                      showlegend=True, height=550,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True)

    st.write("政策段落加权度排名（连接热点的强度）：")
    rank = weighted_degree_ranking(G, kind="policy")[:10]
    st.dataframe(pd.DataFrame(rank, columns=["segment_id", "weighted_degree"]))

# ---------------- 时间演化 ----------------
with tab3:
    st.subheader("热点时间演化与爆发检测")
    # 用 TF-IDF 把热点标题映射到政策主题（演示用；生产可用句向量）
    tm = TfidfTopicModel(max_features=3000).fit(policy_texts)
    hot_vecs = tm.vectorizer.transform(hot_titles)
    centroids = []
    for c in sorted(set(labels)):
        rows = tm.tfidf_matrix[np.asarray(labels) == c]
        centroids.append(np.asarray(rows.mean(axis=0)).ravel())
    centroids = np.vstack(centroids)
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    hot_topics = _cos(hot_vecs, centroids).argmax(axis=1)
    hot_df["topic"] = hot_topics

    topic_sel = st.selectbox("选择主题", sorted(set(labels)))
    dc = daily_counts(hot_df[hot_df["topic"] == topic_sel].assign(topic=topic_sel))
    if dc.empty:
        st.info("该主题暂无热点。")
    else:
        daily = dc.groupby("date")["count"].sum().reset_index().sort_values("date")
        daily["ma7"] = moving_average(daily["count"], window=cfg["temporal"]["window_days"])
        burst = burst_detection(daily, window=cfg["temporal"]["window_days"],
                                z_threshold=cfg["temporal"]["burst_z_threshold"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["count"],
                                 mode="lines+markers", name="日计数"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["ma7"],
                                 mode="lines", name="7日滑动平均"))
        b = burst[burst["is_burst"]]
        if not b.empty:
            fig.add_trace(go.Scatter(x=b["date"], y=b["count"], mode="markers",
                                     name="爆发点", marker=dict(size=12, color="red", symbol="x")))
        fig.update_layout(title=f"主题 {topic_sel} 热点日计数与爆发检测",
                          xaxis_title="日期", yaxis_title="热点数")
        st.plotly_chart(fig, use_container_width=True)

    st.write("分平台热度（各平台独立标准化，**不跨平台求和**）：")
    hn = normalize_heat_per_platform(hot_df)
    rp = rank_percentiles(hot_df)
    st.dataframe(hn[["snapshot_id", "platform", "title", "raw_heat", "heat_norm"]]
                 .sort_values(["platform", "heat_norm"], ascending=[True, False]).head(15))

# ---------------- 关注差异 ----------------
with tab4:
    st.subheader("政策议程 vs 公众注意力：主题分布差异")
    p_dist = build_distribution(policy_df["topic"], N_CLUSTERS)
    h_dist = build_distribution(hot_df["topic"], N_CLUSTERS)
    jsd = js_divergence(p_dist, h_dist)
    st.metric("Jensen-Shannon 散度（0=完全一致，上界=1）", f"{jsd:.3f}")
    topic_names = [f"主题{i}" for i in range(N_CLUSTERS)]
    gt = gap_table(p_dist, h_dist, topic_names)
    fig = px.bar(gt, x="topic", y=["policy_share", "hotspot_share"],
                 barmode="group", title="政策 vs 热点主题分布",
                 labels={"value": "占比", "variable": "来源"})
    st.plotly_chart(fig, use_container_width=True)
    st.write("关注差异表（gap = 热点占比 − 政策占比，正值表示公众更关注）：")
    st.dataframe(gt)

# ---------------- 数据导出 ----------------
with tab5:
    st.subheader("数据导出")
    st.download_button("下载政策片段 CSV",
                       policy_df.to_csv(index=False).encode("utf-8-sig"),
                       "policy_segments.csv", "text/csv")
    st.download_button("下载热点快照 CSV",
                       hot_df.to_csv(index=False).encode("utf-8-sig"),
                       "hotspot_snapshots.csv", "text/csv")
    if "sim" in dir():
        sim_df = pd.DataFrame(sim, index=pids, columns=hids).reset_index(names="segment_id")
        st.download_button("下载相似度矩阵 CSV",
                           sim_df.to_csv(index=False).encode("utf-8-sig"),
                           "similarity_matrix.csv", "text/csv")
    st.caption("导出数据为演示样本；真实数据请见 docs/data_dictionary.md 的 schema 说明。")
