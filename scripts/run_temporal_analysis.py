"""时间演化分析（可重跑）：热点快照时间序列分析。

历史运行:
  v1 = 2026-10-19，6 天（10-14→10-19），图 fig6*，汇总 temporal_summary_20261019.json
  v2 = 2026-10-23，10 天（10-14→10-23），图 fig7*，汇总 temporal_summary_20261023.json
  v3 = 2026-10-27，14 天（10-14→10-27），图 fig8*，汇总 temporal_summary_20261027.json

输入:
  data/raw/hotspots/snapshot_*.csv      # 每日 × 60 条（微博30+百度30），自动 glob 全部
  data/processed/passages_v2.parquet    # 863 政策段落（BGE 匹配用）
  data/processed/policy_topics_v2.parquet  # 15 主题标签（TF-IDF+KMeans）
  configs/config.yaml                   # 模型 BAAI/bge-small-zh-v1.5, 阈值 0.45

输出:
  data/processed/temporal_summary_20261027.json  # 本地，gitignored
  reports/figures/fig8a_jsd_trend.svg
  reports/figures/fig8b_persistence.svg
  reports/figures/fig8c_platform_overlap.svg
  reports/figures/fig8d_bursts.svg（仅当检出爆发时生成）
  reports/temporal_analysis_v3_20261027.md  # 据本脚本输出撰写

方法说明见 reports/temporal_analysis_20261019.md（v1）与
reports/temporal_analysis_v2_20261023.md（v2）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

# 本次运行标记：v1（6 天，2026-10-19）用 fig6* / temporal_summary_20261019.json；
# v2（10 天，2026-10-23）用 fig7* / temporal_summary_20261023.json；
# v3（14 天，2026-10-27）用 fig8* / temporal_summary_20261027.json，避免覆盖旧产物。
RUN_TAG = "v3"
FIG_PREFIX = "fig8"
SUMMARY_NAME = "temporal_summary_20261027.json"

from src.analytics.temporal_analysis import (  # noqa: E402
    daily_platform_overlap,
    normalize_title,
    topic_persistence,
)
from src.analytics.topic_gap import build_distribution, js_divergence  # noqa: E402
from src.modeling.semantic_matcher import SemanticMatcher  # noqa: E402
from src.utils.config import load_config  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def load_snapshots(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("snapshot_*.csv"))
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["captured_at"]).dt.date
    df["raw_heat"] = pd.to_numeric(df["raw_heat"], errors="coerce").fillna(0).astype(int)
    df["norm_title"] = df["title"].map(normalize_title)
    df = df[df["norm_title"] != ""].copy()
    return df, files


def keyword_persistence(df: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    """关键词级持续性：jieba 分词后，统计每个词出现的天数与平台。

    标题级归一化过于严格（6 天里只有 1 个标题原样重复），关键词级才能
    看到同一事件/人物的跨日延续。返回列：keyword, n_days, n_appearances,
    platforms, first_date, last_date。按 n_days 降序。
    """
    import re

    import jieba

    stop = set(list("的了是在有和与或被把对就也都而及个中上为之以于"))
    kw = {}  # keyword -> {"days": set, "n": int, "platforms": set}
    for _, r in df.iterrows():
        toks = {
            t
            for t in jieba.cut(str(r["title"]))
            if len(t) >= min_len and t not in stop
        }
        for t in toks:
            e = kw.setdefault(t, {"days": set(), "n": 0, "platforms": set()})
            e["days"].add(str(r["date"]))
            e["n"] += 1
            e["platforms"].add(str(r["platform"]))
    rows = [
        {
            "keyword": k,
            "n_days": len(v["days"]),
            "n_appearances": v["n"],
            "platforms": ",".join(sorted(v["platforms"])),
            "first_date": min(v["days"]),
            "last_date": max(v["days"]),
        }
        for k, v in kw.items()
        if len(v["days"]) >= 2
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["n_days", "n_appearances"], ascending=[False, False]).reset_index(drop=True)


def burst_scan(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.DataFrame:
    """分 (话题, 平台) 的全程热度 z-score 爆发检测。

    缺席日热度记 0；只对出现 ≥2 天的话题计算（单日话题的 z 恒为
    sqrt(天数-1)，无信息量）。基线点数 = 快照天数，结果仅供参考。
    """
    dates = sorted(df["date"].unique())
    rows = []
    for (norm, plat), g in df.groupby(["norm_title", "platform"]):
        if g["date"].nunique() < 2:
            continue
        heat_by_date = {d: 0 for d in dates}
        for d, gg in g.groupby("date"):
            heat_by_date[d] = int(gg["raw_heat"].max())
        series = np.array([heat_by_date[d] for d in dates], dtype=float)
        mean, std = series.mean(), series.std(ddof=0)
        if std <= 0:
            continue
        z = (series - mean) / std
        for d, x, zz in zip(dates, series, z):
            if zz >= z_threshold:
                rows.append(
                    {
                        "norm_title": norm,
                        "platform": plat,
                        "date": d.isoformat(),
                        "heat": int(x),
                        "z_score": round(float(zz), 2),
                        "example_title": g["title"].iloc[0],
                        "n_days": int(g["date"].nunique()),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("z_score", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    model_name = cfg["model"]["name"]
    threshold = float(cfg["matching"]["similarity_threshold"])
    top_k = int(cfg["matching"]["top_k"])
    z_thr = float(cfg["temporal"]["burst_z_threshold"])

    out_dir = PROJECT_ROOT / "data" / "processed"
    fig_dir = PROJECT_ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 快照加载 ----------
    df, files = load_snapshots(PROJECT_ROOT / "data" / "raw" / "hotspots")
    dates = sorted(df["date"].unique())
    print(f"snapshots: {len(files)}, rows: {len(df)}, dates: {dates[0]} -> {dates[-1]}")

    # ---------- 2. 持续性 / 平台重合 / 爆发 ----------
    persist = topic_persistence(df)
    overlap = daily_platform_overlap(df)
    bursts = burst_scan(df, z_threshold=z_thr)
    multi = persist[persist["n_days"] >= 2].copy()
    kw_persist = keyword_persistence(df)
    print(f"distinct topics: {len(persist)}, multi-day: {len(multi)}")
    print(f"keywords on >=2 days: {len(kw_persist)}")
    print("top keywords by days:")
    print(kw_persist.head(15)[["keyword", "n_days", "platforms"]].to_string(index=False))
    print("platform overlap (jaccard):")
    print(overlap[["date", "n_overlap", "jaccard"]].to_string(index=False))
    print(f"bursts flagged: {len(bursts)}")
    if not bursts.empty:
        print(bursts[["example_title", "platform", "date", "z_score"]].head(10).to_string(index=False))

    # ---------- 3. 每日 BGE 匹配 → 主题分布 → JSD ----------
    passages = pd.read_parquet(out_dir / "passages_v2.parquet")
    topics = pd.read_parquet(out_dir / "policy_topics_v2.parquet")
    pid2cluster = dict(zip(topics["passage_id"].astype(str), topics["cluster"].astype(int)))
    p_policy = build_distribution(topics["cluster"], 15)

    v2 = json.load(open(out_dir / "run_summary_v2.json", encoding="utf-8"))
    topic_names = [f"主题{i}: {'/'.join(v2['cluster_keywords'][str(i)][:3])}" for i in range(15)]

    ptexts = passages["text"].astype(str).tolist()
    pids = passages["passage_id"].astype(str).tolist()

    matcher = SemanticMatcher(model_name=model_name, threshold=threshold, top_k=top_k)
    print("encoding 863 passages (cached model)...")
    pemb = matcher.encode(ptexts)

    daily_stats = []
    all_matched_clusters = []
    for d in dates:
        day = df[df["date"] == d].copy().reset_index(drop=True)
        titles = day["title"].astype(str).tolist()
        hemb = matcher.encode(titles)
        sim = (pemb @ hemb.T).astype(float)
        best_idx = sim.argmax(axis=0)
        best_sim = sim.max(axis=0)
        keep = best_sim >= threshold
        clusters = [pid2cluster[pids[int(i)]] for i in best_idx[keep]]
        all_matched_clusters.extend(clusters)
        n_matched = int(keep.sum())
        if clusters:
            p_hot = build_distribution(pd.Series(clusters), 15)
            jsd = js_divergence(p_policy, p_hot)
        else:
            jsd = float("nan")
        daily_stats.append(
            {
                "date": d.isoformat(),
                "n_hotspots": len(day),
                "n_matched": n_matched,
                "match_rate": round(n_matched / len(day), 3),
                "mean_best_sim": round(float(best_sim.mean()), 3),
                "jsd": round(float(jsd), 4) if not np.isnan(jsd) else None,
            }
        )
        print(f"{d}: matched {n_matched}/{len(day)}, JSD={daily_stats[-1]['jsd']}")

    # ---------- 4. 图 ----------
    n_days = len(dates)

    # 7a: JSD 趋势
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ds = pd.DataFrame(daily_stats)
    ax.plot(ds["date"], ds["jsd"], marker="o", label="JSD（政策‖当日匹配热点）")
    for _, r in ds.iterrows():
        ax.annotate(f'{r["n_matched"]}/60', (r["date"], r["jsd"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("JSD")
    ax.set_title(f"政策–热点主题分布差异（JSD）{n_days} 日趋势\n（标注：当日匹配热点数）")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{FIG_PREFIX}a_jsd_trend.svg")
    plt.close(fig)

    # 7b: 关键词持续性 top-14（标题级持续性≈0，关键词级才有信息量）
    fig, ax = plt.subplots(figsize=(8, 5.5))
    kwtop = kw_persist.head(14).iloc[::-1]
    klabels = [f'{k}（{p}）' for k, p in zip(kwtop["keyword"], kwtop["platforms"])]
    ax.barh(klabels, kwtop["n_days"])
    ax.set_xlabel(f"出现天数（{n_days} 天中）")
    ax.set_title("关键词跨日持续性 Top-14（jieba 分词，≥2字）")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{FIG_PREFIX}b_persistence.svg")
    plt.close(fig)

    # 7c: 平台重合度
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(overlap["date"], overlap["jaccard"])
    for _, r in overlap.iterrows():
        ax.annotate(f'{r["n_overlap"]}', (r["date"], r["jaccard"]),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_ylabel("Jaccard（微博 ∩ 百度 / 并）")
    ax.set_title("微博–百度热搜标题重合度（按日，归一化标题）")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{FIG_PREFIX}c_platform_overlap.svg")
    plt.close(fig)

    # 7d: 爆发
    if not bursts.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        b = bursts.head(10).iloc[::-1]
        blabels = [f'{t[:12]}({p})' for t, p in zip(b["example_title"], b["platform"])]
        ax.barh(blabels, b["z_score"])
        ax.axvline(z_thr, color="red", linestyle="--", label=f"阈值 {z_thr}")
        ax.set_xlabel(f"z-score（{n_days} 日基线，分平台）")
        ax.set_title("热度爆发检测 Top-10（仅出现≥2天的话题）")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{FIG_PREFIX}d_bursts.svg")
        plt.close(fig)

    # ---------- 5. 汇总落盘 ----------
    summary = {
        "run": f"temporal_{RUN_TAG}",
        "date_range": [dates[0].isoformat(), dates[-1].isoformat()],
        "n_days": len(dates),
        "n_snapshots": len(files),
        "n_rows": len(df),
        "model": model_name,
        "threshold": threshold,
        "n_distinct_topics": len(persist),
        "n_multiday_topics": len(multi),
        "top_persistent": multi.head(15).to_dict(orient="records"),
        "platform_overlap": overlap.to_dict(orient="records"),
        "n_bursts": len(bursts),
        "bursts": bursts.head(15).to_dict(orient="records") if not bursts.empty else [],
        "n_keywords_multiday": len(kw_persist),
        "top_keywords": kw_persist.head(25).to_dict(orient="records"),
        "daily": daily_stats,
        "aggregate_matched_topic_share": (
            pd.Series(all_matched_clusters).value_counts(normalize=True)
            .round(4).to_dict()
        ),
        "topic_names": topic_names,
    }
    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("summary saved:", out_dir / SUMMARY_NAME)


if __name__ == "__main__":
    main()
