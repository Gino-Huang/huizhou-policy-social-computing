"""用人工标注校准语义匹配阈值。

读取填好 relevance 列的 data/processed/annotation_batch_20261014.csv，
结合 data/processed/annotation_candidates_20261014.parquet 中的相似度分数：
  1. 计算每个分数段的准确率（分严格/宽松两种口径）；
  2. 对阈值 0.30~0.80（步长 0.05）扫描 precision / recall / F1；
  3. 推荐 F1 最高的阈值，保存 reports/threshold_calibration.md。

口径：
  严格 (strict)：只有 relevant == 2 算正例；
  宽松 (lenient)：relevant >= 1 算正例。

列名说明：标注 CSV 用 `relevance`（见 reports/ANNOTATION_GUIDE.md），
与 src/evaluation/metrics.py 的 `relevant` 列同义；本脚本直接读取 `relevance`。

用法：
    .venv/bin/python scripts/score_annotations.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

BATCH_CSV = PROJECT_ROOT / "data" / "processed" / "annotation_batch_20261014.csv"
CAND_PARQUET = PROJECT_ROOT / "data" / "processed" / "annotation_candidates_20261014.parquet"
REPORT_OUT = PROJECT_ROOT / "reports" / "threshold_calibration.md"


def sweep(df: pd.DataFrame, positive: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    """阈值扫描：df 需含 similarity 列；positive 为布尔 Series（是否为正例）。

    precision = 被阈值保留的 pair 中正例占比；
    recall    = 全部正例中被阈值保留的占比。
    """
    rows = []
    n_pos_total = int(positive.sum())
    for t in thresholds:
        kept = df["similarity"] >= t
        tp = int((kept & positive).sum())
        n_kept = int(kept.sum())
        precision = tp / n_kept if n_kept else 0.0
        recall = tp / n_pos_total if n_pos_total else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append(
            {"threshold": round(t, 2), "n_kept": n_kept, "n_true_pos": tp,
             "precision": round(precision, 4), "recall": round(recall, 4),
             "f1": round(f1, 4)}
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not BATCH_CSV.exists():
        print(f"找不到标注文件：{BATCH_CSV}")
        sys.exit(1)
    ann = pd.read_csv(BATCH_CSV, encoding="utf-8-sig")

    if "relevance" not in ann.columns:
        print("标注 CSV 缺少 relevance 列，请检查文件格式。")
        sys.exit(1)
    labeled = ann[ann["relevance"].notna() & (ann["relevance"] != "")]
    if len(labeled) == 0:
        print("=" * 60)
        print("relevance 列还是空的——请先完成人工标注再运行本脚本。")
        print()
        print("标注方法见 reports/ANNOTATION_GUIDE.md：")
        print("  1. 用 Excel/WPS 打开 data/processed/annotation_batch_20261014.csv；")
        print(f"  2. 给 relevance 列填 0/1/2（共 {len(ann)} 行）；")
        print("  3. 保存为 CSV（UTF-8），放回原路径后重新运行本脚本。")
        print("=" * 60)
        sys.exit(2)
    if len(labeled) < len(ann):
        print(f"注意：{len(ann) - len(labeled)} 行尚未标注，只用已标注的 {len(labeled)} 行计算。")

    labeled = labeled.copy()
    labeled["relevance"] = labeled["relevance"].astype(int)
    bad = labeled[~labeled["relevance"].isin([0, 1, 2])]
    if len(bad):
        print(f"错误：{len(bad)} 行的 relevance 不在 0/1/2 范围内：")
        print(bad[["pair_id", "relevance"]].to_string())
        sys.exit(1)

    cand = pd.read_parquet(CAND_PARQUET)
    df = labeled.merge(
        cand.rename(columns={"policy_id": "passage_id"}),
        on=["hotspot_id", "passage_id"], how="left", suffixes=("", "_cand"),
    )
    # candidates parquet 的 policy_id 即 passage_id；补齐缺失的 similarity
    if df["similarity"].isna().any():
        df["similarity"] = df["similarity"].fillna(df["similarity_cand"])
    assert df["similarity"].notna().all(), "部分标注行在 candidates 中找不到相似度分数"

    print(f"\n已标注：{len(df)} 对（正例严格口径={(df['relevance'] == 2).sum()}，"
          f"宽松口径={(df['relevance'] >= 1).sum()}）\n")

    # ---- 1. 分数段准确率 ----
    print("== 各分数段准确率 ==")
    band_rows = []
    for band, g in df.groupby("score_band", sort=False):
        n = len(g)
        p_strict = (g["relevance"] == 2).mean()
        p_lenient = (g["relevance"] >= 1).mean()
        band_rows.append((band, n, p_strict, p_lenient))
        print(f"  {band:14s} n={n:3d}  严格准确率={p_strict:.2%}  宽松准确率={p_lenient:.2%}")

    # ---- 2. 阈值扫描 ----
    thresholds = [round(x, 2) for x in np.arange(0.30, 0.81, 0.05)]
    results = {}
    for name, pos in [
        ("strict (relevant==2)", df["relevance"] == 2),
        ("lenient (relevant>=1)", df["relevance"] >= 1),
    ]:
        sw = sweep(df, pos, thresholds)
        results[name] = sw
        best = sw.loc[sw["f1"].idxmax()]
        print(f"\n== 阈值扫描（{name}）==")
        print(sw.to_string(index=False))
        print(f"  -> 推荐阈值（F1 最高）: {best['threshold']:.2f} "
              f"(P={best['precision']:.3f}, R={best['recall']:.3f}, F1={best['f1']:.3f}, "
              f"保留 {int(best['n_kept'])}/{len(df)} 对)")

    # ---- 3. 保存报告 ----
    lines = [
        "# 阈值校准报告（基于人工标注）",
        "",
        f"标注对数：{len(df)}；严格正例 {(df['relevance'] == 2).sum()}；"
        f"宽松正例 {(df['relevance'] >= 1).sum()}。",
        "",
        "## 各分数段准确率",
        "",
        "| 分数段 | 样本数 | 严格准确率 (==2) | 宽松准确率 (>=1) |",
        "|---|---|---|---|",
    ]
    for band, n, ps, pl in band_rows:
        lines.append(f"| {band} | {n} | {ps:.2%} | {pl:.2%} |")
    for name, sw in results.items():
        best = sw.loc[sw["f1"].idxmax()]
        lines += [
            "",
            f"## 阈值扫描（{name}）",
            "",
            "| 阈值 | 保留对数 | 真正例 | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in sw.iterrows():
            lines.append(
                f"| {r['threshold']:.2f} | {int(r['n_kept'])} | {int(r['n_true_pos'])} "
                f"| {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
            )
        lines += [
            "",
            f"**推荐阈值：{best['threshold']:.2f}**（F1={best['f1']:.3f}，"
            f"P={best['precision']:.3f}，R={best['recall']:.3f}）",
        ]
    lines += [
        "",
        "## 解读注意事项",
        "",
        "- 本批样本是按分数段分层抽样的，不是随机样本；上表 precision/recall "
        "描述的是**本标注批**内的权衡，可用于选阈值，不宜直接解读为全量数据的准确率。",
        '- 若严格口径与宽松口径推荐的阈值差异大，说明"部分相关"的判定对结论敏感，'
        "建议复核标为 1 的样本。",
        "- 选定阈值后，用该阈值重跑 scripts/run_real_analysis_v2.py（改 configs/config.yaml "
        "中的 similarity_threshold），并更新 JSD / 网络分析结果。",
    ]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存：{REPORT_OUT}")


if __name__ == "__main__":
    main()
