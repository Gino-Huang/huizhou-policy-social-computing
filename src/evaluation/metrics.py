"""评估：基于人工标注的 Precision@K / Recall@K + 阈值灵敏度扫描。

标注 CSV 格式：
    pair_id, relevant
    h001_p003, 2        # 0=不相关，1=部分相关，2=高度相关
其中 pair_id 形如 "{hotspot_id}_{policy_id}"（两部分用最后一个下划线切分，
hotspot_id 本身可含下划线时请用制表符以外的明确分隔——本项目约定 pair_id
= f"{hotspot_id}__{policy_id}"，双下划线分隔）。

判定：relevant >= 1 视为相关。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_annotations(path: str) -> pd.DataFrame:
    """加载标注 CSV，校验列与取值。返回列：pair_id, hotspot_id, policy_id, relevant。"""
    df = pd.read_csv(path)
    required = {"pair_id", "relevant"}
    if not required.issubset(df.columns):
        raise ValueError(f"标注 CSV 缺少列：{required - set(df.columns)}")
    if not df["relevant"].isin([0, 1, 2]).all():
        raise ValueError("relevant 只允许取 0/1/2")
    parts = df["pair_id"].str.rsplit("__", n=1)
    df["hotspot_id"] = parts.str[0]
    df["policy_id"] = parts.str[1]
    if df["policy_id"].isna().any():
        raise ValueError("pair_id 格式错误，应为 {hotspot_id}__{policy_id}")
    return df


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K = 前 K 个中相关的 / K。"""
    if k <= 0:
        raise ValueError("k 必须 > 0")
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for pid in topk if pid in relevant)
    return hits / len(topk)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K = 前 K 个中相关的 / 全部相关数。无相关项时返回 NaN（不计入平均）。"""
    if k <= 0:
        raise ValueError("k 必须 > 0")
    if not relevant:
        return float("nan")
    hits = sum(1 for pid in retrieved[:k] if pid in relevant)
    return hits / len(relevant)


def evaluate_rankings(
    annotations: pd.DataFrame,
    rankings: dict[str, list[str]],
    k: int,
) -> pd.DataFrame:
    """对每个 hotspot 的检索排序计算 P@K / R@K，返回明细 + macro 平均行。"""
    rows = []
    for hid, retrieved in rankings.items():
        rel = set(annotations.loc[annotations["hotspot_id"] == hid].query("relevant >= 1")["policy_id"])
        rows.append(
            {
                "hotspot_id": hid,
                "n_relevant": len(rel),
                f"P@{k}": precision_at_k(retrieved, rel, k),
                f"R@{k}": recall_at_k(retrieved, rel, k),
            }
        )
    df = pd.DataFrame(rows)
    macro = {
        "hotspot_id": "MACRO_AVG",
        "n_relevant": df["n_relevant"].sum(),
        f"P@{k}": df[f"P@{k}"].mean(),
        f"R@{k}": df[f"R@{k}"].mean(skipna=True),
    }
    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)


def threshold_sweep(
    annotations: pd.DataFrame,
    scores: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """阈值灵敏度扫描。

    scores 列：hotspot_id, policy_id, similarity。
    对每个阈值 t：保留 similarity >= t 的匹配，按阈值过滤后的集合计算
    micro precision（保留中相关的占比）与 micro recall（相关中被保留的占比）。
    返回每阈值一行的 DataFrame，便于绘制 P-R 权衡曲线。
    """
    thresholds = thresholds or [round(x, 2) for x in np.arange(0.3, 0.96, 0.05)]
    ann = annotations.copy()
    ann["relevant_bin"] = (ann["relevant"] >= 1).astype(int)
    key = ann["hotspot_id"] + "__" + ann["policy_id"]
    rel_map = dict(zip(key, ann["relevant_bin"]))
    rows = []
    for t in thresholds:
        kept = scores[scores["similarity"] >= t].copy()
        kept_keys = kept["hotspot_id"] + "__" + kept["policy_id"]
        tp = sum(rel_map.get(kk, 0) for kk in kept_keys)
        n_rel_total = int(ann["relevant_bin"].sum())
        precision = tp / len(kept) if len(kept) else 0.0
        recall = tp / n_rel_total if n_rel_total else float("nan")
        rows.append(
            {
                "threshold": t,
                "n_kept": len(kept),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0,
            }
        )
    return pd.DataFrame(rows)
