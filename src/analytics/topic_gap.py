"""关注差异分析：政策文本 vs 媒体热点 的主题分布对比 + Jensen-Shannon 散度。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def build_distribution(labels: pd.Series | list, n_topics: int) -> np.ndarray:
    """由主题标签构建归一化分布向量（长度 n_topics，和为 1）。

    含归一化检查：和偏离 1 超过 1e-6 则报错，避免静默错误。
    """
    counts = pd.Series(labels).value_counts()
    p = np.zeros(n_topics, dtype=float)
    for topic, c in counts.items():
        if 0 <= int(topic) < n_topics:
            p[int(topic)] = c
    total = p.sum()
    if total <= 0:
        raise ValueError("空分布：没有有效的主题标签")
    p = p / total
    if not np.isclose(p.sum(), 1.0, atol=1e-6):
        raise ValueError(f"归一化检查失败，和={p.sum()}")
    return p


def js_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """Jensen-Shannon 散度（0=分布相同；base=2 时上界为 1）。

    scipy 的 jensenshannon 返回的是距离（散度的平方根），这里平方回去。
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p 与 q 维度不一致")
    dist = jensenshannon(p, q, base=base)
    return float(dist**2)


def gap_table(
    policy_p: np.ndarray,
    hotspot_p: np.ndarray,
    topic_names: list[str],
) -> pd.DataFrame:
    """输出主题关注差异表：政策占比、热点占比、差值（热点-政策）。"""
    if not (len(policy_p) == len(hotspot_p) == len(topic_names)):
        raise ValueError("长度不一致")
    df = pd.DataFrame(
        {
            "topic": topic_names,
            "policy_share": policy_p,
            "hotspot_share": hotspot_p,
        }
    )
    df["gap"] = df["hotspot_share"] - df["policy_share"]
    return df.sort_values("gap", ascending=False).reset_index(drop=True)
