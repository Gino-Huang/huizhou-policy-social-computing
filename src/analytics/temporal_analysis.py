"""时间演化分析：日频计数、滑动平均、z-score 爆发检测、分平台热度处理。

核心约束：各平台热度指标口径不同（热搜指数 vs 播放量 vs 在榜时长），
禁止跨平台直接求和原始热度。跨平台比较只允许用"排名百分位"这类
无量纲指标，且必须在同一日期内计算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_counts(
    df: pd.DataFrame, date_col: str = "captured_at", topic_col: str = "topic"
) -> pd.DataFrame:
    """按 (日期, 主题) 计数。返回列：date, topic, count。"""
    d = df.copy()
    d["date"] = pd.to_datetime(d[date_col]).dt.date
    out = d.groupby(["date", topic_col]).size().reset_index(name="count")
    return out.sort_values(["date", topic_col]).reset_index(drop=True)


def moving_average(series: pd.Series, window: int = 7) -> pd.Series:
    """滑动平均（min_periods=1，边缘不丢数据）。"""
    return series.rolling(window=window, min_periods=1).mean()


def burst_detection(
    daily: pd.DataFrame,
    window: int = 7,
    z_threshold: float = 2.0,
    count_col: str = "count",
) -> pd.DataFrame:
    """z-score 爆发检测：当日计数显著高于过去 window 天均值。

    z = (x_t - mean(x_{t-window..t-1})) / std(...)；z >= 阈值记为爆发。
    返回原表 + z_score + is_burst 列。
    """
    df = daily.copy().sort_values("date").reset_index(drop=True)
    df["ma"] = df[count_col].shift(1).rolling(window=window, min_periods=2).mean()
    df["sd"] = df[count_col].shift(1).rolling(window=window, min_periods=2).std(ddof=0)
    df["z_score"] = (df[count_col] - df["ma"]) / df["sd"].replace(0, np.nan)
    df["is_burst"] = (df["z_score"] >= z_threshold).fillna(False)
    return df


def rank_percentiles(
    df: pd.DataFrame,
    platform_col: str = "platform",
    date_col: str = "captured_at",
    rank_col: str = "rank",
) -> pd.DataFrame:
    """同一 (平台, 日期) 内将排名转为百分位：rank_pct = rank / 当日该平台最大排名。

    值越小表示越靠前（rank=1 -> 最小百分位）。跨平台可比，但只反映
    相对位置，不反映绝对热度。
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d[date_col]).dt.date
    max_rank = d.groupby([platform_col, "date"])[rank_col].transform("max")
    d["rank_pct"] = d[rank_col] / max_rank.replace(0, np.nan)
    return d


def normalize_heat_per_platform(
    df: pd.DataFrame,
    platform_col: str = "platform",
    heat_col: str = "raw_heat",
    method: str = "zscore",
) -> pd.DataFrame:
    """分平台标准化热度（各平台独立计算，绝不跨平台混算）。

    method: "zscore" -> (x-mean)/std；"minmax" -> (x-min)/(max-min)。
    输出 heat_norm 列。
    """
    d = df.copy()
    if method == "zscore":
        g = d.groupby(platform_col)[heat_col]
        d["heat_norm"] = (d[heat_col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    elif method == "minmax":
        g = d.groupby(platform_col)[heat_col]
        rng = (g.transform("max") - g.transform("min")).replace(0, np.nan)
        d["heat_norm"] = (d[heat_col] - g.transform("min")) / rng
    else:
        raise ValueError(f"未知 method: {method}")
    return d
