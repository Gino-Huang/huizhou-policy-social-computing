"""热点快照采集：导入各平台热搜快照 CSV，校验 schema、去重。

快照 CSV 必须包含列：
    snapshot_id, platform, captured_at, title, rank, raw_heat, url
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "snapshot_id",
    "platform",
    "captured_at",
    "title",
    "rank",
    "raw_heat",
    "url",
]


def validate_schema(df: pd.DataFrame) -> list[str]:
    """返回缺失的列；无缺失返回空列表。"""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def load_snapshots(paths: list[str | Path]) -> tuple[pd.DataFrame, dict]:
    """合并多个快照 CSV，校验列、解析时间、按 (platform, title, 日期) 去重。

    返回 (去重后的 DataFrame, 统计报告 dict)。
    """
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    missing = validate_schema(df)
    if missing:
        raise ValueError(f"快照 CSV 缺少列：{missing}")

    report: dict = {"n_raw": len(df)}
    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")
    n_bad_time = int(df["captured_at"].isna().sum())
    df = df.dropna(subset=["captured_at"]).reset_index(drop=True)
    report["n_bad_time_dropped"] = n_bad_time

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["raw_heat"] = pd.to_numeric(df["raw_heat"], errors="coerce")
    df = df.dropna(subset=["rank", "raw_heat"]).reset_index(drop=True)

    df["title"] = df["title"].astype(str).str.strip()
    df["date"] = df["captured_at"].dt.date
    before = len(df)
    df = df.drop_duplicates(subset=["platform", "title", "date"], keep="first")
    report["n_duplicates_dropped"] = before - len(df)
    report["n_clean"] = len(df)
    report["platforms"] = sorted(df["platform"].unique().tolist())
    return df.reset_index(drop=True), report
