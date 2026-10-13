"""metrics 单元测试：玩具标注上的 P@K / R@K。"""
import pandas as pd
import pytest

from src.evaluation.metrics import (
    evaluate_rankings,
    load_annotations,
    precision_at_k,
    recall_at_k,
    threshold_sweep,
)


def test_precision_recall_at_k():
    retrieved = ["p1", "p2", "p3", "p4"]
    relevant = {"p2", "p4"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5  # p1x p2✓
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 4) == 1.0


def test_recall_no_relevant_is_nan():
    import math

    assert math.isnan(recall_at_k(["p1"], set(), 5))
    assert precision_at_k([], {"p1"}, 5) == 0.0


def test_precision_k_validation():
    with pytest.raises(ValueError):
        precision_at_k(["p1"], {"p1"}, 0)


def _toy_annotations():
    return pd.DataFrame(
        {
            "pair_id": ["h1__p1", "h1__p2", "h1__p3", "h2__p1"],
            "relevant": [2, 0, 1, 0],
        }
    )


def test_evaluate_rankings(tmp_path):
    p = tmp_path / "ann.csv"
    _toy_annotations().to_csv(p, index=False)
    ann = load_annotations(p)
    assert set(ann.columns) >= {"hotspot_id", "policy_id", "relevant"}
    assert ann.loc[0, "hotspot_id"] == "h1" and ann.loc[0, "policy_id"] == "p1"
    rankings = {"h1": ["p1", "p2", "p3"], "h2": ["p1"]}
    out = evaluate_rankings(ann, rankings, k=2)
    # h1: 相关={p1,p3}; top2=[p1,p2] -> P@2=0.5, R@2=0.5
    h1 = out[out["hotspot_id"] == "h1"].iloc[0]
    assert h1["P@2"] == 0.5 and h1["R@2"] == 0.5
    # h2: 无相关 -> R@2 为 NaN，P@2=0
    h2 = out[out["hotspot_id"] == "h2"].iloc[0]
    assert h2["P@2"] == 0.0
    assert out[out["hotspot_id"] == "MACRO_AVG"].iloc[0]["P@2"] == 0.25


def test_threshold_sweep_monotonic():
    df = _toy_annotations()
    df["hotspot_id"] = df["pair_id"].str.rsplit("__", n=1).str[0]
    df["policy_id"] = df["pair_id"].str.rsplit("__", n=1).str[1]
    scores = pd.DataFrame(
        {
            "hotspot_id": ["h1", "h1", "h1", "h2"],
            "policy_id": ["p1", "p2", "p3", "p1"],
            "similarity": [0.9, 0.4, 0.7, 0.2],
        }
    )
    out = threshold_sweep(df, scores, thresholds=[0.5, 0.8])
    row08 = out[out["threshold"] == 0.8].iloc[0]
    # t=0.8 只保留 p1(0.9)：1 个相关 / 1 个保留 -> precision=1.0
    assert row08["precision"] == 1.0
    row05 = out[out["threshold"] == 0.5].iloc[0]
    assert row05["n_kept"] == 2  # p1, p3
    assert row05["recall"] == 1.0  # 两个相关项都被保留
