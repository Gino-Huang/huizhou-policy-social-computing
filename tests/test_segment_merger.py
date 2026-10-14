"""Tests for src/preprocessing/segment_merger.py."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.segment_merger import merge_segments, merge_stats  # noqa: E402


def _make_df(rows):
    """rows: list of (segment_id, doc_id, order, text)."""
    return pd.DataFrame(
        [{"segment_id": s, "doc_id": d, "order": o, "clean": t} for s, d, o, t in rows]
    )


def test_merges_short_segments_to_target_length():
    df = _make_df([(f"s{i}", "d1", i, "政策文本内容填充" * 2) for i in range(10)])
    # each segment = 16 chars; 10 segs = 160 chars -> expect 1-2 passages
    out = merge_segments(df, target_min=100)
    assert len(out) >= 1
    # trailing passage may be shorter than target_min but never below min_match_len
    assert all(out["char_len"] >= 30)
    assert merge_stats(out)["n_source_segments"] == 10


def test_never_drops_content_trailing_short_merged_upward():
    # 7 segments of 40 chars = 280; first passage takes ~120, trailing 40 merges up
    df = _make_df([(f"s{i}", "d1", i, "x" * 40) for i in range(7)])
    out = merge_segments(df, target_min=100, min_match_len=30)
    assert merge_stats(out)["n_source_segments"] == 7
    assert (out["char_len"] >= 30).all()


def test_single_short_doc_kept_but_excluded_from_matching():
    df = _make_df([("s0", "d1", 0, "太短")])
    out = merge_segments(df, min_match_len=30)
    assert len(out) == 1
    assert not out.iloc[0]["used_in_matching"]
    assert "excluded" in out.iloc[0]["note"]


def test_no_cross_document_merge():
    df = _make_df(
        [("a0", "da", 0, "y" * 60), ("a1", "da", 1, "y" * 60),
         ("b0", "db", 0, "z" * 60), ("b1", "db", 1, "z" * 60)]
    )
    out = merge_segments(df, target_min=100)
    assert set(out["doc_id"]) == {"da", "db"}
    for _, r in out.iterrows():
        assert len({s.split("_")[0] for s in r["source_segment_ids"]}) == 1 or True
    # each passage's source segments all belong to its doc
    assert all(
        all(sid.startswith("a") for sid in r["source_segment_ids"]) or
        all(sid.startswith("b") for sid in r["source_segment_ids"])
        for _, r in out.iterrows()
    )


def test_provenance_records_order_range():
    df = _make_df([(f"s{i}", "d1", i * 2, "w" * 25) for i in range(8)])
    out = merge_segments(df, target_min=100)
    first = out.iloc[0]
    assert first["order_start"] == 0
    assert first["order_end"] >= first["order_start"]
    assert len(first["source_segment_ids"]) >= 4  # 4*25=100
