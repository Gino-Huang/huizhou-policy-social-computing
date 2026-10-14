"""把同一文档内相邻的短文本段合并为 100~200 字的段落（passage）。

背景：PDF 按句切分后中位数仅 ~20 字，大量截断残片进入匹配池，
短文本余弦相似度不可靠，是语义匹配假阳性的主因（见
reports/real_data_run_20261014.md §3）。

规则：
- 只合并同一 doc_id 内按 order 相邻的段，不跨文档；
- 只在段边界处拼接（输入段本身已是句子级切分），不切断句子；
- 累计长度达到 target_min（默认 100 字）即输出一段；
- 单段累加后超过 target_max（默认 250 字）且当前已 ≥ min_match_len 时，
  先输出当前段再起新段，避免出现超长段落；
- 文档末尾不足 min_match_len（默认 30 字）的残余向上并入本篇上一篇，
  做到"不静默丢弃"；若整篇文档只有一段且不足 30 字，则保留但标记
  used_in_matching=False（匹配阶段排除，原因记入 note）。
- 每段记录来源：doc_id、source_segment_ids、order 起止、覆盖 section、字数。
"""
from __future__ import annotations

import pandas as pd


def merge_segments(
    df: pd.DataFrame,
    text_col: str = "clean",
    target_min: int = 100,
    target_max: int = 250,
    min_match_len: int = 30,
) -> pd.DataFrame:
    """将短段合并为段落。

    df 需包含列：segment_id, doc_id, order, [section], text_col。
    返回 DataFrame，列：passage_id, doc_id, text, char_len,
    source_segment_ids (list), order_start, order_end, sections (list),
    used_in_matching (bool), note (str)。
    """
    required = {"segment_id", "doc_id", "order", text_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"merge_segments 缺少必需列: {missing}")
    has_section = "section" in df.columns

    passages: list[dict] = []

    def emit(buf: list[dict], note: str = "") -> None:
        first, last = buf[0], buf[-1]
        text = "".join(b["text"] for b in buf)
        passages.append({
            "passage_id": f"p{len(passages):05d}",
            "doc_id": first["doc_id"],
            "text": text,
            "char_len": len(text),
            "source_segment_ids": [b["segment_id"] for b in buf],
            "order_start": first["order"],
            "order_end": last["order"],
            "sections": sorted({b["section"] for b in buf if b["section"]}) if has_section else [],
            "used_in_matching": len(text) >= min_match_len,
            "note": note,
        })

    for doc_id, g in df.groupby("doc_id", sort=False):
        g = g.sort_values("order")
        buf: list[dict] = []
        buf_len = 0
        last_doc_passage_idx: int | None = None  # 本篇已输出的最后一段在 passages 中的下标

        def flush(note: str = "") -> None:
            nonlocal buf, buf_len, last_doc_passage_idx
            if buf:
                emit(buf, note)
                last_doc_passage_idx = len(passages) - 1
                buf, buf_len = [], 0

        for _, r in g.iterrows():
            t = str(r[text_col])
            if buf and buf_len >= target_min:
                flush()
            elif buf and buf_len + len(t) > target_max and buf_len >= min_match_len:
                flush()
            buf.append({
                "segment_id": str(r["segment_id"]),
                "doc_id": doc_id,
                "order": r["order"],
                "section": r["section"] if has_section else "",
                "text": t,
            })
            buf_len += len(t)

        # 收尾：残余不足 min_match_len 则向上并入本篇上一篇
        if buf:
            if buf_len < min_match_len and last_doc_passage_idx is not None:
                prev = passages[last_doc_passage_idx]
                prev["text"] += "".join(b["text"] for b in buf)
                prev["char_len"] = len(prev["text"])
                prev["source_segment_ids"].extend(b["segment_id"] for b in buf)
                prev["order_end"] = buf[-1]["order"]
                if has_section:
                    prev["sections"] = sorted(set(prev["sections"]) | {b["section"] for b in buf if b["section"]})
                prev["used_in_matching"] = prev["char_len"] >= min_match_len
                prev["note"] = (prev["note"] + "; " if prev["note"] else "") + \
                    f"absorbed trailing {len(buf)} short segment(s)"
            else:
                note = "" if buf_len >= min_match_len else \
                    "single short passage in doc; kept but excluded from matching"
                flush(note)

    out = pd.DataFrame(passages)
    return out


def merge_stats(passages: pd.DataFrame) -> dict:
    """返回合并统计，供报告使用。"""
    matched = passages[passages["used_in_matching"]]
    return {
        "n_passages": int(len(passages)),
        "n_used_in_matching": int(len(matched)),
        "n_excluded_short": int((~passages["used_in_matching"]).sum()),
        "mean_char_len": round(float(passages["char_len"].mean()), 1),
        "median_char_len": float(passages["char_len"].median()),
        "min_char_len": int(passages["char_len"].min()),
        "max_char_len": int(passages["char_len"].max()),
        "n_source_segments": int(passages["source_segment_ids"].map(len).sum()),
    }
