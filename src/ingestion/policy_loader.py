"""政策文档加载：.txt / .md / .pdf（PyMuPDF）-> policy_documents + text_segments。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.text_cleaner import split_sentences


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_md(path: Path) -> str:
    # 简单处理：去掉标题符号与代码块围栏，保留正文
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            continue
        if s.startswith("#"):
            s = s.lstrip("#").strip()
        lines.append(s)
    return "\n".join(lines)


def _read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF，延迟导入

    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


_READERS = {".txt": _read_txt, ".md": _read_md, ".pdf": _read_pdf}


def load_policy_documents(paths: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载政策文档，返回 (policy_documents, text_segments)。

    policy_documents: doc_id, source_path, n_chars
    text_segments: segment_id, doc_id, section, text, order
      - section：按空行分段时的段落序号（P1, P2, ...），句子级切分时为 S<段落>-<句>
    """
    doc_rows, seg_rows = [], []
    for i, p in enumerate(paths):
        path = Path(p)
        reader = _READERS.get(path.suffix.lower())
        if reader is None:
            raise ValueError(f"不支持的文件类型：{path.suffix}（{path}）")
        raw = reader(path)
        doc_id = f"doc_{i:03d}"
        doc_rows.append(
            {"doc_id": doc_id, "source_path": str(path), "n_chars": len(raw)}
        )
        order = 0
        for pi, para in enumerate([x for x in raw.split("\n") if x.strip()], start=1):
            for si, sent in enumerate(split_sentences(para), start=1):
                seg_rows.append(
                    {
                        "segment_id": f"{doc_id}_s{order:04d}",
                        "doc_id": doc_id,
                        "section": f"P{pi}-S{si}",
                        "text": sent,
                        "order": order,
                    }
                )
                order += 1
    docs = pd.DataFrame(doc_rows)
    segs = pd.DataFrame(seg_rows)
    return docs, segs


def save_processed(docs: pd.DataFrame, segs: pd.DataFrame, out_dir: str | Path) -> None:
    """保存为 parquet（processed 层），便于下游复用。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(out / "policy_documents.parquet", index=False)
    segs.to_parquet(out / "text_segments.parquet", index=False)
