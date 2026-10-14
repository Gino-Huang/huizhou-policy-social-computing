"""模型对决：用 120 对人工标注检验中文领域句向量是否优于当前模型。

当前模型 paraphrase-multilingual-MiniLM-L12-v2 在标注中表现不可靠
（所有 ≥0.60 的候选全被标为 0）。本脚本用 BAAI/bge-small-zh-v1.5
重算 863×60 相似度矩阵，在同一标注集上对比：
  - 各分数段准确率（严格 rel==2 / 宽松 rel>=1）
  - 阈值扫描 0.30–0.80（步长 0.05）的 P/R/F1
只做评估，不重跑全流程、不改默认配置（结论写入报告，由人决定是否切换）。

运行：HF_HUB_DISABLE_XET=1 .venv/bin/python scripts/model_shootout.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CHALLENGER = "BAAI/bge-small-zh-v1.5"
BANDS = [(0.70, 1.01), (0.60, 0.70), (0.50, 0.60), (0.40, 0.50), (0.30, 0.40)]


def sweep(df: pd.DataFrame, sim_col: str, strict: bool):
    """阈值扫描，返回每阈值的 (precision, recall, f1, n_kept, n_true_pos)。"""
    pos = (df["relevance"] == 2) if strict else (df["relevance"] >= 1)
    n_pos = int(pos.sum())
    rows = []
    for th in np.arange(0.30, 0.81, 0.05):
        kept = df[df[sim_col] >= th - 1e-9]
        tp = int((kept["relevance"] == 2).sum() if strict else (kept["relevance"] >= 1).sum())
        prec = tp / len(kept) if len(kept) else 0.0
        rec = tp / n_pos if n_pos else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append((round(float(th), 2), len(kept), tp, prec, rec, f1))
    return rows


def band_accuracy(df: pd.DataFrame, sim_col: str):
    out = []
    for lo, hi in BANDS:
        sub = df[(df[sim_col] >= lo) & (df[sim_col] < hi)]
        n = len(sub)
        strict_acc = float((sub["relevance"] == 2).mean()) if n else 0.0
        lenient_acc = float((sub["relevance"] >= 1).mean()) if n else 0.0
        out.append((f"[{lo:.2f},{hi:.2f})", n, strict_acc, lenient_acc))
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer

    passages = pd.read_parquet(PROJECT_ROOT / "data/processed/passages_v2.parquet")
    snap_files = sorted((PROJECT_ROOT / "data/raw/hotspots").glob("snapshot_*.csv"))
    hot = pd.concat([pd.read_csv(f) for f in snap_files], ignore_index=True)
    hot["hid"] = hot["platform"].astype(str) + "__" + hot.index.astype(str)

    p_texts = passages["text"].tolist()
    h_texts = hot["title"].tolist()
    p_index = {pid: i for i, pid in enumerate(passages["passage_id"])}
    h_index = {hid: i for i, hid in enumerate(hot["hid"])}

    print(f"encoding {len(p_texts)} passages + {len(h_texts)} hotspots with {CHALLENGER} ...")
    model = SentenceTransformer(CHALLENGER, device="cpu")
    pe = model.encode(p_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    he = model.encode(h_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    sim = (np.asarray(pe) @ np.asarray(he).T).astype(float)
    print("matrix:", sim.shape, "range: [%.3f, %.3f]" % (sim.min(), sim.max()))

    batch = pd.read_csv(
        PROJECT_ROOT / "data/processed/annotation_batch_20261014.csv", encoding="utf-8-sig"
    )
    assert batch["relevance"].notna().all(), "标注尚未完成"
    batch["bge_sim"] = [
        float(sim[p_index[r.passage_id], h_index[r.hotspot_id]])
        for r in batch.itertuples()
    ]
    # 当前模型分数即 batch 自带的 similarity 列
    batch = batch.rename(columns={"similarity": "cur_sim"})

    results = {"challenger": CHALLENGER, "n_pairs": len(batch)}
    for name, col in [("current(miniLM-L12-v2)", "cur_sim"), ("challenger(bge-small-zh)", "bge_sim")]:
        results[name] = {
            "bands": [
                {"band": b, "n": n, "strict_acc": round(sa, 4), "lenient_acc": round(la, 4)}
                for b, n, sa, la in band_accuracy(batch, col)
            ],
            "sweep_strict": [
                {"th": th, "n_kept": nk, "tp": tp, "p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
                for th, nk, tp, p, r, f in sweep(batch, col, strict=True)
            ],
            "sweep_lenient": [
                {"th": th, "n_kept": nk, "tp": tp, "p": round(p, 4), "r": round(r, 4), "f1": round(f, 4)}
                for th, nk, tp, p, r, f in sweep(batch, col, strict=False)
            ],
        }

    out = PROJECT_ROOT / "data/processed/model_shootout_20261014.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out)

    # 终端摘要
    for name in ["current(miniLM-L12-v2)", "challenger(bge-small-zh)"]:
        r = results[name]
        bs = max(r["sweep_strict"], key=lambda x: x["f1"])
        bl = max(r["sweep_lenient"], key=lambda x: x["f1"])
        print(f"--- {name}")
        print(f"  strict  best F1={bs['f1']:.3f} @th={bs['th']} (P={bs['p']:.3f} R={bs['r']:.3f})")
        print(f"  lenient best F1={bl['f1']:.3f} @th={bl['th']} (P={bl['p']:.3f} R={bl['r']:.3f})")
        top = [b for b in r["bands"] if b["band"].startswith("[0.6") or b["band"].startswith("[0.7")]
        print("  top bands (>=0.60):", [(b["band"], b["n"], round(b["lenient_acc"], 2)) for b in top])


if __name__ == "__main__":
    main()
