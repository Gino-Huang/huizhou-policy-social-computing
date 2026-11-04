"""Export frontend data for the public showcase page (index.html).

Reads real analysis outputs from data/processed/ and writes versioned,
public, static files to public-data/ (committed to GitHub, read by the
page on GitHub Pages). Also writes public-data/data.js
(window.POLICY_DATA) so the page still works when opened via file://.

Per the 2026-10-25 revision proposal: no hand-written numbers in HTML,
every dataset carries a status field (real / sample / missing), and the
manifest records version, time, sources and commit.

Usage:
    python scripts/export_frontend_data.py
"""
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data" / "processed"
OUT = ROOT / "public-data"

FRIENDLY = {
    0: "企业人才", 1: "教育就业", 2: "科技创新", 3: "海洋旅游", 4: "新材料智能",
    5: "工程规划", 6: "城乡建设", 7: "养老医疗", 8: "投资改革", 9: "法治思想",
    10: "安全涉外", 11: "经济增长", 12: "绿色消费", 13: "行政执法", 14: "市场立法",
}
SIZES = {0: 50, 1: 49, 2: 55, 3: 94, 4: 67, 5: 65, 6: 141, 7: 76,
         8: 30, 9: 37, 10: 41, 11: 42, 12: 31, 13: 42, 14: 43}
TOTAL = sum(SIZES.values())  # 863
assert TOTAL == 863, TOTAL


def load_json(name):
    with open(D / name, encoding="utf-8") as f:
        return json.load(f)


def read_csv(name):
    with open(D / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    run_v2 = load_json("run_summary_v2.json")
    synergy = load_json("synergy_summary.json")
    temporal = load_json("temporal_summary_20261027.json")
    run_v3 = load_json("run_summary_v3.json")
    topic_edges = read_csv("synergy_topic_edges.csv")

    kw_by_id = {int(k): v for k, v in run_v2["cluster_keywords"].items()}

    topics = [{
        "id": i,
        "name": FRIENDLY[i],
        "keywords": kw_by_id[i],
        "n_passages": SIZES[i],
        "share": round(SIZES[i] / TOTAL, 4),
        "report": "reports/policy_synergy_20261014.md",
    } for i in range(15)]

    betweenness = {r["topic"]: r["betweenness"]
                   for r in synergy["topic_graph"]["betweenness_ranking"]}
    community_of = {}
    for c in synergy["topic_graph"]["communities"]:
        for m in c["members"]:
            community_of[m] = c["community"]

    topic_network = {
        "nodes": [{
            "name": FRIENDLY[i],
            "size": SIZES[i],
            "betweenness": betweenness[FRIENDLY[i]],
            "community": community_of[FRIENDLY[i]],
            "keywords": kw_by_id[i],
        } for i in range(15)],
        "edges": [{
            "source": r["source"],
            "target": r["target"],
            "weight": round(float(r["weight"]), 4),
        } for r in topic_edges],
    }

    daily = temporal["daily"]
    jsd_vals = [d["jsd"] for d in daily]
    n_matched = sum(d["n_matched"] for d in daily)
    n_total = sum(d["n_hotspots"] for d in daily)
    comparison = {
        "n_matched": n_matched,
        "n_total": n_total,
        "match_rate": round(n_matched / n_total, 4),
        "jsd_mean": round(sum(jsd_vals) / len(jsd_vals), 4),
        "jsd_min": round(min(jsd_vals), 4),
        "jsd_max": round(max(jsd_vals), 4),
        "policy_share": {FRIENDLY[i]: round(SIZES[i] / TOTAL, 4) for i in range(15)},
        "matched_share": {FRIENDLY[int(k)]: v
                          for k, v in temporal["aggregate_matched_topic_share"].items()},
        "precision_strict": 0.22,
        "precision_source": "reports/model_shootout_20261014.md",
        "top_matches": [{
            "hotspot": m["hotspot"],
            "platform": m["platform"],
            "policy_passage": m["policy_passage"],
            "similarity": m["similarity"],
            "policy_topic": m["policy_topic"],
        } for m in run_v3["top_matches"][:5]],
    }

    trends = {
        "date_range": temporal["date_range"],
        "daily": [{
            "date": d["date"],
            "jsd": round(d["jsd"], 4),
            "n_matched": d["n_matched"],
            "n_hotspots": d["n_hotspots"],
        } for d in daily],
        "platform_overlap": [{
            "date": o["date"],
            "jaccard": round(o["jaccard"], 4),
        } for o in temporal["platform_overlap"]],
        "persistence": [{
            "keyword": p["keyword"],
            "n_days": p["n_days"],
            "n_appearances": p["n_appearances"],
        } for p in temporal["top_keywords"][:12]],
        "bursts": [{
            "title": b["example_title"],
            "platform": b["platform"],
            "date": b["date"],
            "z_score": b["z_score"],
        } for b in temporal["bursts"]],
    }

    datasets = {
        "policy-topics.json": topics,
        "topic-network.json": topic_network,
        "hotspot-comparison.json": comparison,
        "temporal-trends.json": trends,
    }
    manifest = {
        "version": "2026-10-27",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "corpus": {
            "policy_docs": 7,
            "passages": TOTAL,
            "topics": 15,
            "annotation_pairs": 120,
            "hotspot_days": len(daily),
            "hotspot_titles": n_total,
            "date_range": temporal["date_range"],
        },
        "sources": {
            "policy-topics.json": "reports/policy_synergy_20261014.md",
            "topic-network.json": "reports/policy_synergy_20261014.md",
            "hotspot-comparison.json": "reports/real_data_run3_20261015.md, reports/temporal_analysis_v3_20261027.md",
            "temporal-trends.json": "reports/temporal_analysis_v3_20261027.md",
        },
        "status": {name: "real" for name in datasets},
        "notes": ("14-day series 2026-10-14 to 2026-10-27 (840 hotspots), "
                  "superseding the 10-day figures in the 2026-10-25 proposal."),
    }

    OUT.mkdir(exist_ok=True)
    for name, data in datasets.items():
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    (OUT / "data-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_data = dict(datasets)
    all_data["data-manifest.json"] = manifest
    (OUT / "data.js").write_text(
        "window.POLICY_DATA = " + json.dumps(all_data, ensure_ascii=False) + ";\n",
        encoding="utf-8")

    # consistency checks against the reports
    assert n_matched == 303 and n_total == 840, (n_matched, n_total)
    assert len(topics) == 15 and len(topic_network["edges"]) == 30
    assert len(trends["daily"]) == 14
    print(f"exported {len(datasets)} datasets + manifest + data.js to {OUT}")
    print(f"match rate {n_matched}/{n_total} = {n_matched / n_total:.1%}, "
          f"JSD mean {sum(jsd_vals) / len(jsd_vals):.3f}")


if __name__ == "__main__":
    main()
