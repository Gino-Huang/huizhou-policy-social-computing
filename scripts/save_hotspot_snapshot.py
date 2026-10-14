"""Save a hotspot snapshot batch to the project's snapshot store.

Usage:
    python scripts/save_hotspot_snapshot.py items.json [--out-dir data/raw/hotspots]

items.json: a JSON array of objects with keys:
    platform, title, rank, raw_heat (nullable), url (nullable), captured_at (ISO, optional)

Writes data/raw/hotspots/snapshot_<UTC-timestamp>.csv with the schema expected
by src/ingestion/hotspot_collector.py:
    snapshot_id, platform, captured_at, title, rank, raw_heat, url
then runs the collector's validation and prints the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.hotspot_collector import load_snapshots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items_json", help="JSON array of hotspot items")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "raw" / "hotspots"))
    args = ap.parse_args()

    items = json.loads(Path(args.items_json).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise SystemExit("items_json must be a non-empty JSON array")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = f"snap_{stamp}"
    path = out_dir / f"snapshot_{stamp}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snapshot_id", "platform", "captured_at", "title", "rank", "raw_heat", "url"])
        for it in items:
            w.writerow([
                snapshot_id,
                it.get("platform", ""),
                it.get("captured_at") or now.isoformat(),
                it.get("title", ""),
                it.get("rank", ""),
                it.get("raw_heat", ""),
                it.get("url", ""),
            ])

    df, report = load_snapshots([str(path)])
    print(f"saved: {path}")
    print(f"snapshot_id: {snapshot_id}")
    print(f"rows loaded: {len(df)}")
    print("report:", json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
