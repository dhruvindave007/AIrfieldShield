#!/usr/bin/env python3
"""Download archived METAR reports from the Iowa Environmental Mesonet (IEM) ASOS
archive (public, no API key) and build the hourly dataset used for training.

    python scripts/fetch_metar.py --start 2016 --end 2024

Raw per-station/year CSVs are cached in data/raw/ (git-ignored); the processed
hourly table is written to data/metar_hourly.csv.gz.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_models.ml.dataset import build_hourly, raw_iem_to_obs  # noqa: E402
from ai_models.ml.stations import STATIONS  # noqa: E402

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
FIELDS = ["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "vsby", "metar"]


def fetch_year(station: str, year: int, raw_dir: Path, retries: int = 4) -> pd.DataFrame:
    path = raw_dir / f"{station}_{year}.csv"
    if not path.exists():
        params = [("station", station)] + [("data", f) for f in FIELDS] + [
            ("year1", year), ("month1", 1), ("day1", 1),
            ("year2", year + 1), ("month2", 1), ("day2", 1),
            ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
            ("missing", "M"), ("trace", "T"), ("direct", "no"),
            ("report_type", 3), ("report_type", 4),
        ]
        for attempt in range(retries):
            try:
                r = requests.get(IEM_URL, params=params, timeout=300,
                                 headers={"User-Agent": "airfieldshield-research-script"})
                r.raise_for_status()
                if not r.text.startswith("station,valid"):
                    raise RuntimeError(f"unexpected response: {r.text[:120]!r}")
                tmp = path.with_suffix(".part")
                tmp.write_text(r.text, encoding="utf-8")
                tmp.replace(path)  # atomic: never leave a truncated cache file
                break
            except Exception as exc:  # network hiccup / rate limit
                if attempt == retries - 1:
                    raise
                print(f"  retry {station} {year}: {exc}")
                time.sleep(10 * (attempt + 1))
        time.sleep(1.5)  # be polite to the public service
    return pd.read_csv(io.StringIO(path.read_text(encoding="utf-8")), na_values=["M"],
                       dtype={"metar": str})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--end", type=int, default=2024, help="last year, inclusive")
    ap.add_argument("--stations", nargs="*", default=list(STATIONS), help="subset of ICAO codes")
    ap.add_argument("--download-only", action="store_true", help="only fill the data/raw cache")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "metar_hourly.csv.gz")
    args = ap.parse_args()

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames, qc = [], {}
    for station in args.stations:
        parts = []
        for year in range(args.start, args.end + 1):
            df = fetch_year(station, year, raw_dir)
            print(f"{station} {year}: {len(df):6d} reports")
            parts.append(df)
        if args.download_only:
            continue
        obs = raw_iem_to_obs(pd.concat(parts, ignore_index=True))
        qc[station] = obs.attrs["qc"]
        print(f"{station} QC: {qc[station]}")
        hourly = build_hourly(
            obs,
            start=pd.Timestamp(f"{args.start}-01-01", tz="UTC"),
            end=pd.Timestamp(f"{args.end + 1}-01-01", tz="UTC") - pd.Timedelta(hours=1),
        )
        hourly.insert(0, "station", station)
        frames.append(hourly.reset_index())
    if args.download_only:
        return
    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.2f", compression={"method": "gzip", "mtime": 0})
    meta = {"source": IEM_URL, "report_types": "METAR + SPECI (3, 4)", "years": [args.start, args.end],
            "stations": args.stations, "hourly_rows": int(len(out)), "qc": qc}
    args.out.with_name("metar_hourly.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(out)} hourly rows, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
