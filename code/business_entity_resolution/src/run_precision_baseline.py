#!/usr/bin/env python3
"""Generate a fully reproducible, precision-first ER submission from supplied TSVs.

The final candidate block is deliberately strict: same normalized open-set country,
suffix-free normalized name, and order-independent normalized address.  Every
candidate has all three agreements and is therefore also the final prediction.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from run_pipeline import normalise, open_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    db_path = args.work_dir / "precision_index.sqlite"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        con = sqlite3.connect(db_path)
        con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF; CREATE TABLE r (entity_id TEXT PRIMARY KEY, country TEXT, name_core TEXT, address_sorted TEXT);")
        insert = []
        for file_name in ("test_source2.tsv", "test_source3.tsv"):
            for row in open_tsv(args.data_root / "test" / file_name):
                n = normalise(row["business_name"], row["business_address"], row["country"])
                if n.name_core and n.address_sorted:
                    insert.append((row["entity_id"], n.country, n.name_core, n.address_sorted))
                if len(insert) >= 50000:
                    con.executemany("INSERT INTO r VALUES (?,?,?,?)", insert); insert.clear()
        if insert: con.executemany("INSERT INTO r VALUES (?,?,?,?)", insert)
        con.execute("CREATE INDEX exact_block ON r(country,name_core,address_sorted)")
        con.commit(); con.close()
    con = sqlite3.connect(db_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "candidate_pairs.tsv").open("w", encoding="utf-8", newline="") as cf, (args.output_dir / "matching_results.tsv").open("w", encoding="utf-8", newline="") as mf:
        cw, mw = csv.writer(cf, delimiter="\t", lineterminator="\n"), csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"]); mw.writerow(["source1_entity_id", "matched_entity_ids"])
        rows = matches = 0
        for s1 in open_tsv(args.data_root / "test" / "test_source1.tsv"):
            n = normalise(s1["business_name"], s1["business_address"], s1["country"])
            ids = [] if not (n.name_core and n.address_sorted) else [x[0] for x in con.execute("SELECT entity_id FROM r WHERE country=? AND name_core=? AND address_sorted=? ORDER BY entity_id", (n.country, n.name_core, n.address_sorted))]
            # These exact candidates are the only records evaluated; each satisfies
            # the strict final decision rule, so matches equal candidates.
            joined = ",".join(ids)
            cw.writerow([s1["entity_id"], joined]); mw.writerow([s1["entity_id"], joined])
            rows += 1; matches += len(ids)
            if rows % 100000 == 0: print(f"wrote {rows:,} S1 rows; {matches:,} strict matches", file=sys.stderr, flush=True)
    con.close()
    print(f"Finished {rows:,} rows and {matches:,} strict matches.", file=sys.stderr)


if __name__ == "__main__":
    main()
