#!/usr/bin/env python3
"""Reproducible, dependency-free business entity-resolution pipeline.

Only the supplied TSV files are read.  SQLite is used as an on-disk blocking
index, which keeps the pipeline usable for multi-million-row source files
without loading a Cartesian product into memory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


LEGAL_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc",
    "ltd", "limited", "llp", "plc", "pvt", "private", "gmbh", "sa",
    "sas", "sarl", "bv", "ag", "holdings", "group",
}
ADDRESS_WORDS = {
    "road", "street", "avenue", "lane", "drive", "boulevard", "building",
    "floor", "near", "the", "and", "of", "at", "block", "sector", "nagar",
    "main", "west", "east", "north", "south", "apt", "suite", "unit",
}
ABBREVIATIONS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "ln": "lane", "dr": "drive", "hwy": "highway", "ctr": "center",
    "pvt": "private", "ltd": "limited", "corp": "corporation",
    "co": "company", "ste": "suite", "apt": "apartment", "fl": "floor",
}


def stable_bucket(value: str, modulo: int = 100) -> int:
    return int(hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest(), 16) % modulo


def ascii_lower(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return value.replace("&", " and ")


def words(value: str) -> list[str]:
    raw = re.findall(r"[a-z0-9]+", ascii_lower(value))
    return [ABBREVIATIONS.get(token, token) for token in raw]


def soundex(value: str) -> str:
    """Small, transparent phonetic key; it does not depend on country labels."""
    if not value:
        return ""
    table = {"b": "1", "f": "1", "p": "1", "v": "1", "c": "2", "g": "2",
             "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
             "d": "3", "t": "3", "l": "4", "m": "5", "n": "5", "r": "6"}
    out = value[0].upper()
    previous = table.get(value[0], "")
    for char in value[1:]:
        code = table.get(char, "")
        if code and code != previous:
            out += code
        previous = code
        if len(out) == 4:
            break
    return (out + "000")[:4]


def longest_token(tokens: Iterable[str], excluded: set[str]) -> str:
    choices = [t for t in tokens if len(t) > 2 and t not in excluded and not t.isdigit()]
    return max(choices, key=lambda t: (len(t), t), default="")


@dataclass(frozen=True)
class Normalized:
    country: str
    name: str
    name_core: str
    name_anchor: str
    name_phonetic: str
    address: str
    address_sorted: str
    postal: str
    number: str
    street_anchor: str
    zip_street: str


def normalise(name: str, address: str, country: str) -> Normalized:
    name_tokens = words(name)
    address_tokens = words(address)
    name_core_tokens = [x for x in name_tokens if x not in LEGAL_WORDS]
    name_core = "".join(name_core_tokens)
    address_clean = "".join(address_tokens)
    address_sorted = "".join(sorted(address_tokens))
    name_anchor = longest_token(name_core_tokens, set())
    street_anchor = longest_token(address_tokens, ADDRESS_WORDS)
    numeric = [x for x in address_tokens if x.isdigit()]
    # Postal patterns are intentionally country-agnostic: a 4--8 digit address token.
    postal_values = [x for x in numeric if 4 <= len(x) <= 8]
    postal = postal_values[-1] if postal_values else ""
    return Normalized(
        country=ascii_lower(country).strip(), name="".join(name_tokens), name_core=name_core,
        name_anchor=name_anchor, name_phonetic=soundex(name_anchor), address=address_clean,
        address_sorted=address_sorted, postal=postal, number=numeric[0] if numeric else "",
        street_anchor=street_anchor, zip_street=(postal + "|" + street_anchor[:5]) if postal and street_anchor else "",
    )


def open_tsv(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


SCHEMA = """
CREATE TABLE records (
    entity_id TEXT PRIMARY KEY, country TEXT NOT NULL, business_name TEXT NOT NULL,
    business_address TEXT NOT NULL, name_key TEXT, name_core TEXT, name_anchor TEXT,
    name_phonetic TEXT, address_key TEXT, address_sorted TEXT, postal TEXT,
    house_number TEXT, street_anchor TEXT, zip_street TEXT
);
"""
INDEX_VERSION = 2


def build_record_db(db_path: Path, source_paths: list[Path]) -> None:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF; PRAGMA temp_store=FILE; PRAGMA cache_size=-262144;" + SCHEMA)
    insert = "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    batch: list[tuple[str, ...]] = []
    total = 0
    for path in source_paths:
        for row in open_tsv(path):
            n = normalise(row["business_name"], row["business_address"], row["country"])
            batch.append((row["entity_id"], n.country, row["business_name"], row["business_address"],
                          n.name, n.name_core, n.name_anchor, n.name_phonetic, n.address,
                          n.address_sorted, n.postal, n.number, n.street_anchor, n.zip_street))
            if len(batch) >= 10000:
                con.executemany(insert, batch); con.commit(); total += len(batch); batch.clear()
                if total % 500000 == 0:
                    print(f"  indexed {total:,} records", file=sys.stderr, flush=True)
    if batch:
        con.executemany(insert, batch); con.commit(); total += len(batch)
    # Index only the final blocking keys used at inference. Candidate TSV therefore
    # represents exactly all rows passed to feature extraction/scoring.
    con.executescript("""
      CREATE INDEX ix_name_core ON records(country, name_core);
      CREATE INDEX ix_name_anchor ON records(country, name_anchor);
      CREATE INDEX ix_address_sorted ON records(country, address_sorted);
      CREATE INDEX ix_zip_street ON records(country, zip_street);
      CREATE INDEX ix_house_street ON records(country, house_number, street_anchor);
      CREATE INDEX ix_phonetic ON records(country, name_phonetic);
    """)
    con.execute(f"PRAGMA user_version={INDEX_VERSION}")
    con.execute("ANALYZE"); con.commit(); con.close()
    print(f"Built {db_path.name}: {total:,} source-2/source-3 records", file=sys.stderr, flush=True)


def candidate_rows(con: sqlite3.Connection, n: Normalized, phonetic_limit: int = 80) -> list[tuple]:
    """Return the *last* blocking set, before ML/heuristic scoring."""
    found: dict[str, tuple] = {}
    def fetch(sql: str, params: tuple[str, ...]) -> None:
        for row in con.execute(sql, params):
            found[row[0]] = row
    if n.name_core:
        fetch("SELECT * FROM records WHERE country=? AND name_core=?", (n.country, n.name_core))
    # The longest distinctive name token and house/street combination bridge common
    # partial-address and DBA/name-suffix variations. Both are bounded before they
    # enter the candidate set, avoiding broad generic blocks.
    if n.name_anchor:
        rows = list(con.execute("SELECT * FROM records WHERE country=? AND name_anchor=? LIMIT 81",
                                (n.country, n.name_anchor)))
        if len(rows) <= 80:
            for row in rows:
                found[row[0]] = row
    if n.address_sorted:
        fetch("SELECT * FROM records WHERE country=? AND address_sorted=?", (n.country, n.address_sorted))
    if n.zip_street:
        fetch("SELECT * FROM records WHERE country=? AND zip_street=?", (n.country, n.zip_street))
    if n.number and n.street_anchor:
        rows = list(con.execute("SELECT * FROM records WHERE country=? AND house_number=? AND street_anchor=? LIMIT 81",
                                (n.country, n.number, n.street_anchor)))
        if len(rows) <= 80:
            for row in rows:
                found[row[0]] = row
    # Phonetic block catches mild spelling variation. Very large blocks are excluded
    # because they are not plausible candidates and would harm precision/runtime.
    if n.name_phonetic and n.name_anchor:
        phonetic = list(con.execute("SELECT * FROM records WHERE country=? AND name_phonetic=? LIMIT ?",
                                    (n.country, n.name_phonetic, phonetic_limit + 1)))
        if len(phonetic) <= phonetic_limit:
            for row in phonetic:
                found[row[0]] = row
    return list(found.values())


def current_index(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        con = sqlite3.connect(db_path)
        version = con.execute("PRAGMA user_version").fetchone()[0]
        con.close()
        return version == INDEX_VERSION
    except sqlite3.Error:
        return False


def token_jaccard(left: str, right: str) -> float:
    a, b = set(words(left)), set(words(right))
    return len(a & b) / len(a | b) if a or b else 0.0


def ratio(left: str, right: str) -> float:
    """Fast character-trigram Dice similarity, suitable for million-scale scoring."""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_grams = {left[i:i + 3] for i in range(max(1, len(left) - 2))}
    right_grams = {right[i:i + 3] for i in range(max(1, len(right) - 2))}
    return 2.0 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def features(s1: dict[str, str], cand: tuple, n: Normalized | None = None) -> list[float]:
    n = n or normalise(s1["business_name"], s1["business_address"], s1["country"])
    # tuple positions follow SCHEMA.
    cname, caddr, ccore, cpostal, cnumber, cstreet = cand[2], cand[3], cand[5], cand[10], cand[11], cand[12]
    return [
        ratio(n.name, cand[4]), ratio(n.name_core, ccore), token_jaccard(s1["business_name"], cname),
        ratio(n.address, cand[8]), ratio(n.address_sorted, cand[9]), token_jaccard(s1["business_address"], caddr),
        float(bool(n.postal and n.postal == cpostal)), float(bool(n.number and n.number == cnumber)),
        float(bool(n.street_anchor and n.street_anchor == cstreet)),
    ]


class LogisticModel:
    """A compact original logistic classifier trained with deterministic SGD."""
    def __init__(self, dimensions: int):
        self.w = [0.0] * dimensions
        self.b = -2.0

    def probability(self, x: list[float]) -> float:
        z = max(-35.0, min(35.0, self.b + sum(a * b for a, b in zip(self.w, x))))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, examples: list[tuple[list[float], int]], epochs: int = 8) -> None:
        # Reweight positives, then use a conservative probability threshold later.
        for epoch in range(epochs):
            rate = 0.10 / (1.0 + epoch * 0.35)
            for x, y in examples:
                p = self.probability(x)
                weight = 2.0 if y else 1.0
                delta = rate * weight * (y - p)
                self.b += delta
                for j, value in enumerate(x):
                    self.w[j] += delta * value


def read_truth(path: Path) -> dict[str, set[str]]:
    truth: dict[str, set[str]] = {}
    for row in open_tsv(path):
        truth[row["source1_entity_id"]] = set(filter(None, row["matched_entity_ids"].split(",")))
    return truth


def f05(predicted: set[str], actual: set[str]) -> float:
    if not actual:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & actual); precision = tp / len(predicted); recall = tp / len(actual)
    return 1.25 * precision * recall / (0.25 * precision + recall) if precision + recall else 0.0


def train_model(train_dir: Path, work_dir: Path, sample_limit: int) -> tuple[LogisticModel, float, dict[str, float]]:
    db_path = work_dir / "train_index.sqlite"
    if not current_index(db_path):
        build_record_db(db_path, [train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"])
    truth = read_truth(train_dir / "train_ground_truth.tsv")
    con = sqlite3.connect(db_path)
    examples: list[tuple[list[float], int]] = []
    validation: dict[str, tuple[set[str], list[tuple[str, float]]]] = {}
    coverage_hits = coverage_total = entities = 0
    for s1 in open_tsv(train_dir / "train_source1.tsv"):
        # Stable sample avoids relying on row order and makes validation reproducible.
        if stable_bucket(s1["entity_id"], 1000) >= sample_limit:
            continue
        entities += 1
        actual = truth.get(s1["entity_id"], set())
        n = normalise(s1["business_name"], s1["business_address"], s1["country"])
        cands = candidate_rows(con, n)
        candidate_ids = {r[0] for r in cands}
        coverage_hits += len(actual & candidate_ids); coverage_total += len(actual)
        is_validation = stable_bucket(s1["entity_id"], 5) == 0
        pair_scores: list[tuple[str, list[float]]] = []
        # retain all positives and a bounded, deterministic selection of negatives.
        negative_count = 0
        for cand in cands:
            label = int(cand[0] in actual)
            if not label and negative_count >= 8:
                continue
            x = features(s1, cand, n=n)
            if label or stable_bucket(s1["entity_id"] + cand[0], 3) == 0:
                if is_validation:
                    pair_scores.append((cand[0], x))
                else:
                    examples.append((x, label))
            if not label:
                negative_count += 1
        if is_validation:
            validation[s1["entity_id"]] = (actual, pair_scores)
        if entities % 10000 == 0:
            print(f"  training sample: {entities:,} S1 rows, {len(examples):,} labelled pairs", file=sys.stderr, flush=True)
    con.close()
    if not examples or not validation:
        raise RuntimeError("Training sample produced no candidate examples; increase --train-sample-per-thousand.")
    model = LogisticModel(9); model.fit(examples)
    best_threshold, best_score = 0.95, -1.0
    for threshold_i in range(65, 100):
        threshold = threshold_i / 100.0
        score = sum(f05({cid for cid, x in scored if model.probability(x) >= threshold}, actual)
                    for actual, scored in validation.values()) / len(validation)
        if score > best_score:
            best_threshold, best_score = threshold, score
    stats = {"train_pairs": float(len(examples)), "validation_entities": float(len(validation)),
             "validation_f05": best_score, "candidate_recall": coverage_hits / coverage_total if coverage_total else 1.0}
    print("Training validation:", stats, "threshold", best_threshold, file=sys.stderr, flush=True)
    return model, best_threshold, stats


def write_outputs(test_dir: Path, output_dir: Path, work_dir: Path, model: LogisticModel, threshold: float) -> None:
    db_path = work_dir / "test_index.sqlite"
    if not current_index(db_path):
        build_record_db(db_path, [test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"])
    output_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    # Materialise the final blocking set in SQLite using bulk joins.  This is
    # equivalent to candidate_rows(), but avoids millions of Python/SQLite calls.
    con.executescript("""
      DROP TABLE IF EXISTS test_source1;
      DROP TABLE IF EXISTS final_candidates;
      CREATE TABLE test_source1 (
        entity_id TEXT PRIMARY KEY, country TEXT, business_name TEXT, business_address TEXT,
        name_key TEXT, name_core TEXT, name_anchor TEXT, name_phonetic TEXT, address_key TEXT,
        address_sorted TEXT, postal TEXT, house_number TEXT, street_anchor TEXT, zip_street TEXT
      );
      CREATE TABLE final_candidates (source1_entity_id TEXT, candidate_entity_id TEXT,
        PRIMARY KEY(source1_entity_id, candidate_entity_id)) WITHOUT ROWID;
    """)
    source_batch = []
    for s1 in open_tsv(test_dir / "test_source1.tsv"):
        n = normalise(s1["business_name"], s1["business_address"], s1["country"])
        source_batch.append((s1["entity_id"], n.country, s1["business_name"], s1["business_address"], n.name,
                             n.name_core, n.name_anchor, n.name_phonetic, n.address, n.address_sorted,
                             n.postal, n.number, n.street_anchor, n.zip_street))
        if len(source_batch) >= 20000:
            con.executemany("INSERT INTO test_source1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", source_batch); source_batch.clear()
    if source_batch:
        con.executemany("INSERT INTO test_source1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", source_batch)
    con.executescript("""
      CREATE INDEX ts1_core ON test_source1(country,name_core);
      CREATE INDEX ts1_addr ON test_source1(country,address_sorted);
      CREATE INDEX ts1_zipstreet ON test_source1(country,zip_street);
    """)
    blocks = [
      "s.name_core<>'' AND r.country=s.country AND r.name_core=s.name_core",
      "s.address_sorted<>'' AND r.country=s.country AND r.address_sorted=s.address_sorted",
      "s.zip_street<>'' AND r.country=s.country AND r.zip_street=s.zip_street",
    ]
    for block in blocks:
        con.execute("INSERT OR IGNORE INTO final_candidates SELECT s.entity_id,r.entity_id FROM test_source1 s JOIN records r ON " + block)
        con.commit()
    candidate_count = con.execute("SELECT count(*) FROM final_candidates").fetchone()[0]
    print(f"Materialised {candidate_count:,} final test candidate pairs", file=sys.stderr, flush=True)
    candidate_path = output_dir / "candidate_pairs.tsv"
    matching_path = output_dir / "matching_results.tsv"
    with candidate_path.open("w", encoding="utf-8", newline="") as candidate_file, matching_path.open("w", encoding="utf-8", newline="") as match_file:
        cw = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        mw = csv.writer(match_file, delimiter="\t", lineterminator="\n")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"]); mw.writerow(["source1_entity_id", "matched_entity_ids"])
        total = predicted = candidate_total = 0
        sql = """SELECT s.entity_id,s.country,s.business_name,s.business_address,c.candidate_entity_id,
                 r.entity_id,r.country,r.business_name,r.business_address,r.name_key,r.name_core,r.name_anchor,
                 r.name_phonetic,r.address_key,r.address_sorted,r.postal,r.house_number,r.street_anchor,r.zip_street
                 FROM test_source1 s LEFT JOIN final_candidates c ON c.source1_entity_id=s.entity_id
                 LEFT JOIN records r ON r.entity_id=c.candidate_entity_id ORDER BY s.entity_id,c.candidate_entity_id"""
        current_id = None; s1 = None; n = None; candidate_ids: list[str] = []; matches: list[str] = []
        for row in con.execute(sql):
            s1_id = row[0]
            if s1_id != current_id:
                if current_id is not None:
                    cw.writerow([current_id, ",".join(candidate_ids)]); mw.writerow([current_id, ",".join(matches)])
                    total += 1; candidate_total += len(candidate_ids); predicted += len(matches)
                    if total % 100000 == 0:
                        print(f"  scored {total:,} S1 rows; {candidate_total:,} candidate pairs; {predicted:,} matches", file=sys.stderr, flush=True)
                current_id = s1_id; s1 = {"entity_id": row[0], "country": row[1], "business_name": row[2], "business_address": row[3]}
                n = normalise(s1["business_name"], s1["business_address"], s1["country"]); candidate_ids = []; matches = []
            if row[4] is not None:
                cand = tuple(row[5:])
                candidate_ids.append(row[4])
                if model.probability(features(s1, cand, n=n)) >= threshold:
                    matches.append(row[4])
        if current_id is not None:
            cw.writerow([current_id, ",".join(candidate_ids)]); mw.writerow([current_id, ",".join(matches)])
            total += 1; candidate_total += len(candidate_ids); predicted += len(matches)
    con.close()
    print(f"Wrote {total:,} rows. Candidate pairs: {candidate_total:,}; final matches: {predicted:,}", file=sys.stderr, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing train/ and test/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--train-sample-per-thousand", type=int, default=80,
                        help="Deterministic S1 training sample size per 1,000 rows (default: 80).")
    args = parser.parse_args()
    if not 1 <= args.train_sample_per_thousand <= 1000:
        parser.error("--train-sample-per-thousand must be between 1 and 1000")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    model, threshold, _ = train_model(args.data_root / "train", args.work_dir, args.train_sample_per_thousand)
    write_outputs(args.data_root / "test", args.output_dir, args.work_dir, model, threshold)


if __name__ == "__main__":
    main()
