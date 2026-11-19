"""Tests of the FLEURS example scan, without a model. Run: python3 tests/test_fleurs.py

Reads the published outputs in data/ and data/fleurs/ (a few MB of JSON; seconds on one CPU core).
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fleurs.scan import scan_row, spoken_end  # noqa: E402
from stglossary.align import t_end  # noqa: E402
from stglossary.coding import lemma, spans  # noqa: E402


def load(d: Path, stream: str):
    recs = {r["sid"]: r for r in map(json.loads, open(d / "sentences.jsonl"))}
    al = {r["sid"]: r for r in map(json.loads, open(d / "align.jsonl"))}
    rows = [r for r in map(json.loads, open(d / stream)) if r["arm"] != "none"]
    return recs, al, rows


def examples(d: Path, stream: str) -> dict:
    recs, al, rows = load(d, stream)
    out = {}
    for row in rows:
        ar = al.get(row["sid"])
        if not ar or not ar.get("aligned"):
            continue
        hits = [h for h in scan_row(row, recs[row["sid"]], ar["words"], spans, t_end, lemma) if h["survives"]]
        if hits:
            out[row["key"]] = hits
    return out


def test_criterion_flags_the_acl6060_case():
    """The criterion was required to flag the post's ACL 60/60 case (dev/147, wrong arm) and dev/110."""
    ex = examples(ROOT / "data", "streaming.jsonl")
    assert "dev/147|wrong|0" in ex and "dev/110|wrong|0" in ex, sorted(ex)


def test_fleurs_counts():
    """The published FLEURS scan: examples per arm of 225 outputs each."""
    ex = examples(ROOT / "data" / "fleurs", "stream.jsonl")
    per_arm = collections.Counter(k.split("|")[1] for k in ex)
    assert per_arm == {"correct": 42, "wrong": 37, "irrelevant": 32}, per_arm
    published = json.load(open(ROOT / "data" / "fleurs" / "scan.json"))
    assert sorted(ex) == sorted(x["key"] for x in published["examples"])


def test_wrong_form_committed_early():
    """w' in the final output of the wrong arm, and committed before t was spoken (as fleurs.early_probe counts)."""
    recs, al, rows = load(ROOT / "data" / "fleurs", "stream.jsonl")
    final = early = 0
    for row in rows:
        if row["arm"] != "wrong" or not al[row["sid"]].get("aligned"):
            continue
        rec = recs[row["sid"]]
        final += rec["w_wrong"] in row["hyp"]
        first = next((s["end"] for s in row["steps"] if rec["w_wrong"] in s["committed"]), None)
        if first is not None:
            te = spoken_end(rec["en"].strip(), rec["t"], al[row["sid"]]["words"], t_end, lemma)
            early += te is None or first < te
    assert (final, early) == (44, 6), (final, early)


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} tests passed")
