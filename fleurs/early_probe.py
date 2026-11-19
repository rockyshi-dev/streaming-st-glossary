"""Count, per arm, how often the arm's form of the probe term t is committed before t has been spoken.

For each streamed output: the first step whose committed text contains the arm's form for t (w in `correct`,
w' in `wrong`), compared with the end of t in the forced alignment (scan.spoken_end). Early: committed at a step
that ends before t ends, or t is never spoken. Also counts finals that contain the form. Descriptive only: the
blog example's context, not part of the pre-registered criterion.

Usage: python -m fleurs.early_probe --stream STREAM.jsonl --align ALIGN.jsonl --sentences S.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fleurs.scan import spoken_end  # noqa: E402
from stglossary.align import t_end  # noqa: E402
from stglossary.coding import lemma  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--sentences", required=True)
    a = ap.parse_args()

    recs = {r["sid"]: r for r in map(json.loads, open(a.sentences))}
    al = {r["sid"]: r for r in map(json.loads, open(a.align))}
    c = collections.Counter()
    early = collections.defaultdict(list)
    for row in map(json.loads, open(a.stream)):
        arm = row["arm"]
        if arm not in ("correct", "wrong") or not al.get(row["sid"], {}).get("aligned"):
            continue
        rec = recs[row["sid"]]
        form = rec["w"] if arm == "correct" else rec["w_wrong"]
        c[arm, "n"] += 1
        c[arm, "final"] += form in row["hyp"]
        first = next((s["end"] for s in row["steps"] if form in s["committed"]), None)
        if first is None:
            continue
        c[arm, "committed"] += 1
        te = spoken_end(rec["en"].strip(), rec["t"], al[row["sid"]]["words"], t_end, lemma)
        if te is None or first < te:
            c[arm, "early"] += 1
            early[arm].append(dict(key=row["key"], first=first, t_end=te))
    for arm in ("correct", "wrong"):
        print(arm, {k: c[arm, k] for k in ("n", "final", "committed", "early")})
    print(json.dumps(early, ensure_ascii=False))


if __name__ == "__main__":
    main()
