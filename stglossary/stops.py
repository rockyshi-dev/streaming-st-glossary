"""Early stops in streaming decodes: rows whose output ends while 3 s or more of audio remain.

A step's continuation has content when it holds a character that is not whitespace and whose Unicode category is not
punctuation (P*), separator (Z*) or symbol (S*). `last` is the last step whose continuation has content. A row is
stopped when a later step exists (every later continuation is only punctuation or empty) and the audio after the end
of step `last` is at least 3 s long (with no content step at all, from 0 s).

  python -m stglossary.stops STREAM.jsonl ...
"""
from __future__ import annotations

import json
import sys
import unicodedata

GAP = 3.0


def content(s: str) -> bool:
    return any(not (unicodedata.category(c).startswith(("P", "Z", "S")) or c.isspace()) for c in s)


def stopped(r: dict) -> bool:
    st = r["steps"]
    idx = [i for i, s in enumerate(st) if content(s["cont"])]
    last = idx[-1] if idx else -1
    return last < len(st) - 1 and r["duration"] - (st[last]["end"] if last >= 0 else 0.0) >= GAP


def count(path: str) -> tuple[int, int, list[str]]:
    rows = [json.loads(l) for l in open(path)]
    keys = [r["key"] for r in rows if stopped(r)]
    return len(keys), len(rows), keys


def main() -> None:
    for p in sys.argv[1:]:
        n, of, keys = count(p)
        print(f"{p}: {n} of {of} rows stop with >= {GAP:g} s of audio left {keys if n <= 5 else ''}")


if __name__ == "__main__":
    main()
