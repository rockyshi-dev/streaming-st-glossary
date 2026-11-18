"""Blind hand-coding of w' hits in offline decodes, and the shared helpers (echo cut, TSV io, hashes).

Every output whose echo-cut translation contains w' (arms `wrong` and `irrelevant`) becomes one row. Rows are
shuffled with a fixed seed and carry an opaque id; system, mode, arm and prompt labels go to a separate KEY file
that the coder does not open.

  python -m stglossary.hits todo --sentences S.jsonl --out CODE.tsv --key KEY.tsv SYSTEM=OUT.jsonl ...
      write CODE.tsv (coder's file) and KEY.tsv; refuses to overwrite a CODE.tsv that has codes in it.

CODE.tsv columns: id, t, w, w_wrong, glossary, output, false_hit. The glossary column holds only the lines that all
three arms share, so it does not reveal the arm. Fill false_hit with
  y = w' is not used as the translation of t (other sense, part of a longer word, or only in a glossary copy)
  n = w' is the model's translation of t.
SYSTEM names are labels such as e2e-txt:o7; they appear only in KEY.tsv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

from . import coding

SEED = 20261117
ARMS = ("wrong", "irrelevant")
CODE_COLS = ["id", "t", "w", "w_wrong", "glossary", "output", "false_hit"]
KEY_COLS = ["id", "system", "key", "sid", "arm", "prompt_kind", "hyp_sha"]


def blocks(rec: dict) -> dict:
    b = rec["blocks"]
    return b if isinstance(b, dict) else json.loads(b)


def shared_block(rec: dict) -> list:
    """The glossary lines common to all three arms. The arm's own block would show `t -> w'` (wrong) or a
    replacement line (irrelevant) and so reveal the arm to the coder; t, w and w' are shown in their own columns."""
    b = blocks(rec)
    common = [ln for ln in b["correct"] if ln in b["irrelevant"] and ln in b["wrong"]]
    if any(s == rec["t"] for s, _ in common):
        raise SystemExit(f"{rec['sid']}: the shared glossary lines still contain t")
    return common


def cut(text: str, block) -> str:
    """The echo cut (coding.translation_part), then a cut at a line that begins with a block source term and `:`/`：`."""
    text = coding.translation_part(text)
    terms = [s for s, _ in (block or [])]
    if terms:
        pat = re.compile(r"^\s*(?:" + "|".join(re.escape(s) for s in terms) + r")\s*[:：]", re.M | re.I)
        m = pat.search(text)
        if m:
            text = text[: m.start()]
    return text


def sha(x: str) -> str:
    return hashlib.sha256(x.encode()).hexdigest()[:16]


def flat(x: str) -> str:
    return x.replace("\t", " ").replace("\r", " ").replace("\n", " ⏎ ")


def load(sentences: str) -> dict:
    return {r["sid"]: r for r in map(json.loads, open(sentences))}


def hits(recs: dict, specs: list[str]):
    for spec in specs:
        system, path = spec.split("=", 1)
        for r in map(json.loads, open(path)):
            if r["arm"] not in ARMS or r.get("extra"):
                continue
            rec = recs[r["sid"]]
            blk = blocks(rec)[r["arm"]]
            if coding.contains(cut(r["hyp"], blk), rec["w_wrong"]):
                yield system, r, rec, blk


def read(path: str, cols: list[str]) -> list[dict]:
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        if rd.fieldnames != cols:
            raise SystemExit(f"{path}: header {rd.fieldnames} != {cols}")
        return list(rd)


def write(path: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def todo(a) -> None:
    if Path(a.out).exists() and any(r["false_hit"] for r in read(a.out, CODE_COLS)):
        raise SystemExit(f"{a.out} already has codes; move it away first")
    recs = load(a.sentences)
    found = list(hits(recs, a.outputs))
    random.Random(SEED).shuffle(found)
    code, key = [], []
    for i, (system, r, rec, blk) in enumerate(found):
        rid = f"r{i:04d}"
        code.append(dict(id=rid, t=rec["t"], w=rec["w"], w_wrong=rec["w_wrong"],
                         glossary=flat("; ".join(f"{s} → {t}" for s, t in shared_block(rec))), output=flat(r["hyp"]), false_hit=""))
        key.append(dict(id=rid, system=system, key=r["key"], sid=r["sid"], arm=r["arm"],
                        prompt_kind=r.get("prompt_kind", "plain"), hyp_sha=sha(r["hyp"])))
    write(a.out, CODE_COLS, code)
    write(a.key, KEY_COLS, key)
    print(f"{len(code)} rows to code -> {a.out}")


def confirmed(a, recs) -> set:
    """(system, sid, arm) with a confirmed w' hit. Every row must be coded and match the current outputs."""
    return coded(a.code, a.key, recs, a.outputs)[0]


def coded(code_path: str, key_path: str, recs: dict, outputs: list[str], code_cols: list[str] = CODE_COLS) -> tuple[set, set]:
    """(confirmed, false): (system, sid, arm) coded n and coded y. Every row must be coded and match the outputs.
    `code_cols`: the coder file's header (the streaming study's file has other columns; `id` and `false_hit` are read)."""
    a = argparse.Namespace(code=code_path, key=key_path, outputs=outputs)
    code = {r["id"]: r for r in read(a.code, code_cols)}
    key = {r["id"]: r for r in read(a.key, KEY_COLS)}
    if code.keys() != key.keys():
        raise SystemExit("CODE and KEY ids differ")
    cur = {(s, r["key"]): sha(r["hyp"]) for s, r, _, _ in hits(recs, a.outputs)}
    if set(cur) != {(k["system"], k["key"]) for k in key.values()}:
        raise SystemExit("the outputs' w' hits differ from KEY; rerun todo")
    out, false = set(), set()
    for rid, k in key.items():
        c = code[rid]["false_hit"].strip()
        if c not in ("y", "n"):
            raise SystemExit(f"{rid}: false_hit must be y or n, got {c!r}")
        if cur[(k["system"], k["key"])] != k["hyp_sha"]:
            raise SystemExit(f"{rid}: output changed since todo")
        (out if c == "n" else false).add((k["system"], k["sid"], k["arm"]))
    return out, false


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["todo"])
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("outputs", nargs="+")
    a = ap.parse_args()
    todo(a)


if __name__ == "__main__":
    main()
