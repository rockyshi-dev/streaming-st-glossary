"""Make the published data files from the decode outputs.

The decode scripts store provenance that only makes sense on the machine that ran them: the git commit of the code
and the local path of the model snapshot. The published files replace the model path by the Hugging Face id and
revision, drop the commit, and keep everything the counts read (prompts, hypotheses, per-step texts, times).

  python -m stglossary.export jsonl SRC.jsonl DST.jsonl --model ID@REVISION [--slim-steps]
      one row per row of SRC; `commit` removed, `model` replaced. --slim-steps keeps only key, sid, arm, rep,
      duration, policy and, per step, end, cont and committed (enough for `stops`).
  python -m stglossary.export tsv CODE.tsv KEY.tsv OUT_CODE.tsv OUT_KEY.tsv --systems A,B,...
      the hand codes of the listed systems only (KEY's `system` column), rows in their original order.

Every written JSONL file is checked: no string value may hold an absolute path.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DROP = ("commit",)
SLIM_ROW = ("key", "sid", "arm", "rep", "duration", "policy")
SLIM_STEP = ("end", "cont", "committed")
ABS_PATH = re.compile(r"(^|[\s\"'=])/(srv|nfs|home|data|tmp|root|mnt|Users)/")


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


def export_row(r: dict, model: str, slim: bool) -> dict:
    if slim:
        out = {k: r[k] for k in SLIM_ROW}
        out["steps"] = [{k: s[k] for k in SLIM_STEP} for s in r["steps"]]
        return out
    out = {k: v for k, v in r.items() if k not in DROP}
    if "model" in out:
        out["model"] = model
    return out


def jsonl(a) -> None:
    n = 0
    with open(a.src) as fi, open(a.dst, "x") as fo:
        for line in fi:
            row = export_row(json.loads(line), a.model, a.slim_steps)
            bad = [s for s in strings(row) if ABS_PATH.search(s)]
            if bad:
                raise SystemExit(f"{a.src}: row {n} still holds a path: {bad[0][:80]}")
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} rows -> {a.dst}")


def tsv(a) -> None:
    keep = set(a.systems.split(","))
    with open(a.key, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        kcols, krows = rd.fieldnames, [r for r in rd if r["system"] in keep]
    ids = {r["id"] for r in krows}
    with open(a.code, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        ccols, crows = rd.fieldnames, [r for r in rd if r["id"] in ids]
    if len(crows) != len(ids):
        raise SystemExit(f"{len(ids)} key rows but {len(crows)} code rows")
    for path, cols, rows in ((a.out_code, ccols, crows), (a.out_key, kcols, krows)):
        with open(path, "x", newline="") as fh:
            w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(krows)} rows of {sorted(keep)} -> {a.out_code}, {a.out_key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jsonl")
    j.add_argument("src")
    j.add_argument("dst")
    j.add_argument("--model", default="")
    j.add_argument("--slim-steps", action="store_true")
    t = sub.add_parser("tsv")
    for x in ("code", "key", "out_code", "out_key"):
        t.add_argument(x)
    t.add_argument("--systems", required=True)
    a = ap.parse_args()
    if a.cmd == "jsonl" and not a.slim_steps and not a.model:
        raise SystemExit("--model is required unless --slim-steps")
    {"jsonl": jsonl, "tsv": tsv}[a.cmd](a)


if __name__ == "__main__":
    main()
