"""Offline panel: loading decodes, the talk-clustered bootstrap, per-system counts, the pair set and the
correct-arm spot check.

Systems are labelled `txt` (text-only LLM on the gold transcript), `e2e-txt:<m>` (speech LLM <m> fed the gold
transcript), `e2e:<m>` (speech LLM <m> fed the audio). `recount` uses `load` and `counts` for the offline panel
(text-only LLM, Qwen2-Audio and Qwen2.5-Omni on the gold transcript).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import hits as blind
from . import insertion as e1
from . import prompts
from .coding import contains, is_latin

SEED = 20261117
B = 9999
EQ = e1.EQ_MARGIN


def load(path: Path, recs: dict, label: str, kind: str) -> dict:
    """(sid, arm, rep) -> echo-cut translation (an e1.Fixed whose .orig is the raw output). Checks every stored
    prompt against the item table and the prompt kind."""
    d = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("extra"):
            continue
        rec, arm = recs[r["sid"]], r["arm"]
        mode = "speech" if label.startswith("e2e:") else "text"
        want = prompts.text_after_source(rec, arm, mode, kind)
        if want not in r["prompt"] or r.get("prompt_kind") != kind:
            raise SystemExit(f"{label}: prompt of {r['key']} is not the {kind} prompt")
        if mode == "text" and f"English: {rec['en']}" not in r["prompt"]:
            raise SystemExit(f"{label}: text prompt of {r['key']} lacks the gold sentence")
        blk = None if arm == "none" else blind.blocks(rec)[arm]
        f = e1.Fixed(blind.cut(r["hyp"], blk))
        f.orig = r["hyp"]
        d[(r["sid"], arm, r["rep"])] = f
    return d



# ---------------------------------------------------------------- shared bootstrap

class Boot:
    """One draw of Webb six-point weights per talk and bootstrap replicate, shared by every interval."""

    def __init__(self, talks: list[str], seed: int = SEED):
        rng = random.Random(seed)
        self.talks = sorted(set(talks))
        self.w = [{g: rng.choice(e1.W6) for g in self.talks} for _ in range(B)]

    def interval(self, d: list[int], talks: list[str]) -> tuple[float, float]:
        n = len(d)
        if n == 0:
            return (float("nan"), float("nan"))
        mean = sum(d) / n
        S: dict[str, float] = {}
        for x, g in zip(d, talks):
            S[g] = S.get(g, 0.0) + x - mean
        bs = sorted(mean + sum(w[g] * s for g, s in S.items()) / n for w in self.w)
        return bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


# ---------------------------------------------------------------- per system

recs_by_sid: dict = {}  # sid -> sentence record; filled by the caller before `counts`


def counts(label: str, out: dict, items: list[dict], false: set) -> tuple[dict, set, set]:
    res, keep = e1.system_counts(label, out, items, false)

    def w_in(sid, arm, rep=0):
        return contains(e1.tpart(out[(sid, arm, rep)]), recs_by_sid[sid]["w"])

    def wp_in(sid, arm):
        return contains(e1.tpart(out[(sid, arm, 0)]), recs_by_sid[sid]["w_wrong"]) and (label, sid, arm) not in false

    kept = [it for it in items if it["item"] in keep]
    M = [it for it in kept if not is_latin(it["w"]) and not any(w_in(it["sid"], "none", r) for r in range(3))]
    c = sum(w_in(it["sid"], "correct") for it in M)
    wr = sum(w_in(it["sid"], "wrong") for it in M)
    ir = sum(w_in(it["sid"], "irrelevant") for it in M)
    nowp = [it for it in M if not wp_in(it["sid"], "wrong")]
    res.update(C_tf=c - wr, cue_only=wr - ir, M_items=len(M),
               C_tf_no_wprime=sum(w_in(it["sid"], "correct") for it in nowp)
               - sum(w_in(it["sid"], "wrong") for it in nowp), C_prime_E1=res["C_prime"])
    res["reads"] = res["C_tf"] > max(res["V"], 2) or res["H"] > max(res["V"], 2)
    return res, keep, {it["item"] for it in M}


def pair_items(outs: dict, a: str, b: str, items: list[dict], keep: dict) -> tuple[list, list]:
    """(both, its): items kept by both systems, and those of them whose `irrelevant` output has Han in both (the
    pair-set exclusion)."""
    both = [it for it in items if it["item"] in keep[a] and it["item"] in keep[b]]
    its = [it for it in both if e1.han(outs[a], it["sid"], "irrelevant") and e1.han(outs[b], it["sid"], "irrelevant")]
    return both, its


CCOLS = ["id", "t", "w", "glossary", "output", "uptake"]
CKEYS = ["id", "system", "sid", "hyp_sha"]


def correct_sample(outs: dict, Ms: dict, items: list[dict]) -> list[tuple]:
    """Spot check of the `correct` arm: per system, up to 10 `correct` decodes whose cut translation contains w, on M items."""
    rows = []
    for lab in sorted(outs):
        cand = sorted(it["sid"] for it in items if it["item"] in Ms[lab]
                      and contains(e1.tpart(outs[lab][(it["sid"], "correct", 0)]), it["w"]))
        for sid in random.Random(f"{SEED}:{lab}").sample(cand, min(10, len(cand))):
            rows.append((lab, sid, outs[lab][(sid, "correct", 0)].orig))
    random.Random(SEED).shuffle(rows)
    return rows


def write_ctodo(ccode: str, ckey: str, rows: list[tuple], recs: dict) -> None:
    """The spot-check reading file (rows from correct_sample) and its key."""
    code, key = [], []
    for i, (lab, sid, hyp) in enumerate(rows):
        rec = recs[sid]
        code.append(dict(id=f"c{i:04d}", t=rec["t"], w=rec["w"], output=blind.flat(hyp), uptake="",
                         glossary=blind.flat("; ".join(f"{x} → {y}" for x, y in blind.blocks(rec)["correct"]))))
        key.append(dict(id=f"c{i:04d}", system=lab, sid=sid, hyp_sha=blind.sha(hyp)))
    blind.write(ccode, CCOLS, code)
    blind.write(ckey, CKEYS, key)
    print(f"{len(code)} correct-arm rows to read -> {ccode}")


def correct_check(ccode: str, ckey: str, outs: dict) -> dict:
    code = {r["id"]: r for r in blind.read(ccode, CCOLS)}
    key = {r["id"]: r for r in blind.read(ckey, CKEYS)}
    if code.keys() != key.keys():
        raise SystemExit("CCODE and CKEY ids differ")
    res = {}
    for rid, k in key.items():
        c = code[rid]["uptake"].strip()
        if c not in ("y", "n"):
            raise SystemExit(f"{rid}: uptake must be y or n, got {c!r}")
        if blind.sha(outs[k["system"]][(k["sid"], "correct", 0)].orig) != k["hyp_sha"]:
            raise SystemExit(f"{rid}: output changed since ctodo")
        r = res.setdefault(k["system"], dict(read=0, uptake=0))
        r["read"] += 1
        r["uptake"] += c == "y"
    return res
