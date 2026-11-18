"""Condition-blind hand-coding of w' hits in the streaming study (offline vs streaming decodes of one model).

Every decision-sentence output of the `wrong` and `irrelevant` arms whose echo-cut translation contains w'
(`hits.hits`, the same reading H uses) becomes one row, from both conditions: the offline decode and the streaming
decode. Rows are shuffled with a fixed seed and carry an opaque id. Each row shows t, w, w' and a window of the
normalised translation around every w' that H counts (`window`, spans found on the raw scored hyp: `scored`).
Condition, system key, arm and prompt kind go to a separate KEY file (`system` holds the condition) that the coder
does not open. One coder codes the whole file in one session.

  python -m stglossary.sheets todo --sentences S.jsonl --offline OFF.jsonl --streaming ST.jsonl --out CODE.tsv --key KEY.tsv
      write the coder's file and the key; refuses to overwrite a CODE.tsv that has codes or guesses in it.

CODE.tsv columns: id, t, w, w_wrong, window, false_hit, condition_guess. Fill false_hit with
  n = w' is the model's rendering of t, including w' attached to a correct rendering of t (as a modifier or gloss)
  y = w' is not the rendering of t: another sense, part of a longer word, or a copy of the glossary.
After every row is coded, fill condition_guess with `offline` or `streaming` for every row: your guess of how the
output was decoded. `contrast count` reports the balanced accuracy of these guesses (`blinding`).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from . import coding
from . import hits as e1b_blind  # echo cut, hits, TSV io, sha, coded

SEED = e1b_blind.SEED
HALF = 20  # characters on each side of w'
CONDITIONS = ("offline", "streaming")
BA_MAX = 0.75  # balanced accuracy of condition guesses above this is a blinding failure
CODE_COLS = ["id", "t", "w", "w_wrong", "window", "false_hit", "condition_guess"]
KEY_COLS = e1b_blind.KEY_COLS
NL = "⏎"
MARKERS = re.compile(r"→|->|=>|⇒|:")  # glossary-copy separators after NFKC (full-width ： becomes :)
GUESS = {"offline": "offline", "o": "offline", "streaming": "streaming", "s": "streaming"}


def normalise(text: str) -> str:
    """NFKC (full-width punctuation and letters to their plain forms), a whitespace run holding a line break -> ' ⏎ ',
    any other whitespace run -> one space, stripped. Arrows and colons are kept."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", lambda m: f" {NL} " if ("\n" in m.group() or "\r" in m.group()) else " ", text)
    return text.strip()


OPEN, CLOSE = "\ue000", "\ue001"  # private-use marks around the raw spans; NFKC and the whitespace rule keep them


def window(text: str, form: str, half: int = HALF) -> str:
    """±`half` characters of the normalised text around every occurrence of `form` that H counts: the spans are found
    on the raw text (the `contains` reading, `coding.spans`), and only the display is normalised (a form that
    appears only after NFKC, e.g. full-width Latin, is not shown as a hit). When an arrow or
    colon stands on the same line beyond that reach, the window is widened to include it and up to `half` characters
    past it, so a glossary copy (`t → w'`, `t: w'`) stays recognisable. Overlapping windows merge; pieces are joined
    by ' … ', and a cut edge is marked with '…'. With no raw occurrence the whole normalised text is shown."""
    raw = coding.spans(text, form)
    if not raw:
        return normalise(text)
    marked, i = [], 0
    for s, e in raw:
        marked += [text[i:s], OPEN, text[s:e], CLOSE]
        i = e
    marked.append(text[i:])
    chars, sp, start = [], [], 0
    for ch in normalise("".join(marked)):
        if ch == OPEN:
            start = len(chars)
        elif ch == CLOSE:
            sp.append((start, len(chars)))
        else:
            chars.append(ch)
    text = "".join(chars)
    pieces = []
    for s, e in sp:
        lo, hi = max(0, s - half), min(len(text), e + half)
        ls = text.rfind(NL, 0, s) + 1
        le = text.find(NL, e)
        le = len(text) if le < 0 else le
        left = [m for m in MARKERS.finditer(text, ls, s)]
        if left and left[-1].start() < lo:
            lo = max(ls, left[-1].start() - half)
        right = MARKERS.search(text, e, le)
        if right and right.end() > hi:
            hi = min(le, right.end() + half)
        if pieces and lo <= pieces[-1][1]:
            pieces[-1][1] = max(pieces[-1][1], hi)
        else:
            pieces.append([lo, hi])
    out = " … ".join(text[lo:hi].strip() for lo, hi in pieces)
    return ("…" if pieces[0][0] > 0 else "") + out + ("…" if pieces[-1][1] < len(text) else "")


def scored(row: dict) -> str:
    """The `hyp` that is scored: one trailing U+FFFD stripped when the decode stopped at the token cap (`hit_cap`), in
    both conditions alike, since a cap can stop mid-character. Nothing else changes."""
    h = row["hyp"]
    return h[:-1] if row.get("hit_cap") and h.endswith("\ufffd") else h


def specs(offline: str, streaming: str) -> list[str]:
    return [f"offline={offline}", f"streaming={streaming}"]


def decision_only(recs: dict, paths: list[str]) -> None:
    """The streaming study decodes the decision sentences only; another row would enter no count."""
    for p in paths:
        bad = [r["key"] for r in map(json.loads, open(p)) if not recs[r["sid"]]["first"]]
        if bad:
            raise SystemExit(f"{p}: {len(bad)} rows are not decision sentences, e.g. {bad[:3]}")


def build(recs: dict, outputs: list[str]) -> tuple[list[dict], list[dict]]:
    """(code rows, key rows) for every w' hit, shuffled."""
    found = list(e1b_blind.hits(recs, outputs))
    random.Random(SEED).shuffle(found)
    code, key = [], []
    for i, (cond, r, rec, blk) in enumerate(found):
        rid = f"r{i:04d}"
        code.append(dict(id=rid, t=rec["t"], w=rec["w"], w_wrong=rec["w_wrong"],
                         window=window(e1b_blind.cut(scored(r), blk), rec["w_wrong"]), false_hit="", condition_guess=""))
        key.append(dict(id=rid, system=cond, key=r["key"], sid=r["sid"], arm=r["arm"],
                        prompt_kind=r.get("prompt_kind", "plain"), hyp_sha=e1b_blind.sha(r["hyp"])))
    return code, key


def todo(a) -> None:
    if Path(a.out).exists() and any(r["false_hit"] or r["condition_guess"] for r in e1b_blind.read(a.out, CODE_COLS)):
        raise SystemExit(f"{a.out} already has codes; move it away first")
    recs = e1b_blind.load(a.sentences)
    decision_only(recs, [a.offline, a.streaming])
    code, key = build(recs, specs(a.offline, a.streaming))
    e1b_blind.write(a.out, CODE_COLS, code)
    e1b_blind.write(a.key, KEY_COLS, key)
    print(f"{len(code)} rows to code -> {a.out}")


def coded(code: str, key: str, recs: dict, offline: str, streaming: str) -> tuple[set, set]:
    """(confirmed, false) as (condition, sid, arm); every row coded, and the hits and outputs unchanged since todo."""
    return e1b_blind.coded(code, key, recs, specs(offline, streaming), code_cols=CODE_COLS)


def code_pairs(code: str, key: str, scored_text: dict) -> dict:
    """The same (sid, arm) coded in both conditions. `scored_text`: condition -> row key ->
    the string H reads (echo cut of the scored hyp). Refuses (SystemExit, listing them) when the two are identical
    but carry different false_hit codes, since that difference would move D_st with no difference in model output;
    reports the pairs whose windows (all the coder saw) are identical while the outputs differ and the codes
    disagree."""
    c = {r["id"]: r for r in e1b_blind.read(code, CODE_COLS)}
    by: dict = {}
    for kr in e1b_blind.read(key, KEY_COLS):
        cr = c[kr["id"]]
        by.setdefault((kr["sid"], kr["arm"]), {})[kr["system"]] = dict(
            id=kr["id"], code=cr["false_hit"].strip(), window=cr["window"], hyp=scored_text[kr["system"]][kr["key"]])
    same_out, bad_out, bad_win = 0, [], []
    for (sid, arm), d in sorted(by.items()):
        if set(d) != set(CONDITIONS):
            continue
        o, s = d["offline"], d["streaming"]
        same_out += o["hyp"] == s["hyp"]
        if o["code"] == s["code"]:
            continue
        entry = dict(sid=sid, arm=arm, offline=(o["id"], o["code"]), streaming=(s["id"], s["code"]))
        if o["hyp"] == s["hyp"]:
            bad_out.append(entry)
        elif o["window"] == s["window"]:
            bad_win.append(entry)
    if bad_out:
        raise SystemExit(f"{len(bad_out)} (sid, arm) have byte-identical offline and streaming outputs coded "
                         f"differently; recode them: {bad_out}")
    return dict(both_conditions=sum(set(d) == set(CONDITIONS) for d in by.values()), identical_output=same_out,
                identical_window_different_code=bad_win)


def blinding(code: str, key: str) -> dict:
    """The coder's condition guesses against the key: per-condition recall and their mean (balanced accuracy).
    Above BA_MAX (0.75) is a blinding failure: the coding is to be redone by a second coder, and
    `contrast count` gives no decision row. With one condition absent from the file BA cannot be computed (None);
    that is not a failure, and `note` says so."""
    c = {r["id"]: r for r in e1b_blind.read(code, CODE_COLS)}
    k = {r["id"]: r for r in e1b_blind.read(key, KEY_COLS)}
    if c.keys() != k.keys():
        raise SystemExit("CODE and KEY ids differ")
    tab = {x: [0, 0] for x in CONDITIONS}  # condition -> [guessed right, rows]
    for rid, kr in k.items():
        g = GUESS.get(c[rid]["condition_guess"].strip().lower())
        if g is None:
            raise SystemExit(f"{rid}: condition_guess must be offline or streaming, got {c[rid]['condition_guess']!r}")
        tab[kr["system"]][1] += 1
        tab[kr["system"]][0] += g == kr["system"]
    rec = {x: (v[0] / v[1] if v[1] else None) for x, v in tab.items()}
    ba = sum(rec.values()) / 2 if all(v is not None for v in rec.values()) else None
    return dict(rows={x: v[1] for x, v in tab.items()}, right={x: v[0] for x, v in tab.items()}, recall=rec,
                balanced_accuracy=ba, failure=(ba is not None and ba > BA_MAX),
                note=(None if ba is not None else "balanced accuracy not computable: a condition has no coded rows"))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("todo")
    for x in ("sentences", "offline", "streaming", "out", "key"):
        t.add_argument("--" + x, required=True)
    a = ap.parse_args()
    {"todo": todo}[a.cmd](a)


if __name__ == "__main__":
    main()
