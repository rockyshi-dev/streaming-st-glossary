"""Scan streaming outputs for a step that commits glossary forms of terms the speaker has not yet said.

Pre-registered criterion (README.md, section "FLEURS example", "Search criterion"; written before any FLEURS output
existed):
an output is an EXAMPLE when one committed step adds the glossary Chinese forms of at least 2 distinct terms
whose English term has not yet been spoken by that step's end, and that committed text survives into the final
output.

- Glossary: the block of the output's own arm (correct, irrelevant or wrong; `none` has no block and is skipped).
- "Adds the form" of an entry (english, form): the number of occurrences of `form` in the committed text after the
  step exceeds the number before it (stglossary.coding.spans: substring for Han forms, token boundaries for Latin
  forms). Entries that share one form count as one term.
- "Not yet spoken by the step's end": the end time of the word piece holding the last character of the first
  occurrence of `english` in the sentence (stglossary.align.t_end, from the forced alignment of the gold transcript)
  is later than the step's prefix end, or `english` does not occur in the sentence at all (never spoken).
- "Survives": the final output starts with the step's committed text (LA-2 never revises committed text, so this
  holds by construction; it is checked, not assumed).
- Reported per example: whether the probe term t is among the early terms (in the `wrong` arm its form is w'), and,
  separately, whether the step adds t's form of this arm at all (w' in `wrong`), spoken or not.

Outputs whose sentence is not aligned are not scanned and are listed.

Usage: python -m fleurs.scan --stream STREAM.jsonl --align ALIGN.jsonl --sentences S.jsonl --out SCAN.json
Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stglossary.align import t_end  # noqa: E402
from stglossary.coding import lemma, spans  # noqa: E402


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spoken_end(en: str, e: str, words: list, t_end, lemma) -> float | None:
    """End time of the first occurrence of e in the sentence, matched by spelling or, failing that, by lemma
    (a block keeps the article's first spelling: block "dinosaurs", sentence "dinosaur")."""
    te = t_end(en, e, words)
    if te is not None:
        return te
    toks = re.findall(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*", en)
    n, L = len(e.split()), lemma(e)
    for i in range(len(toks) - n + 1):
        cand = " ".join(toks[i:i + n])
        if lemma(cand) == L:
            te = t_end(en, cand, words)
            if te is not None:
                return te
    return None


def scan_row(row: dict, rec: dict, words: list, spans, t_end, lemma) -> list[dict]:
    """Every step of one output that meets the criterion (an output is an example when this is non-empty)."""
    blk = rec["blocks"]
    blk = blk if isinstance(blk, dict) else json.loads(blk)
    block = blk[row["arm"]]
    en = rec["en"].strip()
    spoken = {e: spoken_end(en, e, words, t_end, lemma) for e, _ in block}  # None: not in the sentence
    probe_form = next((f for e, f in block if e == rec["t"]), None)  # the arm's form for t (w' in `wrong`)
    final = row["hyp"]
    hits, prev = [], ""
    last = len(row["steps"]) - 1
    for i, st in enumerate(row["steps"]):
        cur = st["committed"]
        early = {}
        for e, f in block:
            if len(spans(cur, f)) > len(spans(prev, f)):
                te = spoken[e]
                # At the flush (prefix = whole audio) every word of the sentence has been spoken; alignment may
                # place a last word's end up to 0.05 s past the duration (and step ends are rounded), so the last step is it.
                if te is None or (te > st["end"] and i < last):
                    early.setdefault(f, []).append(dict(en=e, form=f, spoken_end=te, probe=e == rec["t"]))
        # A form nested in another added form (伊拉克 in 伊拉克研究小组) is one written term, not two.
        def inside(f):  # every occurrence of f in cur lies within an occurrence of another early form
            outer = [sp for g in early if g != f and f in g for sp in spans(cur, g)]
            return all(any(a <= x and y <= b for a, b in outer) for x, y in spans(cur, f))
        early = {f: v for f, v in early.items() if not inside(f)}
        if len(early) >= 2:
            ents = [x for v in early.values() for x in v]
            hits.append(dict(step_end=st["end"], committed_before=prev, committed_after=cur,
                             added=cur[len(prev):] if cur.startswith(prev) else None,
                             n_terms=len(early), terms=ents,
                             probe_among=any(x["probe"] for x in ents),
                             probe_form_added=probe_form is not None
                             and len(spans(cur, probe_form)) > len(spans(prev, probe_form)),
                             survives=final.startswith(cur.strip())))
        prev = cur
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    recs = {r["sid"]: r for r in map(json.loads, open(a.sentences))}
    al = {r["sid"]: r for r in map(json.loads, open(a.align))}
    per_arm = collections.Counter()
    examples, unaligned, n = [], [], collections.Counter()
    for row in map(json.loads, open(a.stream)):
        if row["arm"] == "none":
            continue
        n[row["arm"]] += 1
        ar = al.get(row["sid"])
        if not ar or not ar.get("aligned"):
            unaligned.append(row["key"])
            continue
        rec = recs[row["sid"]]
        hits = scan_row(row, rec, ar["words"], spans, t_end, lemma)
        hits = [h for h in hits if h["survives"]]
        if hits:
            per_arm[row["arm"]] += 1
            examples.append(dict(key=row["key"], sid=row["sid"], arm=row["arm"], talk=rec.get("talk"), t=rec["t"],
                                 w=rec["w"], w_wrong=rec["w_wrong"], en=rec["en"], duration=row["duration"],
                                 final=row["hyp"], steps=hits,
                                 probe_among=any(h["probe_among"] for h in hits),
                                 probe_form_added=any(h["probe_form_added"] for h in hits)))
    res = dict(stream=a.stream, stream_sha256=sha(a.stream), align_sha256=sha(a.align),
               sentences_sha256=sha(a.sentences), outputs_scanned=dict(n), unaligned=unaligned,
               examples_per_arm=dict(per_arm), examples=examples)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"scanned {dict(n)}; not aligned {len(unaligned)}; examples per arm {dict(per_arm)}")
    for x in examples:
        h = x["steps"][0]
        terms = ", ".join(f"{t['en']}={t['form']}" + ("*" if t["probe"] else "") for t in h["terms"])
        print(f"  {x['key']}: step {h['step_end']}s adds {h['n_terms']} terms [{terms}]"
              f" probe_among={x['probe_among']} probe_form_added={x['probe_form_added']} | {h['added']}")


if __name__ == "__main__":
    main()
