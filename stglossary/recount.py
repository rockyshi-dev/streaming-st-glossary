"""Recount every number of the post from the published decodes and hand codes, on CPU, in about a minute.

  python -m stglossary.recount [--data data] [--out result.json]

1. Streaming vs offline (Qwen2.5-Omni-7B, instructed prompt, 116 decision items): wrong-form insertion H per
   condition, the paired difference with its talk-clustered bootstrap interval, whether a rise of k is excluded, how many outputs
   showed w' in any streaming hypothesis, early commits, and the latency of the wrong arm (`contrast.count`).
2. Early stops with and without the sentence-end hold-back (`stops`).
3. The offline panel on the gold transcript: H for the text-only LLM, Qwen2-Audio and Qwen2.5-Omni (`panel.counts`).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import contrast, hits, panel, stops

PANEL = (("txt", "text-llm.jsonl", "Qwen2.5-7B-Instruct (text only)"),
         ("e2e-txt:qa2", "qwen2-audio-text.jsonl", "Qwen2-Audio-7B-Instruct, text in"),
         ("e2e-txt:o7", "qwen25-omni-text.jsonl", "Qwen2.5-Omni-7B, text in"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--out", help="also write the full streaming-vs-offline result as JSON")
    a = ap.parse_args()
    d = Path(a.data)
    S = str(d / "sentences.jsonl")

    recs = hits.load(S)
    items = [r for r in recs.values() if r["first"]]
    segs = [r["duration"] for r in map(json.loads, open(d / "align.jsonl"))]
    gl = [len(hits.blocks(r)["wrong"]) for r in items]
    print(f"items: {len(items)} decision sentences from {len({r['talk'] for r in items})} talks; "
          f"segments {min(segs):.1f}-{max(segs):.1f} s (median {statistics.median(segs):.1f} s); "
          f"glossary entries per item {min(gl)}-{max(gl)} (median {statistics.median(gl):g})")

    # 1. streaming vs offline
    c = d / "coding"
    res = contrast.count(argparse.Namespace(sentences=S, offline=str(d / "offline.jsonl"),
                                            streaming=str(d / "streaming.jsonl"), code=str(c / "CODE.tsv"),
                                            key=str(c / "KEY.tsv"), ccode=str(c / "CCODE.tsv"),
                                            ckey=str(c / "CKEY.tsv"), tokenizer=None, text_control=None))
    k = res["contrast"]
    lo, hi = k["interval_D_over_N"]
    print("\n1. streaming vs offline, Qwen2.5-Omni-7B")
    print(f"   items whose output inserts the wrong glossary form, wrong arm minus irrelevant arm (H): "
          f"offline {k['H']['offline']}, streaming {k['H']['streaming']}")
    print(f"   difference D = {k['D_st']:+d} of N = {k['N']} (items up {k['discordant']['up']}, down "
          f"{k['discordant']['down']}); 95% interval of D/N {lo:.3f} .. {hi:.3f}; threshold k = {k['k']:.1f} items")
    verdict = {"P0": "the counts are too low to see insertion", "P1": "streaming raises insertion by k or more",
               "P2": "a rise of k or more is excluded", "P3": "neither a rise of k nor its absence is shown"}
    print(f"   {verdict[res['decision_row']]}: {'; '.join(res['decision_clauses'])}")
    ec = res["early_commit"]
    for arm in ("wrong", "irrelevant"):
        e = ec[arm]
        print(f"   {arm:10s}: w' in any streaming hypothesis {e['any_hypothesis']}, in the final output "
              f"{e['at_risk']}, committed before the speaker reached the term {e['early']}")
    lat = res["latency"]["wrong"]
    print(f"   latency, wrong arm: LAAL {lat['LAAL_ms']:.0f} ms, AL {lat['AL_ms']:.0f} ms")
    print(f"   spot check of the correct arm: {res['correct_spot_check']['uptake']} of "
          f"{res['correct_spot_check']['read']} use w as the translation of t; condition guesses of the coder: "
          f"balanced accuracy {res['blinding']['balanced_accuracy']:.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # 2. early stops
    print("\n2. streaming outputs that stop with 3 s or more of audio left")
    for name, f in (("without the sentence-end hold-back", "streaming-before-holdback.jsonl"),
                    ("with the hold-back", "streaming.jsonl")):
        n, of, _ = stops.count(str(d / f))
        print(f"   {name}: {n} of {of}")

    # 3. offline panel, gold transcript in
    p = d / "panel"
    panel.recs_by_sid.update(recs)
    specs = [f"{lab}={p / f}" for lab, f, _ in PANEL]
    _, false = hits.coded(str(p / "CODE.tsv"), str(p / "KEY.tsv"), recs, specs)
    print("\n3. offline panel, gold transcript in, instructed prompt: H")
    for lab, f, name in PANEL:
        out = panel.load(p / f, recs, lab, "instructed")
        r, _, _ = panel.counts(lab, out, items, false)
        print(f"   {name:36s} H = {r['H']}")


if __name__ == "__main__":
    main()
