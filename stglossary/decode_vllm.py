"""Decode a text LLM (default Qwen/Qwen2.5-7B-Instruct) with vLLM, greedy (system txt; with --asr, a cascade).

Usage: python -m stglossary.decode_vllm --sentences S.jsonl --out OUT.jsonl [--model M] [--asr ASR.jsonl] [--limit N] [--prompt plain|instructed]
Without --asr the source is the gold transcript (txt); with --asr it is the ASR hypothesis (cas).

Batch composition is fixed: three generate calls (pass 0, 1, 2), each holding every sentence x
[none, correct, irrelevant, wrong] in the same order. Pass r supplies `none` repeat r; the arms
correct, irrelevant and wrong are read from pass 0 (passes 1-2 are kept as extra records).
OUT must not exist (the run is not resumable). Each record stores the commit, the model and `hit_cap` (vLLM's
finish_reason is "length").
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from vllm import LLM, SamplingParams

from . import prompts

ARMS = ["none", "correct", "irrelevant", "wrong"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", choices=["plain", "instructed"], default="plain")
    a = ap.parse_args()

    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists; move it away first")
    here = str(Path(__file__).resolve().parent)
    commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    recs = [json.loads(l) for l in open(a.sentences)]
    if a.limit:
        recs = recs[: a.limit]
    asr = {}
    if a.asr:
        asr = {json.loads(l)["sid"]: json.loads(l)["hyp"] for l in open(a.asr)}
    llm = LLM(a.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    jobs = []
    for r in recs:
        src = asr[r["sid"]] if a.asr else r["en"]
        for arm in ARMS:
            user = prompts.text_prompt(r, arm, src, a.prompt)
            text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True)
            jobs.append((r["sid"], arm, text))

    with open(a.out, "x") as fh:
        for p in range(3):
            res = llm.generate([j[2] for j in jobs], sp, use_tqdm=False)
            for (sid, arm, text), o in zip(jobs, res):
                if arm == "none":
                    rep, extra = p, False
                else:
                    rep, extra = (0, False) if p == 0 else (p, True)
                fh.write(json.dumps(dict(key=prompts.key(sid, arm, rep), sid=sid, arm=arm, rep=rep, pass_=p,
                                         extra=extra, source="asr" if a.asr else "gold", prompt_kind=a.prompt, commit=commit, dirty=dirty,
                                         model=a.model, hit_cap=o.outputs[0].finish_reason == "length", prompt=text,
                                         hyp=o.outputs[0].text.strip()), ensure_ascii=False) + "\n")
            fh.flush()
            print(f"pass {p} done: {len(jobs)} decodes", flush=True)


if __name__ == "__main__":
    main()
