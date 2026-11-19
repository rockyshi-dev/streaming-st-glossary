"""Term extraction: Qwen2.5-7B-Instruct (vLLM, greedy) reads each English line with its FLoRes Chinese translation
and lists glossary terms with the Chinese form copied from the translation and a class.

Usage: python -m fleurs.terms --lines LINES.jsonl --model PATH --out TERMS_RAW.jsonl
One generate call over every line, in file order; temperature 0, max 512 new tokens, seed 0, bfloat16,
max_model_len 4096 (the settings of stglossary.decode_vllm). The raw output is saved untouched; parsing and every
check live in items.py, so a parsing change never needs a re-decode. OUT must not exist (not resumable; one call).
Each row records the commit, the dirty flag, the model path and whether generation hit the token cap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CLASSES = ("person", "place", "organization", "event", "work", "species", "food", "object", "concept")

PROMPT = """Below is one English sentence from the {domain} article "{title}" and its Chinese translation.

List the terms in the English sentence that a translator would put into a glossary for this article: names of \
people, places, organizations, events, works and products, names of species, and specialised nouns or noun phrases \
of the article's subject. Do not list common everyday words, pronouns, numbers, dates or units.

For each term give
- "en": the term exactly as it is written in the English sentence,
- "zh": its Chinese rendering, copied character for character from the Chinese translation,
- "class": one of {classes}.

Answer with a JSON list only, for example [{{"en": "...", "zh": "...", "class": "..."}}]. Answer [] if there is none.

English: {en}
Chinese: {zh}"""


def prompt(r: dict) -> str:
    return PROMPT.format(domain=r["domain"], title=r["title"], classes=", ".join(CLASSES), en=r["en"], zh=r["zh"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    here = str(Path(__file__).resolve().parent)
    commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    rows = [json.loads(l) for l in open(a.lines)]
    llm = LLM(a.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=512)
    texts = [tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    res = llm.generate(texts, sp, use_tqdm=False)
    part = out.with_name(out.name + ".part")
    with open(part, "w") as fh:
        for r, t, o in zip(rows, texts, res):
            fh.write(json.dumps(dict(line=r["line"], talk=r["talk"], commit=commit, dirty=dirty, model=a.model,
                                     hit_cap=o.outputs[0].finish_reason == "length", prompt=t,
                                     raw=o.outputs[0].text), ensure_ascii=False) + "\n")
    part.rename(out)
    print(f"{len(rows)} lines -> {out}; hit cap {sum(o.outputs[0].finish_reason == 'length' for o in res)}", flush=True)


if __name__ == "__main__":
    main()
