"""Decode a Qwen speech LLM with HF transformers, batch size 1, greedy, text output only.

Systems: `e2e:m` (audio in) and `e2e-txt:m` (gold transcript in) for m = Qwen2-Audio (qa2), Qwen2.5-Omni (omni25),
Qwen3-Omni (omni3). For the Omni models the talker is disabled and the thinker generates; Qwen3-Omni is sharded over every
visible card (device_map auto).

Usage: python -m stglossary.decode_hf --mode speech|text --sentences S.jsonl --corpus ROOT --out OUT.jsonl
         [--family qa2|omni25|omni3] [--model PATH] [--prompt plain|instructed] [--runs main|screen|plain3|hear]
         [--subset all|first|nonfirst] [--limit N] [--order after|before]
Defaults: qa2, plain, main, all. Resumable: keys already in OUT are skipped.
The load report (missing / unexpected keys) is written to OUT.load.json; the run aborts unless both are empty.
Generation stops at <|im_end|> or <|endoftext|> for every family: Qwen3-Omni's thinker config has no eos (snapshot
26291f7), and the wrapper that normally passes it is bypassed here. Each record stores the number of new
tokens and whether the decode stopped at the cap (`hit_cap`), the commit and the model path; a resumed run refuses
an OUT whose rows came from another commit, model, items file or setting, so the code must not change
while a run is resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import librosa
import torch
import transformers as tf

from . import prompts
from .prov import done_keys, provenance

FAMILY = {  # model class, audio content key in the chat message
    "qa2": ("Qwen2AudioForConditionalGeneration", "audio_url"),
    "omni25": ("Qwen2_5OmniForConditionalGeneration", "audio"),
    "omni3": ("Qwen3OmniMoeForConditionalGeneration", "audio"),
}


def load(family: str, path: str, out: Path):
    cls = getattr(tf, FAMILY[family][0])
    kw = dict(dtype=torch.bfloat16)
    if family == "omni3":
        kw["device_map"] = "auto"
    # transformers' own loading info is computed after its key conversion (e.g. fused MoE experts). Qwen2.5-Omni
    # overrides from_pretrained only to add speaker voices for the talker, which is disabled here, and that override
    # cannot return loading info; so its parent's from_pretrained is called.
    load_fn = super(cls, cls).from_pretrained if family == "omni25" else cls.from_pretrained
    model, info = load_fn(path, output_loading_info=True, **kw)
    rep = {k: sorted(map(str, v)) for k, v in info.items()}
    # Only the module that decodes must load clean: qa2 the whole model, Omni its thinker. Keys of the unused
    # talker / token2wav are recorded, not gated.
    scope = "" if family == "qa2" else "thinker."
    gated = {k: [x for x in rep.get(k, []) if x.startswith(scope)] for k in ("missing_keys", "unexpected_keys")}
    Path(str(out) + ".load.json").write_text(json.dumps(dict(model=path, family=family, gated_prefix=scope,
                                                             gated=gated, **rep), indent=1))
    if gated["missing_keys"] or gated["unexpected_keys"]:
        raise SystemExit(f"load report not clean under '{scope}': {len(gated['missing_keys'])} missing, "
                         f"{len(gated['unexpected_keys'])} unexpected (see {out}.load.json)")
    if family != "omni3":
        model = model.to("cuda:0")
    if family in ("omni25", "omni3"):
        model.disable_talker()
        model = model.thinker
    model.eval()
    return model


class Decoder:
    """Processor, model and stop tokens of one family; `chat` renders a prompt, `run` decodes it greedily.
    Shared by this script's offline decodes and the streaming decodes (`stream`)."""

    def __init__(self, family: str, path: str, out: Path):
        self.family = family
        self.proc = tf.AutoProcessor.from_pretrained(path)
        self.model = load(family, path, out)
        self.dev = self.model.device if family != "omni3" else "cuda:0"
        self.sr = self.proc.feature_extractor.sampling_rate
        self.akey = FAMILY[family][1]
        tok = self.proc.tokenizer
        ids = [tok.convert_tokens_to_ids(x) for x in ("<|endoftext|>", "<|im_end|>")]
        if None in ids or tok.unk_token_id in ids:
            raise SystemExit(f"stop tokens not in the vocabulary: {ids}")
        self.eos = sorted(set(ids))

    def chat(self, rec: dict, arm: str, mode: str, prompt_kind: str, order: str = "after",
             source_text: str | None = None) -> str:
        """The chat-template text of one decode, up to and including the assistant header. `source_text` replaces
        the gold sentence in text mode (word prefixes)."""
        if mode == "speech":
            after = prompts.text_after_source(rec, arm, "speech", prompt_kind)
            content = [{"type": "audio", self.akey: rec["wav"]}, {"type": "text", "text": after}]
            if order == "before":
                content = ([{"type": "text", "text": prompts.text_before_audio(rec, arm)}]
                           if arm != "none" else []) + [content[0], {"type": "text",
                                                                      "text": prompts.INSTR["speech"]}]
        else:
            src = rec["en"] if source_text is None else source_text
            content = [{"type": "text", "text": prompts.text_prompt(rec, arm, src, prompt_kind)}]
        conv = [{"role": "user", "content": content}]
        return self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    def run(self, text: str, audio, max_new_tokens: int, forced: list[int] | None = None) -> tuple[str, list, bool]:
        """Greedy decode of `text` (with `audio` when not None). `forced` token ids are appended after the rendered
        prompt, i.e. at the start of the assistant turn (the streaming committed text; never re-tokenized). Returns the
        decoded new text (special tokens skipped, not stripped), the new token ids (stop token included) and whether
        the decode stopped at the cap."""
        kw = dict(text=text, return_tensors="pt", padding=True)
        if audio is not None:
            kw.update(audio=[audio], sampling_rate=self.sr)
        inp = self.proc(**kw).to(self.dev)
        self.last_input_len = int(inp["input_ids"].shape[1])  # prompt ids incl. audio placeholders, before forced ids
        if forced:
            f = torch.tensor([forced], dtype=inp["input_ids"].dtype, device=inp["input_ids"].device)
            inp["input_ids"] = torch.cat([inp["input_ids"], f], dim=1)
            inp["attention_mask"] = torch.cat([inp["attention_mask"], torch.ones_like(f)], dim=1)
        with torch.inference_mode():
            gen = self.model.generate(**inp, do_sample=False, max_new_tokens=max_new_tokens,
                                      eos_token_id=self.eos, pad_token_id=self.eos[0])
        new = gen[0, inp["input_ids"].shape[1]:].tolist()
        hit_cap = len(new) >= max_new_tokens and not set(new) & set(self.eos)
        dec = self.proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return dec, new, hit_cap

    def text(self, ids: list[int]) -> str:
        """Token ids to text, special tokens skipped, as the offline `hyp` is decoded (not stripped)."""
        return self.proc.batch_decode([ids], skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["speech", "text"], required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", choices=sorted(FAMILY), default="qa2")
    ap.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    ap.add_argument("--prompt", choices=["plain", "instructed"], default="plain")
    ap.add_argument("--runs", choices=sorted(prompts.RUNSETS), default="main")
    ap.add_argument("--subset", choices=["all", "first", "nonfirst"], default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--order", choices=["after", "before"], default="after")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    if a.runs == "hear" and a.mode != "speech":
        raise SystemExit("--runs hear needs --mode speech")
    if a.order == "before" and (a.family != "qa2" or a.prompt != "plain"):
        raise SystemExit("--order before is a smoke variant for qa2 with the plain prompt only")

    recs = [json.loads(l) for l in open(a.sentences)]
    if a.subset != "all":
        recs = [r for r in recs if r["first"] == (a.subset == "first")]
    if a.limit:
        recs = recs[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    commit, dirty = provenance(here)
    setting = dict(commit=commit, model=a.model, family=a.family, mode=a.mode, order=a.order, prompt_kind=a.prompt,
                   runs=a.runs, max_new_tokens=a.max_new_tokens,
                   items_sha256=hashlib.sha256(Path(a.sentences).read_bytes()).hexdigest())
    done = done_keys(out, setting)

    if not recs:
        raise SystemExit("no sentences selected")
    if all(prompts.key(r["sid"], arm, rep) in done for r in recs for arm, rep in prompts.RUNSETS[a.runs]):
        print(f"{out}: every key already decoded; model not loaded", flush=True)
        return
    dc = Decoder(a.family, a.model, out)
    sr = dc.sr

    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for rec in recs:
            audio = None
            if a.mode == "speech":
                audio, _ = librosa.load(os.path.join(a.corpus, rec["wav"]), sr=sr)
            for arm, rep in prompts.RUNSETS[a.runs]:
                k = prompts.key(rec["sid"], arm, rep)
                if k in done:
                    continue
                text = dc.chat(rec, arm, a.mode, a.prompt, a.order)
                dec, new, hit_cap = dc.run(text, audio, a.max_new_tokens)
                hyp = dec.strip()
                fh.write(json.dumps(dict(key=k, sid=rec["sid"], arm=arm, rep=rep, **setting, dirty=dirty,
                                         new_tokens=len(new), hit_cap=hit_cap,
                                         prompt=text, hyp=hyp), ensure_ascii=False) + "\n")
                fh.flush()
                n += 1
                if n % 50 == 0:
                    print(f"{n} decodes, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new decodes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
