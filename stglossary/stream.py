"""Streaming condition: re-translation of growing source prefixes with local agreement.

Re-translation of growing source prefixes with local agreement, LA-2 (Liu, Spanakis and Niehues 2020, arXiv
2005.11185; Polak et al. 2022, arXiv 2204.06028), on the same model, prompt and decoding as the offline decodes
(`decode_hf`). Per sentence and run (none x3, correct, irrelevant, wrong: prompts.RUNSETS["main"]):

- Prefix grid. Speech (`e2e`): the audio is cut at k * CHUNK seconds (k = 1, 2, ... while shorter than the file); the
  last prefix is the whole file (a short final chunk is kept; no silence trimming). Text (`e2e-txt`, the same model
  fed the gold transcript): the transcript is cut on the same grid into the word pieces whose forced-alignment end time
  (`align`) is <= the prefix end; a grid point that adds no piece is not decoded (the source did not change), and
  the last prefix is the whole sentence (rec["en"], as offline) at the end of the audio.
- Each prefix is decoded with exactly the offline prompt (Decoder.chat: same system and user turn, glossary block,
  prompt kind), greedy, the same max_new_tokens. The committed token ids are appended after the rendered prompt, i.e.
  forced at the start of the assistant turn, never detokenized and re-tokenized; the model generates a continuation.
- Each step's hypothesis = committed ids + continuation ids (up to the stop token). The echo cut of the offline
  coding (`hits.cut`: coding.translation_part, then the block-term line cut) is applied to the hypothesis
  text, and the hypothesis is shortened to the longest token prefix whose text is a prefix of the cut text (never
  shorter than the committed ids). The number of tokens cut is recorded per step.
- Commit (LA-2). Step 1 commits nothing. After step i > 1 the committed ids become the longest common prefix, in token
  ids, of the cut hypotheses of steps i-1 and i (both start with the ids committed before step i, which were forced,
  so committed text is never revised). A commit ending in a mark of `SENTENCE_END` is shortened before the last prefix (a forced
  sentence end makes the model stop translating). At the last prefix the whole cut hypothesis is committed (end-of-source flush).
- A sentence with a single prefix is decoded once, exactly as offline (nothing forced).

Usage:
  python -m stglossary.stream decode --mode speech|text --family omni25|omni3 --model PATH --sentences S.jsonl --corpus ROOT
         --out OUT.jsonl [--align ALIGN.jsonl] [--chunk 1.0] [--prompt instructed] [--subset first|all] [--limit N]
         (--align is required in text mode; in speech mode it only supplies t_end)
  python -m stglossary.stream compare --new NEW.jsonl --ref REF.jsonl --out CMP.json
         a re-decode against another decode of the same keys: byte-identical `hyp` count per arm

Output: one JSONL row per (sentence, arm, rep), readable by `hits`: `key`, `sid`, `arm`, `rep`, `prompt_kind`,
`hyp` (= final committed text, stripped; already echo-cut, which the hits module's own cut leaves unchanged), `hyp_uncut`
(the last step's whole hypothesis before the cut), the offline rows' provenance (`commit`, `dirty`, `model`, `family`,
`mode`, `items_sha256`, ...), `policy`, `chunk`, `duration`, `t_end` (end time of t's last word piece from ALIGN; None
when absent or not aligned), `token_commit_end` (per final committed token, the end time of the prefix whose step
committed it), and `steps`: per prefix its end time (s), source text (text mode), number of forced ids, the raw
continuation text and ids, the whole hypothesis text (committed + uncommitted, before the cut), tokens cut by the
echo cut, the committed text and number of committed ids after the step, new tokens and whether it hit the cap.
`prompt` is the offline prompt (forced ids not included). Resumable: keys already in OUT are skipped; rows from
another setting are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import prompts
from .hits import blocks, cut as echo_cut
from .prov import done_keys, provenance

RUNS = prompts.RUNSETS["main"]
POLICY = "la2-forced-token-prefix-holdend"  # was "la2-forced-token-prefix" before the sentence-end hold-back


# A commit that ends in any of these marks (after trailing whitespace) is shortened by whole tokens until it does not,
# before the end-of-source flush; this also holds back quotes that are not sentence-final, which only delays them. Forced after
# the assistant turn, a committed "。" makes the model stop, and later audio is never translated
# (without the hold-back, 10-17 of 116 outputs per arm stopped early).
SENTENCE_END = ("。", "！", "？", ".", "!", "?", "”", "\"", "」", "』", "…")


def lcp(a, b):
    """Longest common prefix of two sequences (token-id lists, or strings by character)."""
    n, m = 0, min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return a[:n]


def la_commit(committed: list, prev: list, cur: list) -> list:
    """LA-2: committed ids after a step. Both hypotheses start with `committed` (it was forced)."""
    k = len(committed)
    assert prev[:k] == committed and cur[:k] == committed, "a hypothesis does not start with the committed ids"
    return list(lcp(prev, cur))


def cut_len(ids: list, detok, cut_text) -> int:
    """Number of leading tokens kept by the echo cut: all when the cut changes nothing, else the longest prefix whose
    text is a prefix of the cut text."""
    full = detok(ids)
    c = cut_text(full)
    if c == full:
        return len(ids)
    for k in range(len(ids), -1, -1):
        if c.startswith(detok(ids[:k])):
            return k
    return 0


class Stream:
    """LA-2 state of one (sentence, arm, rep). Per prefix: `forced()` gives the ids to force, `feed()` takes the
    continuation ids (stop token removed). `detok` turns ids into text; `cut_text` is the echo cut."""

    def __init__(self, ends: list[float], detok, cut_text, sources: list[str] | None = None):
        assert ends and (sources is None or len(sources) == len(ends))
        self.ends, self.sources, self.detok, self.cut_text = ends, sources, detok, cut_text
        self.i, self.committed, self.prev, self.steps = 0, [], None, []
        self.first_end = []  # per committed token, the prefix end at which it was committed

    @property
    def done(self) -> bool:
        return self.i == len(self.ends)

    def forced(self) -> list:
        return list(self.committed)

    def feed(self, cont: list, **info) -> None:
        assert not self.done
        full = self.committed + list(cont)
        cl = cut_len(full, self.detok, self.cut_text)
        keep = max(cl, len(self.committed))
        hyp = full[:keep]
        if self.i == len(self.ends) - 1:
            new = hyp  # end-of-source flush
        elif self.prev is None:
            new = list(self.committed)
        else:
            new = la_commit(self.committed, self.prev, hyp)
            # hold back a character split across tokens and a trailing sentence end, in one loop so that removing either can never leave the other at the end
            while len(new) > len(self.committed):
                t = self.detok(new)
                if not (t.endswith("\ufffd") or t.rstrip().endswith(SENTENCE_END)):
                    break
                new = new[:-1]
        end = round(self.ends[self.i], 4)
        self.first_end += [end] * (len(new) - len(self.committed))
        step = dict(end=end)
        if self.sources is not None:
            step["source"] = self.sources[self.i]
        step.update(n_forced=len(self.committed), forced_ids=list(self.committed), cut_clamped=cl < len(self.committed),
                    cont=self.detok(list(cont)), cont_ids=list(cont),
                    hypothesis=self.detok(full), cut_tokens=len(full) - keep, committed=self.detok(new),
                    n_committed=len(new), **info)
        self.steps.append(step)
        self.committed, self.prev = new, hyp
        self.i += 1

    @property
    def hyp(self) -> str:
        assert self.done
        return self.detok(self.committed).strip()


def speech_ends(n_samples: int, sr: int, chunk: float) -> list[int]:
    """Prefix ends in samples: k * chunk while shorter than the file, then the whole file."""
    step = round(chunk * sr)
    return list(range(step, n_samples, step)) + [n_samples]


def text_steps(words: list, en: str, duration: float, chunk: float) -> tuple[list[float], list[str]]:
    """Prefix end times and source texts for the gold transcript. `words` are [piece, start, end] from p1/align.py,
    concatenating to " " + en. A grid point is kept only when it adds a piece and is not yet the whole sentence;
    the last prefix is the whole sentence at the end of the audio."""
    ends, srcs, k_prev, g = [], [], 0, 1
    while g * chunk < duration:
        T = g * chunk
        k = 0
        while k < len(words) and words[k][2] <= T:
            k += 1
        src = "".join(w[0] for w in words[:k]).strip()
        if k_prev < k < len(words) and src:
            ends.append(T)
            srcs.append(src)
            k_prev = k
        g += 1
    return ends + [duration], srcs + [en]


def strip_stop(ids: list, eos: list) -> list:
    """Continuation ids up to (not including) the first stop token."""
    for i, t in enumerate(ids):
        if t in eos:
            return ids[:i]
    return ids


def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(a) -> None:
    import librosa
    from .decode_hf import Decoder

    if a.mode == "text" and not a.align:
        raise SystemExit("--mode text needs --align")
    recs = [json.loads(l) for l in open(a.sentences)]
    if a.subset == "first":
        recs = [r for r in recs if r["first"]]
    if a.limit:
        recs = recs[: a.limit]
    if not recs:
        raise SystemExit("no sentences selected")
    al = {r["sid"]: r for r in map(json.loads, open(a.align))} if a.align else {}
    unaligned = []
    if a.mode == "text":
        unaligned = [r["sid"] for r in recs if not al.get(r["sid"], {}).get("aligned")]
        if unaligned:  # word prefixes need word times: such sentences are excluded and listed
            print(f"excluded, not aligned: {len(unaligned)} {unaligned}", flush=True)
            if len(unaligned) > 0.10 * len(recs):  # the text control is then not reported
                raise SystemExit(f"{len(unaligned)} of {len(recs)} sentences not aligned (> 10%); text control dropped")
            recs = [r for r in recs if r["sid"] not in set(unaligned)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    commit, dirty = provenance(Path(__file__).resolve().parent)
    setting = dict(commit=commit, model=a.model, family=a.family, mode=a.mode, order="after", prompt_kind=a.prompt,
                   runs="main", max_new_tokens=a.max_new_tokens, items_sha256=_sha(a.sentences), policy=POLICY,
                   chunk=a.chunk, align_sha256=_sha(a.align) if a.align else None)
    done = done_keys(out, setting)
    if all(prompts.key(r["sid"], arm, rep) in done for r in recs for arm, rep in RUNS):
        print(f"{out}: every key already decoded; model not loaded", flush=True)
        return
    dc = Decoder(a.family, a.model, out)
    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for rec in recs:
            ar = al.get(rec["sid"], {})
            if a.mode == "speech":
                audio, _ = librosa.load(str(Path(a.corpus) / rec["wav"]), sr=dc.sr)
                cuts = speech_ends(len(audio), dc.sr, a.chunk)
                ends, srcs, dur = [e / dc.sr for e in cuts], None, len(audio) / dc.sr
            else:
                audio, dur = None, ar["duration"]
                ends, srcs = text_steps(ar["words"], rec["en"], dur, a.chunk)
            for arm, rep in RUNS:
                k = prompts.key(rec["sid"], arm, rep)
                if k in done:
                    continue
                blk = None if arm == "none" else blocks(rec)[arm]
                st = Stream(ends, dc.text, lambda x, b=blk: echo_cut(x, b), srcs)
                bases = {}
                for i in range(len(ends)):
                    if a.mode == "speech":
                        base = bases.setdefault("", dc.chat(rec, arm, "speech", a.prompt))
                        src = audio[: cuts[i]]
                    else:
                        base = dc.chat(rec, arm, "text", a.prompt, source_text=srcs[i])
                        src = None
                    budget = max(1, a.max_new_tokens - len(st.forced()))  # whole output capped as offline
                    _, new, cap = dc.run(base, src, budget, forced=st.forced())
                    st.feed(strip_stop(new, dc.eos), new_tokens=len(new), hit_cap=cap, budget=budget,
                            input_ids=dc.last_input_len, audio_samples=None if src is None else len(src))
                    n += 1
                last = st.steps[-1]
                fh.write(json.dumps(dict(key=k, sid=rec["sid"], arm=arm, rep=rep, **setting, dirty=dirty,
                                         duration=dur, t_end=ar.get("t_end"), n_steps=len(st.steps),
                                         new_tokens=sum(s["new_tokens"] for s in st.steps),
                                         hit_cap=any(s["hit_cap"] for s in st.steps),
                                         token_commit_end=st.first_end, steps=st.steps, prompt=base,
                                         hyp_uncut=last["hypothesis"].strip(), hyp=st.hyp,
                                         excluded_unaligned=unaligned if a.mode == "text" else None),
                                    ensure_ascii=False) + "\n")
                fh.flush()
            print(f"{rec['sid']}: {len(ends)} prefixes; {n} decodes, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} new decodes in {time.time() - t0:.0f}s", flush=True)


def compare(a) -> None:
    ref = {r["key"]: r for r in map(json.loads, open(a.ref))}
    new = [json.loads(l) for l in open(a.new)]
    per, diff = {}, []
    for r in new:
        o = ref.get(r["key"])
        if o is None:
            raise SystemExit(f"{r['key']} is not in {a.ref}")
        if o.get("prompt") != r.get("prompt") or o.get("prompt_kind") != r.get("prompt_kind"):
            raise SystemExit(f"{r['key']}: the rendered prompt differs from the reference")
        p = per.setdefault(r["arm"], [0, 0])
        p[1] += 1
        if o["hyp"] == r["hyp"]:
            p[0] += 1
        else:
            diff.append(dict(key=r["key"], ref=o["hyp"], new=r["hyp"]))
    res = dict(new=a.new, ref=a.ref, new_sha256=_sha(a.new), ref_sha256=_sha(a.ref),
               identical={k: dict(same=v[0], of=v[1]) for k, v in per.items()},
               same=sum(v[0] for v in per.values()), of=len(new), differing=diff)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for k, v in per.items():
        print(f"{k}: {v[0]} of {v[1]} byte-identical")
    print(f"all: {res['same']} of {res['of']} byte-identical -> {a.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--mode", choices=["speech", "text"], required=True)
    d.add_argument("--family", choices=["omni25", "omni3"], required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--sentences", required=True)
    d.add_argument("--corpus", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--align")
    d.add_argument("--chunk", type=float, default=1.0)
    d.add_argument("--prompt", choices=["plain", "instructed"], default="instructed")
    d.add_argument("--subset", choices=["first", "all"], default="first")
    d.add_argument("--limit", type=int, default=0)
    d.add_argument("--max-new-tokens", type=int, default=256)
    c = sub.add_parser("compare")
    c.add_argument("--new", required=True)
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"decode": decode, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
