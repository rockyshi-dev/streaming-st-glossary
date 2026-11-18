"""Word timings of the gold transcript (t_end for the early-commit count, word prefixes for the text control).

Forced alignment with faster-whisper-large-v3 (the CTranslate2 conversion of Whisper large-v3): the gold
sentence is tokenized, forced through Whisper's decoder (`Whisper.align`, no search), and each word is timed by dynamic
time warping over the cross-attention of the model's alignment heads - the method of OpenAI Whisper's word timestamps,
here applied to the given text instead of a hypothesis. No ASR output is used.

Usage: python -m stglossary.align --sentences S.jsonl --corpus ROOT --out ALIGN.jsonl [--model PATH] [--subset first|all]
One row per sentence: sid, duration (s), words [[text, start, end]] whose concatenation is " " + en (Whisper word
pieces, a leading space marks a new word; punctuation may be its own piece), aligned (False when the concatenation
check fails or the audio exceeds Whisper's 30 s window; words is then empty), t_end (end time of the piece holding
the last character of the first occurrence of t; None when t is not found or the sentence is not aligned), commit,
dirty, model. A sentence also fails when a word's times are not monotone or leave [0, duration]. Written to
OUT.part and renamed to OUT only when every sentence is done, so a crash never leaves a usable-looking OUT.
OUT must not exist.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .prov import provenance


def t_span(en: str, t: str) -> tuple[int, int] | None:
    """Character span of the first occurrence of t in en: case-insensitive, at token boundaries."""
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", en, re.I)
    return (m.start(), m.end()) if m else None


def t_end(en: str, t: str, words: list) -> float | None:
    """End time of the word piece holding the last character of t. `words` concatenate to " " + en."""
    sp = t_span(en, t)
    if sp is None or not words:
        return None
    last = sp[1] - 1 + 1  # +1: the leading space of " " + en
    pos = 0
    for w, _, e, *_ in words:
        if pos <= last < pos + len(w):
            return float(e)
        pos += len(w)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--subset", choices=["first", "all"], default="first")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio, pad_or_trim
    from faster_whisper.tokenizer import Tokenizer

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    commit, dirty = provenance(Path(__file__).resolve().parent)
    m = WhisperModel(a.model, device="cuda", compute_type="float16")
    tok = Tokenizer(m.hf_tokenizer, m.model.is_multilingual, task="transcribe", language="en")
    fe = m.feature_extractor
    recs = [json.loads(l) for l in open(a.sentences)]
    if a.subset == "first":
        recs = [r for r in recs if r["first"]]
    bad = 0
    part = out.with_name(out.name + ".part")
    with open(part, "w") as fh:
        for r in recs:
            audio = decode_audio(str(Path(a.corpus) / r["wav"]), sampling_rate=fe.sampling_rate)
            dur = len(audio) / fe.sampling_rate
            feats = fe(audio)
            n = feats.shape[-1] - 1  # content frames, as faster-whisper's transcribe counts them
            words, ok = [], n <= fe.nb_max_frames
            if ok:
                enc = m.encode(pad_or_trim(feats[:, :n]))
                toks = tok.encode(" " + r["en"].strip())
                ws = m.find_alignment(tok, [toks], enc, n)[0]
                words = [[w["word"], round(float(w["start"]), 3), round(float(w["end"]), 3),
                          round(float(w["probability"]), 4)] for w in ws]
                ok = "".join(w[0] for w in words) == " " + r["en"].strip()
                ok = ok and all(0 <= w[1] <= w[2] <= dur + 0.05 for w in words) \
                    and all(words[i][2] <= words[i + 1][1] + 1e-6 for i in range(len(words) - 1))
                if not ok:
                    words = []
            bad += not ok
            fh.write(json.dumps(dict(sid=r["sid"], duration=dur, aligned=ok, words=words,
                                     t_end=t_end(r["en"].strip(), r["t"], words) if ok else None,
                                     commit=commit, dirty=dirty, model=a.model), ensure_ascii=False) + "\n")
    part.rename(out)
    print(f"aligned {len(recs) - bad} of {len(recs)} sentences -> {out}", flush=True)


if __name__ == "__main__":
    main()
