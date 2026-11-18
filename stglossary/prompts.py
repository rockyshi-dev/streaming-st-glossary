"""One prompt template for every system; the `instructed` kind adds one sentence asking for the glossary forms.

Order inside the user turn: the source (audio, or the English text), then the glossary block, then the
instruction. Only the probe term's glossary line differs between arms; `none` has no block.
"""
from __future__ import annotations

# (arm, repeat). `none` is decoded three times to measure run-to-run noise.
RUNS = [("none", 0), ("none", 1), ("none", 2), ("correct", 0), ("irrelevant", 0), ("wrong", 0)]
# run sets: main = RUNS (the only one the published numbers use); the others are smaller variants
RUNSETS = {"main": RUNS, "screen": [("irrelevant", 0), ("wrong", 0)],
           "plain3": [("correct", 0), ("irrelevant", 0), ("wrong", 0)], "hear": [("hear", 0)]}
# `instructed`: one sentence before the translation request, only when a glossary block is present
INSTRUCT = "Use the Chinese forms given in the glossary for the listed terms."
HEAR = "Transcribe the English speech."

INSTR = {
    "speech": "Translate the following English speech into Chinese. Output only the Chinese translation.",
    "text": "Translate the following English text into Chinese. Output only the Chinese translation.",
}


def glossary(block: list[list[str]] | None) -> str:
    if not block:
        return ""
    return "Glossary:\n" + "\n".join(f"{s} → {t}" for s, t in block) + "\n\n"


def text_after_source(rec: dict, arm: str, modality: str, kind: str = "plain") -> str:
    """The text that follows the source. For `text` modality the caller puts the English sentence first."""
    if arm == "hear":
        assert modality == "speech"
        return HEAR
    blk = None if arm == "none" else rec["blocks"][arm]
    ins = INSTRUCT + "\n" if (kind == "instructed" and blk) else ""
    assert kind in ("plain", "instructed")
    return glossary(blk) + ins + INSTR[modality]


def text_prompt(rec: dict, arm: str, source_text: str, kind: str = "plain") -> str:
    return f"English: {source_text}\n\n" + text_after_source(rec, arm, "text", kind)


def text_before_audio(rec: dict, arm: str) -> str:
    """Smoke-test variant only: context placed before the audio."""
    blk = None if arm == "none" else rec["blocks"][arm]
    return glossary(blk)


def key(sid: str, arm: str, rep: int) -> str:
    return f"{sid}|{arm}|{rep}"
