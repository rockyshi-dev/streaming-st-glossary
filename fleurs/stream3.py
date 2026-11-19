"""Run `stglossary.stream decode` with the three glossary arms only.

stglossary.stream decodes prompts.RUNSETS["main"] (none x3, correct, irrelevant, wrong). The FLEURS example search
needs no `none` repeats, so this wrapper sets stream.RUNS to prompts.RUNSETS["plain3"] (correct, irrelevant, wrong)
and calls stream.main with the same arguments. Everything else (LA-2 policy, forced committed ids, echo cut, prompt,
chunking) is stglossary.stream unchanged. The rows still carry runs="main" in their setting, because stream.decode
writes that literal; their keys (`sid|arm|0`) are the ones the main run set gives those three arms.

Usage: python -m fleurs.stream3 decode --mode speech ... (all arguments as stglossary.stream)
The provenance commit in each row is this repository's (stream.py's own checkout).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from stglossary import prompts, stream

    assert stream.RUNS == prompts.RUNSETS["main"], "stglossary.stream.RUNS is not the main run set; check the checkout"
    stream.RUNS = prompts.RUNSETS["plain3"]
    stream.main()
