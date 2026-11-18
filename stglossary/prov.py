"""Provenance of a decode: the commit of the decoding code and whether the checkout was dirty, and the resume check.

Standard library only, so every environment (HF transformers, vLLM, faster-whisper) can import it.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def provenance(here: Path) -> tuple[str, bool]:
    """Commit and dirty flag (tracked files only) of the checkout holding `here`."""
    commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip())
    return commit, dirty


def done_keys(out: Path, setting: dict) -> set:
    """Keys already in OUT; refuses an OUT whose rows came from another setting."""
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            bad = {k: (r.get(k), v) for k, v in setting.items() if r.get(k) != v}
            if bad:
                raise SystemExit(f"{out} holds rows from another setting {bad}; move it away to start over")
            done.add(r["key"])
    return done
