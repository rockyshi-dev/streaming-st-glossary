"""Read ACL 60/60 (dev + eval) as aligned sentences with talk ids, tagged terms and the glossary.

Paths are relative to the extracted corpus root (the directory holding `2/`).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

SPLITS = ("dev", "eval")


def _lines(p: Path) -> list[str]:
    t = p.read_text(encoding="utf-8-sig").split("\n")
    return t[:-1] if t and t[-1] == "" else t


def load(root: str | Path) -> tuple[list[dict], dict[str, list[str]]]:
    root = Path(root) / "2"
    sents = []
    for s in SPLITS:
        base = root / "acl_6060" / s / "text"
        en = _lines(base / "tagged_terminology" / f"ACL.6060.{s}.tagged.en-xx.en.txt")
        zh = _lines(base / "tagged_terminology" / f"ACL.6060.{s}.tagged.en-xx.zh.txt")
        xml = (base / "xml" / f"ACL.6060.{s}.en-xx.en.xml").read_text(encoding="utf-8")
        talks = []
        for did, body in re.findall(r'<doc[^>]*docid="([^"]+)"(.*?)</doc>', xml, re.S):
            talks += [did] * len(re.findall(r"<seg\b", body))
        assert len(en) == len(zh) == len(talks), (s, len(en), len(zh), len(talks))
        for j, (e, z, d) in enumerate(zip(en, zh, talks)):
            sents.append(dict(sid=f"{s}/{j + 1}", split=s, idx=j + 1, talk=d,
                              wav=f"2/acl_6060/{s}/segmented_wavs/gold/sent_{j + 1}.wav",
                              en_tagged=e, zh_tagged=z,
                              en=re.sub(r"[\[\]]", "", e), zh=re.sub(r"[\[\]]", "", z),
                              terms=[t.strip() for t in re.findall(r"\[([^\]]+)\]", e)],
                              zh_brackets=[t.strip() for t in re.findall(r"\[([^\]]+)\]", z)]))
    gl: dict[str, list[str]] = {}
    with open(root / "intermediate_files" / "terminology_glossary.csv", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            gl.setdefault(r["english"].strip().lower(), []).extend(
                v.strip() for v in r["chinese"].split(":::") if v.strip())
    return sents, gl
