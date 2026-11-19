"""FLEURS en_us (dev + test) joined to FLoRes-200 devtest: one row per FLoRes line of every kept article.

FLEURS sentences are read FLoRes sentences. Each FLEURS recording is matched to its FLoRes devtest line by its
English text (lowercased, letters and digits only; every match is required to be unique), and the line's article is
the URL column of FLoRes `metadata_devtest.tsv`. One article = one "talk" of the glossary design.

Kept articles: at least one line has a FLEURS recording, and neither the topic nor the URL matches TECH (language
technology, NLP, speech, computing, media and communication), so the examples are everyday-topic.

Per FLEURS sentence one recording is used: the first row for that id in dev.tsv then test.tsv (FLEURS has 1-3
speakers per sentence). Its duration is num_samples / 16000.

Usage: python -m fleurs.prep --corpus CORPUS --out DIR   (CORPUS: the directory fleurs/fetch_fleurs.sh fills)
Writes DIR/lines.jsonl (line, talk, url, domain, topic, title, en, zh, fleurs {id, split, wav, duration, n_rec,
fleurs_en_equal} or null), DIR/articles.tsv (every article with a FLEURS sentence: kept or the drop reason) and
DIR/prep.json (counts). Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from urllib.parse import unquote

TECH = re.compile(r"technolog|comput|artificial|intelligence|language|linguist|speech|translat|communicat|internet"
                  r"|software|digital|robot|blog|media|\bai\b|electronic", re.I)

# Articles whose subject is computing or consumer electronics although topic and title do not match TECH; listed by hand
# from the titles and English lines of the kept articles, before any term or model output existed.
DROP_TITLES = {
    "IEEE approves 802.11n standard after six years",
    "Taiwanese consumer laptop market faces change following ASUStek winning sustainability award",
    "Nintendo unveils controller for Revolution console",
    "California bans sale of violent video games to minors",
    "Amazon to buy smart doorbell startup Ring",
    "Modern radio drama Paranoria, TX releases 100th episode",
    "Cognitive Psychology and Cognitive Neuroscience/Cognitive Psychology and the Brain",
    "Blended Learning in K-12/Blended Learning in Grades 3-6",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def lines(p: Path) -> list[str]:
    t = p.read_text(encoding="utf-8").split("\n")
    return t[:-1] if t and t[-1] == "" else t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.corpus), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = root / "flores200" / "flores200_dataset"
    en = lines(fl / "devtest" / "eng_Latn.devtest")
    zh = lines(fl / "devtest" / "zho_Hans.devtest")
    md = [l.split("\t") for l in lines(fl / "metadata_devtest.tsv")]
    assert md[0][:3] == ["URL", "domain", "topic"], md[0]
    md = md[1:]
    assert len(en) == len(zh) == len(md) == 1012, (len(en), len(zh), len(md))
    by_text = collections.defaultdict(list)
    for j, e in enumerate(en):
        by_text[norm(e)].append(j)

    rec_of: dict[int, dict] = {}  # FLoRes line -> chosen recording
    n_rec = collections.Counter()
    for split in ("dev", "test"):
        for l in lines(root / "en_us" / f"{split}.tsv"):
            r = l.split("\t")
            hit = by_text[norm(r[2])]
            assert len(hit) == 1, (r[0], r[2], hit)
            j = hit[0]
            n_rec[j] += 1
            if j not in rec_of:
                rec_of[j] = dict(id=int(r[0]), split=split, wav=f"en_us/audio/{split}/{r[1]}",
                                 duration=round(int(r[5]) / 16000, 3), fleurs_en_equal=r[2].strip() == en[j].strip())
    for j, r in rec_of.items():
        r["n_rec"] = n_rec[j]
    ids = collections.Counter(r["id"] for r in rec_of.values())
    assert max(ids.values()) == 1, "one FLEURS id maps to two FLoRes lines"

    art_lines = collections.defaultdict(list)
    for j, m in enumerate(md):
        art_lines[m[0]].append(j)
    rows, arts = [], []
    for url, js in art_lines.items():
        nf = sum(j in rec_of for j in js)
        if nf == 0:
            continue
        domain, topic = md[js[0]][1], md[js[0]][2]
        title = unquote(url.split("/wiki/", 1)[-1]).replace("_", " ").replace("|", "/")
        talk = f"{domain}:{title}"
        m = TECH.search(topic) or TECH.search(title)
        why = f"tech topic ({m.group(0)!r})" if m else "tech subject (hand list)" if title in DROP_TITLES else "kept"
        arts.append((talk, url, domain, topic, len(js), nf, why))
        if why != "kept":
            continue
        for j in js:
            rows.append(dict(line=j, talk=talk, url=url, domain=domain, topic=topic, title=title, en=en[j], zh=zh[j],
                             fleurs=rec_of.get(j)))
    with open(out / "lines.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "articles.tsv", "w") as f:
        f.write("talk\turl\tdomain\ttopic\tflores_lines\tfleurs_sentences\tstatus\n")
        for x in arts:
            f.write("\t".join(map(str, x)) + "\n")
    kept = [x for x in arts if x[6] == "kept"]
    summ = dict(fleurs_ids=len(rec_of), articles_with_fleurs=len(arts), articles_kept=len(kept),
                articles_dropped=[x[0] for x in arts if x[6] != "kept"],
                lines_kept=len(rows), fleurs_sentences_kept=sum(r["fleurs"] is not None for r in rows),
                fleurs_en_differs=sum(1 for r in rows if r["fleurs"] and not r["fleurs"]["fleurs_en_equal"]))
    (out / "prep.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "articles_dropped"}), "dropped:", summ["articles_dropped"])


if __name__ == "__main__":
    main()
