"""Build the FLEURS item table in the schema of the public repository's data/sentences.jsonl.

Mirrors stglossary/build_items.py (the ACL 60/60 items); the differences are listed in README.md (section "FLEURS
example", "Differences from the ACL 60/60 items"). Steps:

1. Terms. Parse each raw output of terms.py (the first JSON list in it). An entry is valid when its class is one of
   terms.CLASSES, its English term occurs in the line's English at token boundaries (case-insensitive; the
   sentence's own spelling is kept) and has at most 6 words, and its Chinese form (at most 15 characters) occurs in the
   line's FLoRes Chinese (stglossary.coding.contains). Every entry and its verdict goes to terms.tsv.
2. Specific terms, as in build_items.specific: several words, a digit, an inner capital, or found in at most 3 of the
   kept lines' English.
3. Glossary block per article ("talk"): one line per specific lemma (stglossary.coding.lemma), in order of first
   occurrence over all FLoRes lines of the article (also lines without a recording); a lemma keeps the form and class
   of its first occurrence.
4. Pool and items. A line with a FLEURS recording of MIN_DUR..30 s contributes its first specific term (in order of
   position in the English) whose block form w occurs in the line's Chinese. Items are (talk, lemma); an item's
   first sentence is its decision sentence (`first`).
5. w' (wrong) and w'' (irrelevant) are drawn from the other articles' block terms exactly as in build_items
   (base_ok, MAX_REUSE 3, MAX_REFS 9, the seed and the shuffled item order), with the class labels of step 1 in place
   of the hand labels and no part-of-speech or hand pair-reject check.

Usage: python -m fleurs.items --lines LINES.jsonl --terms TERMS_RAW.jsonl --out DIR [--min-dur 8.0] [--seed 20261117]
Writes DIR/{terms.tsv, items.tsv, dropped.tsv, sentences.jsonl, summary.json}. Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fleurs.terms import CLASSES  # noqa: E402
from stglossary.coding import contains, is_latin, lemma  # noqa: E402

MAX_REUSE = 3
MAX_REFS = 9
MAX_DUR = 30.0  # Whisper's window: longer audio cannot be aligned


def parse(raw: str):
    """The first JSON list in the output, or None."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < i:
        return None
    try:
        v = json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
    return v if isinstance(v, list) else None


def span(en: str, t: str):
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", en, re.I)
    return m.span() if m else None


def stem(t: str) -> str:
    return t.lower()[:5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--terms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dur", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = [json.loads(l) for l in open(a.lines)]
    raw = {r["line"]: r for r in map(json.loads, open(a.terms))}
    assert set(raw) == {r["line"] for r in lines}, "terms and lines cover different lines"

    # -- 1. terms --------------------------------------------------------------------------------------
    verdicts, found = [], {}  # found[line] = [(pos, t, w, cls)]
    n_unparsed = 0
    for r in lines:
        v = parse(raw[r["line"]]["raw"])
        if v is None:
            n_unparsed += 1
            verdicts.append((r["line"], "", "", "", "unparsed output"))
            found[r["line"]] = []
            continue
        got = []
        for e in v:
            if not isinstance(e, dict):
                verdicts.append((r["line"], json.dumps(e, ensure_ascii=False), "", "", "not an object"))
                continue
            t, w, c = (str(e.get(k, "")).strip() for k in ("en", "zh", "class"))
            c = c.lower()
            sp = span(r["en"], t) if t else None
            why = ("class" if c not in CLASSES else
                   "en empty or over 6 words" if not t or len(t.split()) > 6 else
                   "en is a number" if re.fullmatch(r"[\d.,\s]+", t) else
                   "en not in the sentence" if sp is None else
                   "zh empty or over 15 characters" if not w or len(w) > 15 else
                   "zh not in the translation" if not contains(r["zh"], w) else "ok")
            verdicts.append((r["line"], t, w, c, why))
            if why == "ok":
                got.append((sp[0], r["en"][sp[0]:sp[1]], w, c))
        found[r["line"]] = sorted(set(got))
    with open(out / "terms.tsv", "w") as f:
        f.write("line\ten\tzh\tclass\tverdict\n")
        for x in verdicts:
            f.write("\t".join(str(y).replace("\t", " ") for y in x) + "\n")

    # -- 2. specific -------------------------------------------------------------------------------------
    def freq(t: str) -> int:
        return sum(span(r["en"], t) is not None for r in lines)

    fcache = {}

    def specific(t: str) -> bool:
        if " " in t or re.search(r"\d", t) or re.fullmatch(r"[A-Z][A-Za-z]*[A-Z][A-Za-z]*", t):
            return True
        if t.lower() not in fcache:
            fcache[t.lower()] = freq(t)
        return fcache[t.lower()] <= 3

    # -- 3. blocks -------------------------------------------------------------------------------------
    block = collections.defaultdict(dict)  # talk -> lemma -> (t, w, cls), insertion order = first occurrence
    for r in lines:
        for _, t, w, c in found[r["line"]]:
            if specific(t):
                block[r["talk"]].setdefault(lemma(t), (t, w, c))
    block_text = {k: " ".join(f"{t} {w}" for t, w, _ in v.values()) for k, v in block.items()}

    # -- 4. pool and items -----------------------------------------------------------------------------
    pool, too_short = [], 0
    for r in lines:
        fr = r["fleurs"]
        if fr is None:
            continue
        if not (a.min_dur <= fr["duration"] <= MAX_DUR):
            too_short += 1
            continue
        for _, t, _, _ in found[r["line"]]:
            L = lemma(t)
            if L not in block[r["talk"]]:
                continue
            bt, bw, bc = block[r["talk"]][L]
            if contains(r["zh"], bw):
                pool.append(dict(r, sid=f"{fr['split']}/{fr['id']}", t=t, w=bw, cls=bc, lemma=L))
                break
    items = {}
    for p in pool:
        key = (p["talk"], p["lemma"])
        if key not in items:
            items[key] = dict(talk=p["talk"], lemma=p["lemma"], t=p["t"], w=p["w"], cls=p["cls"], sid=p["sid"],
                              sids=[])
        items[key]["sids"].append(p["sid"])
        p["item"] = f"{key[0]}|{key[1]}"

    # -- 5. w' and w'' -------------------------------------------------------------------------------
    zh_refs = [r["zh"] for r in lines]
    cands = {}
    for talk, bl in block.items():
        for L, (t, w, c) in bl.items():
            if L not in cands:
                cands[L] = dict(t=t, lemma=L, w=w, cls=c, talks=set())
            cands[L]["talks"].add(talk)
    for c in cands.values():
        c["nref"] = sum(contains(z, c["w"]) for z in zh_refs)
    cand_terms = sorted(cands.values(), key=lambda c: c["t"])

    def base_ok(it, c, sent_zh, lenwin):
        w, wc = it["w"], c["w"]
        if it["talk"] in c["talks"] or c["nref"] >= MAX_REFS:
            return False
        if is_latin(w) != is_latin(wc):
            return False
        if not is_latin(wc) and len(wc) < 2:
            return False
        if abs(len(wc) - len(w)) > lenwin:
            return False
        if wc == w or wc in w or w in wc:
            return False
        if contains(sent_zh, wc) or contains(block_text[it["talk"]], wc):
            return False
        return True

    rng = random.Random(a.seed)
    order = sorted(items)
    rng.shuffle(order)
    use1, use2 = collections.Counter(), collections.Counter()
    dropped = []
    zh_of = {p["sid"]: p["zh"] for p in pool}
    en_of = {p["sid"]: p["en"] for p in pool}
    for key in order:
        it = items[key]
        zs = " ".join(zh_of[x] for x in it["sids"])
        es = " ".join(en_of[x] for x in it["sids"]).lower()
        c1 = [c for c in cand_terms
              if base_ok(it, c, zs, 1) and c["cls"] == it["cls"]
              and stem(c["t"]) != stem(it["t"]) and c["lemma"] != it["lemma"] and use1[c["w"]] < MAX_REUSE]
        if not c1:
            dropped.append((key, "no w' candidate"))
            continue
        pick1 = rng.choice(c1)
        c2 = [c for c in cand_terms
              if base_ok(it, c, zs, 2) and c["cls"] != it["cls"] and c["w"] != pick1["w"]
              and not re.search(r"\b" + re.escape(c["t"].lower()) + r"\b", es) and use2[c["w"]] < MAX_REUSE]
        if not c2:
            dropped.append((key, "no w'' candidate"))
            continue
        pick2 = rng.choice(c2)
        use1[pick1["w"]] += 1
        use2[pick2["w"]] += 1
        it.update(t1=pick1["t"], w1=pick1["w"], cls1=pick1["cls"], n_cand1=len(c1),
                  t2=pick2["t"], w2=pick2["w"], cls2=pick2["cls"], n_cand2=len(c2))
    kept = {k: v for k, v in items.items() if "w1" in v}

    # -- outputs -------------------------------------------------------------------------------------
    cols = ["item", "talk", "sid", "n_sents", "t", "w", "cls", "stratum", "t1", "w1", "cls1", "n_cand1",
            "t2", "w2", "cls2", "n_cand2"]
    with open(out / "items.tsv", "w") as f:
        f.write("\t".join(cols).replace("t1", "t_wrong").replace("w1", "w_wrong").replace("cls1", "cls_wrong")
                .replace("t2", "t_irr").replace("w2", "w_irr").replace("cls2", "cls_irr") + "\n")
        for k in sorted(kept):
            v = kept[k]
            row = dict(v, item=f"{k[0]}|{k[1]}", n_sents=len(v["sids"]),
                       stratum="latin" if is_latin(v["w"]) else "han")
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    with open(out / "dropped.tsv", "w") as f:
        for k, why in dropped:
            f.write(f"{k[0]}|{k[1]}\t{items[k]['t']}\t{items[k]['w']}\t{why}\n")
    n_out = 0
    part = out / "sentences.jsonl.part"
    with open(part, "w") as f:
        for p in pool:
            k = (p["talk"], p["lemma"])
            if k not in kept:
                continue
            it = kept[k]
            # The probe line is found by lemma: the block keeps the article's first spelling of a lemma
            # ("Gymnastics"), the item the recorded sentence's ("gymnastics").
            lines_ = [(L, t, w) for L, (t, w, _) in block[p["talk"]].items()]
            assert sum(L == it["lemma"] for L, _, _ in lines_) == 1 and block[p["talk"]][it["lemma"]][1] == it["w"]

            def blk(repl):
                return [repl if L == it["lemma"] else [t, w] for L, t, w in lines_]

            rec = dict(sid=p["sid"], talk=p["talk"], wav=p["fleurs"]["wav"], en=p["en"], zh=p["zh"], item=p["item"],
                       first=(p["sid"] == it["sid"]), t=it["t"], w=it["w"], w_wrong=it["w1"], w_irr=it["w2"],
                       blocks=dict(correct=blk([it["t"], it["w"]]), irrelevant=blk([it["t2"], it["w2"]]),
                                   wrong=blk([it["t"], it["w1"]])),
                       duration=p["fleurs"]["duration"], flores_line=p["line"], url=p["url"], cls=it["cls"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    part.rename(out / "sentences.jsonl")  # atomic: a crash leaves no sentences.jsonl for the stage guard to skip
    firsts = [kept[k] for k in kept]
    bl_sizes = sorted(len(block[v["talk"]]) for v in firsts)
    summ = dict(seed=a.seed, min_dur=a.min_dur, lines=len(lines), unparsed_outputs=n_unparsed,
                term_verdicts=collections.Counter(x[4] for x in verdicts),
                recordings_outside_duration=too_short, pool_sentences=len(pool), items_total=len(items),
                items_kept=len(kept), items_dropped=collections.Counter(w for _, w in dropped),
                sentences_written=n_out, decision_sentences=len(kept),
                talks_with_items=len({v["talk"] for v in kept.values()}),
                item_classes=collections.Counter(v["cls"] for v in kept.values()),
                strata=collections.Counter("latin" if is_latin(v["w"]) else "han" for v in kept.values()),
                block_lines_per_item_min_median_max=[bl_sizes[0], bl_sizes[len(bl_sizes) // 2], bl_sizes[-1]]
                if bl_sizes else None,
                distinct_w_wrong=len({v["w1"] for v in kept.values()}),
                distinct_w_irr=len({v["w2"] for v in kept.values()}))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
