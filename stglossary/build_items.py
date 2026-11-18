"""Build the item table: decision items, sentences and the three glossary blocks per item.

Usage: python -m stglossary.build_items CORPUS_ROOT OUT_DIR [--seed 20261117]

Writes, before any decoding:
  items.tsv        one row per item (talk, t): sentence, t, w, labels, t', w', t'', w''
  sentences.jsonl  every pool sentence with its item and the three glossary blocks (correct, irrelevant, wrong)
  dropped.tsv      items with no candidate, and why
  summary.json     counts (pool, items, strata, distinct w', per-talk cluster sizes)
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from . import corpus
from .coding import contains, is_latin, lemma

HERE = Path(__file__).resolve().parent
MAX_REUSE = 3
MAX_REFS = 9  # w' must occur in fewer than 9 of the 884 zh references


def load_labels() -> dict[str, dict]:
    rows = [l.split("\t") for l in (HERE / "labels.tsv").read_text().splitlines()
            if l and not l.startswith("#")]
    head, body = rows[0], rows[1:]
    return {r[0]: dict(zip(head, r)) for r in body}


def load_pair_rejects() -> set[tuple[str, str]]:
    """Hand label 4: (t, t') pairs where w' is an acceptable translation of t. Written after reading candidates."""
    p = HERE / "pair_rejects.tsv"
    if not p.exists():
        return set()
    return {tuple(l.split("\t")[:2]) for l in p.read_text().splitlines() if l and not l.startswith("#")}


def stem(t: str) -> str:
    return t.lower()[:5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_root")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sents, gl = corpus.load(a.corpus_root)
    lab = load_labels()
    rejects = load_pair_rejects()
    missing = sorted(t for t in gl if t not in lab)
    assert not missing, f"unlabelled glossary terms: {missing}"

    freq = collections.Counter(t.lower() for s in sents for t in s["terms"])
    zh_refs = [s["zh"] for s in sents]
    bracketed = collections.Counter(b for s in sents for b in s["zh_brackets"])

    def specific(t: str) -> bool:
        tl = t.lower()
        if tl not in gl or lab[tl]["class"] == "stop":
            return False
        return (" " in t or bool(re.search(r"\d", t)) or bool(re.fullmatch(r"[A-Z][A-Za-z]*[A-Z][A-Za-z]*", t))
                or freq[tl] <= 3)

    def form_in(w: str, text: str) -> bool:
        return contains(text, w)

    # -- pool ---------------------------------------------------------------------------
    pool = []
    for s in sents:
        for t in s["terms"]:
            if not specific(t):
                continue
            ws = [w for w in dict.fromkeys(gl[t.lower()]) if form_in(w, s["zh"])]
            if ws:
                pool.append(dict(s, t=t, w=ws[0]))
                break

    items: dict[tuple[str, str], dict] = {}
    for p in pool:
        key = (p["talk"], lemma(p["t"]))
        if key not in items:
            items[key] = dict(talk=p["talk"], lemma=key[1], t=p["t"], w=p["w"], sid=p["sid"],
                              cls=lab[p["t"].lower()]["class"], pos=lab[p["t"].lower()]["pos"],
                              zh_ok=lab[p["t"].lower()]["zh_ok"], sids=[])
        items[key]["sids"].append(p["sid"])
        p["item"] = f"{key[0]}|{key[1]}"
        p["w"] = items[key]["w"]  # merged variants share one line and one w

    # glossary block per talk: one line per item, in order of first occurrence
    block = collections.defaultdict(list)
    for it in items.values():
        block[it["talk"]].append((it["t"], it["w"]))
    block_text = {k: " ".join(f"{t} {w}" for t, w in v) for k, v in block.items()}

    # -- candidates ---------------------------------------------------------------------
    cand_terms = []
    for tl, forms in gl.items():
        L = lab[tl]
        if L["class"] == "stop" or L["zh_ok"] != "y":
            continue
        w = forms[0]
        talks = {s["talk"] for s in sents if any(x.lower() == tl for x in s["terms"])}
        nref = sum(form_in(w, z) for z in zh_refs)
        cand_terms.append(dict(t=tl, w=w, cls=L["class"], pos=L["pos"], talks=talks, nref=nref,
                               brk=bracketed[w] > 0))

    def base_ok(it, c, sent_zh, lenwin):
        w, wc = it["w"], c["w"]
        if it["talk"] in c["talks"] or not c["brk"] or c["nref"] >= MAX_REFS:
            return False
        if is_latin(w) != is_latin(wc):
            return False
        if not is_latin(wc) and len(wc) < 2:
            return False
        if abs(len(wc) - len(w)) > lenwin:
            return False
        if wc == w or wc in w or w in wc:
            return False
        if form_in(wc, sent_zh) or form_in(wc, block_text[it["talk"]]):
            return False
        return True

    rng = random.Random(a.seed)
    order = sorted(items)
    rng.shuffle(order)
    use_w1, use_w2 = collections.Counter(), collections.Counter()
    dropped = []
    zh_of = {s["sid"]: s["zh"] for s in sents}
    en_of = {s["sid"]: s["en"] for s in sents}
    for key in order:
        it = items[key]
        zs = " ".join(zh_of[x] for x in it["sids"])
        es = " ".join(en_of[x] for x in it["sids"]).lower()
        c1 = [c for c in cand_terms
              if base_ok(it, c, zs, 1) and c["cls"] == it["cls"] and c["pos"] == it["pos"]
              and stem(c["t"]) != stem(it["t"]) and lemma(c["t"]) != it["lemma"]
              and (it["t"].lower(), c["t"]) not in rejects and use_w1[c["w"]] < MAX_REUSE]
        if not c1:
            dropped.append((key, "no w' candidate"))
            continue
        c1.sort(key=lambda c: c["t"])
        pick1 = rng.choice(c1)
        c2 = [c for c in cand_terms
              if base_ok(it, c, zs, 2) and c["cls"] != it["cls"] and c["w"] != pick1["w"]
              and not re.search(r"\b" + re.escape(c["t"]) + r"\b", es) and use_w2[c["w"]] < MAX_REUSE]
        if not c2:
            dropped.append((key, "no w'' candidate"))
            continue
        c2.sort(key=lambda c: c["t"])
        pick2 = rng.choice(c2)
        use_w1[pick1["w"]] += 1
        use_w2[pick2["w"]] += 1
        it.update(t1=pick1["t"], w1=pick1["w"], cls1=pick1["cls"], n_cand1=len(c1),
                  t2=pick2["t"], w2=pick2["w"], cls2=pick2["cls"], n_cand2=len(c2))

    kept = {k: v for k, v in items.items() if "w1" in v}

    # -- outputs -------------------------------------------------------------------------------------
    cols = ["item", "talk", "sid", "n_sents", "t", "w", "cls", "pos", "zh_ok", "stratum",
            "t1", "w1", "cls1", "n_cand1", "t2", "w2", "cls2", "n_cand2"]
    with open(out / "items.tsv", "w") as f:
        f.write("\t".join(cols).replace("t1", "t_wrong").replace("w1", "w_wrong")
                .replace("t2", "t_irr").replace("w2", "w_irr") + "\n")
        for k in sorted(kept):
            v = kept[k]
            row = dict(v, item=f"{k[0]}|{k[1]}", n_sents=len(v["sids"]),
                       stratum="latin" if is_latin(v["w"]) else "han")
            f.write("\t".join(str(row[c]) for c in cols) + "\n")
    with open(out / "dropped.tsv", "w") as f:
        for k, why in dropped:
            f.write(f"{k[0]}|{k[1]}\t{items[k]['t']}\t{items[k]['w']}\t{why}\n")

    with open(out / "sentences.jsonl", "w") as f:
        for p in pool:
            k = (p["talk"], lemma(p["t"]))
            if k not in kept:
                continue
            it = kept[k]
            lines = block[p["talk"]]

            def blk(repl):
                return [repl if (t, w) == (it["t"], it["w"]) else [t, w] for t, w in lines]

            rec = dict(sid=p["sid"], talk=p["talk"], wav=p["wav"], en=p["en"], zh=p["zh"], item=p["item"],
                       first=(p["sid"] == it["sid"]), t=it["t"], w=it["w"], w_wrong=it["w1"], w_irr=it["w2"],
                       blocks=dict(correct=blk([it["t"], it["w"]]),
                                   irrelevant=blk([it["t2"], it["w2"]]),
                                   wrong=blk([it["t"], it["w1"]])))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    strata = collections.Counter("latin" if is_latin(v["w"]) else "han" for v in kept.values())
    summ = dict(seed=a.seed, pool_sentences=len(pool), items_total=len(items), items_kept=len(kept),
                items_dropped=len(dropped), strata=strata,
                pool_sentences_kept=sum(len(v["sids"]) for v in kept.values()),
                distinct_w_wrong=len({v["w1"] for v in kept.values()}),
                distinct_w_irr=len({v["w2"] for v in kept.values()}),
                items_per_talk=collections.Counter(v["talk"] for v in kept.values()),
                block_lines_per_talk={k: len(v) for k, v in block.items()})
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
