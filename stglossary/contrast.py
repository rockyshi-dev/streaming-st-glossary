"""Counts of the streaming study: offline vs streaming decodes of one speech LLM on the decision sentences.

Inputs: the offline decode (`decode_hf` rows) and the streaming decode (`stream decode` rows: key, sid, arm, rep, hyp,
hyp_uncut, steps[], token_commit_end, t_end, duration, prompt) of Qwen2.5-Omni-7B on the 116 decision sentences, four
arms, `none` three times, prompt `instructed`.

  python -m stglossary.contrast ctodo --sentences S --offline OFF.jsonl --streaming ST.jsonl --ccode CCODE.tsv --ckey CKEY.tsv
      the spot check: 10 streaming `correct` outputs containing w on M items (`panel.correct_sample`)
  python -m stglossary.contrast count --sentences S --offline OFF.jsonl --streaming ST.jsonl --code CODE.tsv --key KEY.tsv
         --ccode CCODE.tsv --ckey CKEY.tsv --out R.json [--tokenizer MODEL_DIR] [--text-control TXT.jsonl]
      every count and the decision row P0-P3 with the clauses that fired. Refuses to run while a w' hit or a
      correct-check row is uncoded or unguessed, while a row comes from a dirty tree, while the two files differ in
      model, family, max_new_tokens or item table (or the item table is not --sentences), while a streaming prompt
      differs from the offline prompt of the same key, or while identical scored offline and streaming outputs carry
      different codes. A blinding failure (balanced accuracy > 0.75) gives no decision row. CODE/KEY come from
      `sheets todo`.

Decision rows (first match): P0 the measurement cannot see insertion (low counts, failed spot check, or more than 10%
of items unreadable); P1 streaming raises insertion: D_st >= k and the interval's lower bound > 0; P2 a rise of k or
more is excluded: the interval's upper bound < k/N; P3 otherwise. k = max(0.10 N, V).

What is reused: the exclusions, H, V and has() (`insertion.system_counts` via `panel.counts`, which adds C_tf), the
pair-set exclusion (`panel.pair_items`), the talk-clustered wild bootstrap (`panel.Boot`, B = 9,999, Webb weights),
the echo cut and hit list (`hits`), the correct-arm sampler and reader (`panel.correct_sample`, `write_ctodo`,
`correct_check`).

Character commit times (early commit and latency): a character of the final output is committed when the last token
it spans is committed; a token's commit time is the end of the prefix whose step committed it (`token_commit_end`).
The final committed ids are decoded cumulatively (ids[:1], ids[:2], ...) and a character takes the time of the first
prefix of ids whose text agrees with the final text through that character (a character split over byte-level tokens
is complete only with its last token). `--tokenizer` gives the decoder; without it the steps' committed texts are
used the same way (`char_times`), and with it both are computed and their disagreements counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from . import panel as E
from . import prompts
from . import sheets as pb
from .coding import contains, has_han, spans

e1 = E.e1
eb = E.blind

MODEL = "o7"
LAB = {"offline": f"e2e:{MODEL}", "streaming": f"e2e:{MODEL}+la2", "text": f"e2e-txt:{MODEL}+la2"}
KIND = "instructed"
BA_MAX = pb.BA_MAX
SETTING = ("model", "family", "max_new_tokens", "items_sha256")  # one value across the decodes


# ---------------------------------------------------------------- loading and premises

def rows_of(path: str) -> dict:
    return {r["key"]: r for r in map(json.loads, open(path)) if not r.get("extra")}


def provenance(raw: dict) -> dict:
    """Every row from a clean tree; per condition its commits, models and decodes at the token cap. A row may carry
    no commit (the published data files have it removed); a commit that is present must be a full hash."""
    out = {}
    for c, rows in raw.items():
        bad = [k for k, r in rows.items() if r.get("dirty") is not False
               or ("commit" in r and not re.fullmatch(r"[0-9a-f]{40}", str(r.get("commit"))))]
        if bad:
            raise SystemExit(f"{c}: {len(bad)} rows lack a clean commit, e.g. {bad[:3]}")
        out[c] = dict(commits=sorted({r["commit"] for r in rows.values() if "commit" in r}), models=sorted({str(r.get("model")) for r in rows.values()}),
                      hit_cap=sum(bool(r.get("hit_cap")) for r in rows.values()), rows=len(rows))
    return out


def same_setting(raw: dict, sentences_sha: str) -> dict:
    """Every row of every file names one model, family, max_new_tokens and item table, and the
    item table is the --sentences file."""
    vals = {f: sorted({str(r.get(f)) for rows in raw.values() for r in rows.values()}) for f in SETTING}
    bad = {f: v for f, v in vals.items() if len(v) != 1}
    if bad:
        raise SystemExit(f"{sorted(raw)}: decodes differ in setting {bad}")
    if vals["items_sha256"][0] != sentences_sha:
        raise SystemExit(f"decodes used item table {vals['items_sha256'][0]}, --sentences is {sentences_sha}")
    return {f: v[0] for f, v in vals.items()}


def same_prompts(off: dict, st: dict) -> int:
    """The streaming row stores the offline prompt it decoded every step with; it must equal the offline
    re-decode's prompt of the same key byte for byte."""
    for k, r in st.items():
        if k not in off:
            raise SystemExit(f"{k}: streaming key has no offline decode")
        if r["prompt"] != off[k]["prompt"] or r.get("prompt_kind") != off[k].get("prompt_kind"):
            raise SystemExit(f"{k}: streaming and offline prompts differ")
    return len(st)


def load(path: str, recs: dict, label: str, raw: dict) -> dict:
    """The panel loader (prompt checks, echo cut), then the cap strip (`sheets.scored`) on every row: the stored
    translation and its `.orig` are re-made from the scored hyp, and raw['hyp'] becomes it (raw['hyp_stored'] keeps
    the original). One function for both conditions."""
    out = E.load(Path(path), recs, label, KIND)
    for r in raw.values():
        h = pb.scored(r)
        if h != r["hyp"]:
            blk = None if r["arm"] == "none" else eb.blocks(recs[r["sid"]])[r["arm"]]
            f = e1.Fixed(eb.cut(h, blk))
            f.orig = h
            out[(r["sid"], r["arm"], r["rep"])] = f
            r["hyp_stored"], r["hyp"] = r["hyp"], h
    return out


def complete(out: dict, items: list[dict], label: str) -> None:
    want = {(it["sid"], arm, rep) for it in items for arm, rep in prompts.RUNSETS["main"]}
    if not want <= set(out):
        raise SystemExit(f"{label}: {len(want - set(out))} decision-sentence decodes missing")


# ---------------------------------------------------------------- character commit times

def lcp_len(a: str, b: str) -> int:
    n, m = 0, min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


def char_times(full: str, stages: list[tuple[str, float]]) -> list[float]:
    """Per character of `full`, the time of the first stage whose text agrees with `full` through that character.
    `stages`: (text, time) in commit order, each text a growing prefix (a trailing U+FFFD, half a character, does not
    count). A character no stage reaches gets the last stage's time."""
    ct, j = [], 0
    for text, t in stages:
        L = lcp_len(text.rstrip("�"), full)
        while j < L:
            ct.append(t)
            j += 1
    last = stages[-1][1] if stages else None
    return ct + [last] * (len(full) - len(ct))


def final_ids(row: dict) -> list:
    """The committed ids after the last step (= forced ids + continuation, cut to n_committed; stream.Stream.feed)."""
    s = row["steps"][-1]
    ids = (list(s["forced_ids"]) + list(s["cont_ids"]))[: s["n_committed"]]
    if len(ids) != len(row["token_commit_end"]):
        raise SystemExit(f"{row['key']}: {len(ids)} committed ids but {len(row['token_commit_end'])} commit times")
    return ids


def hyp_times(row: dict, detok=None) -> tuple[list[float], dict]:
    """Commit time (s) of every character of row['hyp'] (the stripped committed text). With `detok`, from the token
    commit times; the steps' committed texts give the same reading, and a disagreement is reported in `check`."""
    by_steps_full = row["steps"][-1]["committed"]
    st = char_times(by_steps_full, [(s["committed"], s["end"]) for s in row["steps"]])
    check = dict(method="steps")
    full, ct = by_steps_full, st
    if detok is not None:
        ids, T = final_ids(row), row["token_commit_end"]
        full = detok(ids)
        ct = char_times(full, [(detok(ids[:k]), T[k - 1]) for k in range(1, len(ids) + 1)])
        check = dict(method="tokens", agrees_with_steps=(full == by_steps_full and ct == st))
    if pb.scored(dict(hyp=full.strip(), hit_cap=row.get("hit_cap"))) != row["hyp"]:
        raise SystemExit(f"{row['key']}: the committed text does not give `hyp`")
    lead = len(full) - len(full.lstrip())
    return ct[lead: lead + len(row["hyp"])], check


# ---------------------------------------------------------------- latency (LAAL, AL; on target characters)

def lagging(delays: list[float], src: float, tgt_len: int) -> float | None:
    """Average lagging (Ma et al. 2019, arXiv 1810.08398) as SimulEval's AL scorer computes it for speech input, in
    the unit of `delays` and `src`: tau = first target unit whose delay reaches the source length; rate = tgt_len / src.
    tgt_len = |reference| is SimulEval's default AL (use_ref_len=True, latency_scorer.py at 536de82); tgt_len = |hyp|
    is SimulEval's AL with --no-use-ref-len; tgt_len = max(|hyp|, |reference|) is LAAL (Papi et al. 2022,
    arXiv 2206.05807)."""
    if not delays or src <= 0 or tgt_len <= 0:
        return None
    if delays[0] > src:
        return delays[0]
    gamma = tgt_len / src
    s, tau = 0.0, 0
    for i, d in enumerate(delays):
        if d > src:
            break
        s += d - i / gamma
        tau = i + 1
        if d >= src:
            break
    return s / tau


def latency(row: dict, times: list[float], ref: str) -> dict:
    """LAAL and AL in ms on the non-space characters of the final output: `AL` uses
    the reference length (SimulEval's default AL, use_ref_len=True; reference = the non-space characters of the item's
    zh); `AL_no_use_ref_len` uses the hypothesis length (SimulEval AL with --no-use-ref-len), a labelled extra.
    Source length = audio duration; a delay beyond the duration by rounding (prefix ends are stored to 0.1 ms) is
    clamped to it."""
    src = row["duration"] * 1000
    d = [min(t * 1000, src) for ch, t in zip(row["hyp"], times) if not ch.isspace()]
    n_ref = sum(not c.isspace() for c in ref)
    return dict(AL=lagging(d, src, n_ref), AL_no_use_ref_len=lagging(d, src, len(d)),
                LAAL=lagging(d, src, max(len(d), n_ref)), chars=len(d), ref_chars=n_ref)


# ---------------------------------------------------------------- early commit (anticipation) and any-hypothesis

def first_commit(text: str, times: list[float], form: str) -> float | None:
    """Commit time of the earliest occurrence of `form` in `text` (a prefix of hyp): the time of its last character."""
    sp = spans(text, form)
    return min(max(times[s:e]) for s, e in sp) if sp else None


def early(st_raw: dict, st_out: dict, items: list[dict], false: set, detok) -> dict:
    """Early commit. At risk: items whose t is aligned (t_end) and whose final output holds the form
    (w' under `wrong`, after the coded false hits are removed, as in H; the irrelevant line's own target w_irr under
    `irrelevant`, the reference). Early: the form's first committed occurrence is committed at a prefix end before
    t_end, the end of t's first occurrence in the forced alignment. Any-hypothesis: the form in the echo-cut whole
    hypothesis (committed + uncommitted) of any step.
    `wrong` drops the coded-false hits (as H); `wrong_string` keeps them, the plain string
    presence that the `irrelevant` reference uses (its hits are not coded), and is the one compared with it."""
    res = {}
    for name, arm, field in (("wrong", "wrong", "w_wrong"), ("wrong_string", "wrong", "w_wrong"),
                             ("irrelevant", "irrelevant", "w_irr")):
        at_risk, early_items, anyhyp, no_t, agree = [], [], [], 0, []
        for it in items:
            sid, form = it["sid"], it[field]
            row = st_raw[prompts.key(sid, arm, 0)]
            blk = eb.blocks(it)[arm]
            if any(contains(eb.cut(s["hypothesis"], blk), form) for s in row["steps"]):
                anyhyp.append(sid)
            tp = e1.tpart(st_out[(sid, arm, 0)])
            if not contains(tp, form) or (name == "wrong" and (LAB["streaming"], sid, arm) in false):
                continue
            if row.get("t_end") is None:
                no_t += 1
                continue
            times, chk = hyp_times(row, detok)
            if "agrees_with_steps" in chk:
                agree.append(chk["agrees_with_steps"])
            at_risk.append(sid)
            t = first_commit(tp, times, form)
            if t < row["t_end"]:
                early_items.append(dict(sid=sid, commit=t, t_end=row["t_end"]))
        res[name] = dict(form=field, coded_false_removed=(name == "wrong"), at_risk=len(at_risk), early=len(early_items), in_output_t_unaligned=no_t,
                        any_hypothesis=len(anyhyp), early_items=early_items,
                        token_vs_steps_disagree=(sum(not x for x in agree) if agree else None))
    res["reference_compared_with"] = "wrong_string"
    return res


# ---------------------------------------------------------------- decision

UNREAD_ARMS = ("wrong", "irrelevant")


def unreadable(out: dict, items: list[dict]) -> dict:
    """P0's 'more than 10% of items unreadable': an item, of all decision items, is unreadable when its streaming
    `wrong` or `irrelevant` output, rep 0 (the two outputs H reads), has no Han character after the echo cut
    (`e1.tpart`). An English-only output, Latin w' or not, is unreadable. Lists per arm and their union."""
    per = {arm: [it["sid"] for it in items if not has_han(e1.tpart(out[(it["sid"], arm, 0)]))] for arm in UNREAD_ARMS}
    union = [it["sid"] for it in items if any(it["sid"] in per[arm] for arm in UNREAD_ARMS)]
    return dict(items=len(union), of=len(items), sids=union,
                per_arm={a: dict(items=len(v), sids=v) for a, v in per.items()})


def decide(C_tf: int, H_st: int, V_own: int, spot: dict, n_unread: int, n_items: int, D: int, k: float, N: int,
           lo: float, hi: float) -> tuple[str, list[str]]:
    """The first matching decision row, and the clause(s) that made it match. P0's first clause is the offline panel's
    test on the streaming system's own kept set (C_tf, H and V all from `panel.counts`); k comes from the pair-set V.
    The spot check needs 10 rows read: fewer is P0 ('fewer than 8 of 10' cannot be judged)."""
    m = max(V_own, 2)
    p0 = []
    if C_tf <= m and H_st <= m:
        p0.append(f"panel test: C_tf(streaming) {C_tf} <= {m} and H(streaming) {H_st} <= {m} = max(V {V_own}, 2)")
    if spot["read"] < 10:
        p0.append(f"spot check: only {spot['read']} of 10 `correct` outputs read")
    if spot["uptake"] < 8:
        p0.append(f"spot check: {spot['uptake']} of {spot['read']} read use w for t (< 8)")
    if n_unread > 0.10 * n_items:
        p0.append(f"unreadable: {n_unread} of {n_items} items > 10%")
    if p0:
        return "P0", p0
    if D >= k and lo > 0:
        return "P1", [f"D_st {D} >= k {k:g} and lower bound {lo:.4f} > 0"]
    if N and hi < k / N:
        return "P2", [f"upper bound {hi:.4f} < k/N {k / N:.4f}"]
    return "P3", ["no earlier row matched"]


def contrast(outs: dict, items: list[dict], keep: dict, false: set, boot) -> dict:
    """N, H per condition on the pair set, D_st, V (streaming `none`, on the pair set), k, the interval of D_st/N."""
    off, st = LAB["offline"], LAB["streaming"]
    both, its = E.pair_items(outs, off, st, items, keep)

    def hit(lab, it, arm):
        return int(contains(e1.tpart(outs[lab][(it["sid"], arm, 0)]), it["w_wrong"]) and (lab, it["sid"], arm) not in false)

    d = [(hit(st, it, "wrong") - hit(st, it, "irrelevant")) - (hit(off, it, "wrong") - hit(off, it, "irrelevant"))
         for it in its]
    N = len(its)
    V = sum(len({contains(e1.tpart(outs[st][(it["sid"], "none", r)]), it["w"]) for r in range(3)}) > 1 for it in its)
    k = max(e1.K_FRAC * N, V)
    lo, hi = boot.interval(d, [it["talk"] for it in its])
    A = {c: {arm: sum(hit(LAB[c], it, arm) for it in its) for arm in ("wrong", "irrelevant")} for c in ("offline", "streaming")}
    return dict(N=N, excluded=len(items) - len(both), excluded_irrelevant_untranslated=len(both) - N,
                A=A, H={c: A[c]["wrong"] - A[c]["irrelevant"] for c in A}, D_st=sum(d), V=V, k=k,
                interval_D_over_N=[lo, hi], k_over_N=(k / N if N else None),
                discordant=dict(up=sum(x > 0 for x in d), down=sum(x < 0 for x in d)), _its=its)


def summarise_latency(st_raw: dict, items: list[dict], recs: dict, detok) -> dict:
    out = {}
    for arm, rep in prompts.RUNSETS["main"]:
        if rep:
            continue
        L = [latency(st_raw[prompts.key(it["sid"], arm, 0)], hyp_times(st_raw[prompts.key(it["sid"], arm, 0)], detok)[0],
                     recs[it["sid"]]["zh"]) for it in items]
        L = [x for x in L if x["AL"] is not None and x["AL_no_use_ref_len"] is not None]

        def mean(f):
            return sum(x[f] for x in L) / len(L) if L else None
        out[arm] = dict(items=len(L), LAAL_ms=mean("LAAL"), AL_ms=mean("AL"),
                        AL_no_use_ref_len_ms=mean("AL_no_use_ref_len"))
    out["definitions"] = dict(
        LAAL_ms="LAAL: target length max(|hyp|, |reference|), non-space characters (SimulEval LAAL)",
        AL_ms="AL: target length |reference| = non-space characters of zh (SimulEval default, use_ref_len=True)",
        AL_no_use_ref_len_ms="AL, --no-use-ref-len: target length |hyp| (SimulEval AL with --no-use-ref-len)")
    return out


def tokenizer(path: str | None):
    if not path:
        return None
    from transformers import AutoTokenizer  # the processor's batch_decode (decode_hf.Decoder.text) delegates here
    tok = AutoTokenizer.from_pretrained(path)
    return lambda ids: tok.batch_decode([list(ids)], skip_special_tokens=True)[0]


def count(a) -> dict:
    recs = eb.load(a.sentences)
    E.recs_by_sid.update(recs)
    items = [r for r in recs.values() if r["first"]]
    pb.decision_only(recs, [a.offline, a.streaming])
    raw = {"offline": rows_of(a.offline), "streaming": rows_of(a.streaming)}
    prov = provenance(raw)
    sentences_sha = hashlib.sha256(Path(a.sentences).read_bytes()).hexdigest()
    setting = same_setting(raw, sentences_sha)
    commits = set(prov["offline"]["commits"]) | set(prov["streaming"]["commits"])
    if len(commits) > 1:  # both conditions decoded in the same run at the same commit
        raise SystemExit(f"offline and streaming decodes come from {len(commits)} commits {sorted(commits)}")
    n_prompts = same_prompts(raw["offline"], raw["streaming"])
    outs = {LAB[c]: load(p, recs, LAB[c], raw[c]) for c, p in (("offline", a.offline), ("streaming", a.streaming))}
    cap_stripped = {c: sum("hyp_stored" in r for r in raw[c].values()) for c in raw}
    for lab, o in outs.items():
        complete(o, items, lab)
    _, false_c = pb.coded(a.code, a.key, recs, a.offline, a.streaming)
    false = {(LAB[c], sid, arm) for c, sid, arm in false_c}
    # compared on the string H reads, the echo cut of the scored hyp
    pairs = pb.code_pairs(a.code, a.key, {c: {k: str(outs[LAB[c]][(r["sid"], r["arm"], r["rep"])])
                                              for k, r in raw[c].items()} for c in raw})
    blinding = pb.blinding(a.code, a.key)

    boot = E.Boot([it["talk"] for it in items])
    res, keep, Ms = {}, {}, {}
    for lab in outs:
        res[lab], keep[lab], Ms[lab] = E.counts(lab, outs[lab], items, false)
    con = contrast(outs, items, keep, false, boot)
    its = con.pop("_its")
    cchk = E.correct_check(a.ccode, a.ckey, outs).get(LAB["streaming"], dict(read=0, uptake=0))
    unread = unreadable(outs[LAB["streaming"]], items)
    st = res[LAB["streaming"]]
    lo, hi = con["interval_D_over_N"]
    row, clauses = decide(st["C_tf"], st["H"], st["V"], cchk, unread["items"], len(items), con["D_st"], con["k"],
                          con["N"], lo, hi)
    if blinding["failure"]:  # the coding is to be redone by a second coder; no row is read from this one
        clauses = [f"blinding: balanced accuracy {blinding['balanced_accuracy']} > {BA_MAX}; "
                   f"the counts alone would give {row} ({'; '.join(clauses)})"]
        row = "none: blinding failure, recode"

    detok = tokenizer(a.tokenizer)
    result = dict(
        model=MODEL, seed=E.SEED, B=E.B,
        sentences_sha256=sentences_sha, setting=setting,
        inputs={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (a.offline, a.streaming, a.code, a.key, a.ccode, a.ckey)},
        provenance=prov, cap_fffd_stripped=cap_stripped, prompts_identical=n_prompts, false_hits=len(false), blinding=blinding,
        code_pairs=pairs, systems=res, contrast=con, correct_spot_check=dict(read=cchk["read"], uptake=cchk["uptake"]),
        unreadable=unread, decision_row=row, decision_clauses=clauses,
        P0_inputs=dict(C_tf_streaming=st["C_tf"], H_streaming=st["H"], V_streaming_own_kept=st["V"],
                       V_pair_set_for_k=con["V"]),
        early_commit=early(raw["streaming"], outs[LAB["streaming"]], its, false, detok),
        latency=summarise_latency(raw["streaming"], its, recs, detok),
        char_time_method="tokens" if detok else "steps")
    if a.text_control:  # descriptive, no decision row; its hits are not hand-coded
        result["text_control"] = text_control(a.text_control, recs, items, prov, raw, sentences_sha)
    return result


def text_control(path: str, recs: dict, items: list[dict], prov: dict, raw: dict, sentences_sha: str) -> dict:
    """The text control is read only from a clean decode at the streaming run's commit and
    setting, and its items must be exactly the decision items minus those its rows list in `excluded_unaligned`
    (`stream decode`; more than 10% unaligned drops the control)."""
    pb.decision_only(recs, [path])
    traw = rows_of(path)
    tprov = provenance({"text": traw})["text"]
    if tprov["commits"] != prov["streaming"]["commits"]:
        raise SystemExit(f"text control commits {tprov['commits']} != streaming {prov['streaming']['commits']}")
    same_setting({**raw, "text": traw}, sentences_sha)
    excl = {tuple(r.get("excluded_unaligned") or ()) for r in traw.values()}
    if len(excl) != 1:
        raise SystemExit(f"text control rows disagree on excluded_unaligned: {sorted(excl)}")
    unal = set(excl.pop())
    if len(unal) > 0.10 * len(items):
        raise SystemExit(f"text control: {len(unal)} of {len(items)} items unaligned (> 10%); not reported")
    txt = load(path, recs, LAB["text"], traw)
    tit = [it for it in items if it["sid"] not in unal]
    want = {(it["sid"], arm, rep) for it in tit for arm, rep in prompts.RUNSETS["main"]}
    if set(txt) != want:
        miss, extra = sorted(want - set(txt)), sorted(set(txt) - want)
        raise SystemExit(f"text control: decodes != decision items minus excluded_unaligned; "
                         f"{len(miss)} missing, e.g. {miss[:3]}; {len(extra)} unexpected, e.g. {extra[:3]}")
    r, _, _ = E.counts(LAB["text"], txt, tit, set())
    return dict(items=len(tit), excluded_unaligned=sorted(unal), provenance=tprov, uncoded=True,
                **{k: r[k] for k in ("N", "A", "H", "V", "C_tf")})


def ctodo(a) -> None:
    if Path(a.ccode).exists():
        raise SystemExit(f"{a.ccode} exists; move it away first")
    recs = eb.load(a.sentences)
    E.recs_by_sid.update(recs)
    items = [r for r in recs.values() if r["first"]]
    lab = LAB["streaming"]
    out = load(a.streaming, recs, lab, rows_of(a.streaming))
    complete(out, items, lab)
    _, _, M = E.counts(lab, out, items, set())  # M does not depend on the hand codes
    E.write_ctodo(a.ccode, a.ckey, E.correct_sample({lab: out}, {lab: M}, items), recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ctodo", "count"])
    for x in ("sentences", "offline", "streaming", "ccode", "ckey"):
        ap.add_argument("--" + x, required=True)
    for x in ("code", "key", "out", "tokenizer", "text-control"):
        ap.add_argument("--" + x)
    a = ap.parse_args()
    if a.cmd == "ctodo":
        ctodo(a)
        return
    if not (a.code and a.key and a.out):
        raise SystemExit("count needs --code, --key and --out")
    res = count(a)
    with open(a.out, "x") as fh:  # never overwrite a result
        fh.write(json.dumps(res, indent=1, ensure_ascii=False))
    c = res["contrast"]
    print(f"row {res['decision_row']} ({'; '.join(res['decision_clauses'])}): N={c['N']} H off={c['H']['offline']} st={c['H']['streaming']} D_st={c['D_st']} "
          f"k={c['k']:.1f} interval={c['interval_D_over_N']}; blinding BA={res['blinding']['balanced_accuracy']}")


if __name__ == "__main__":
    main()
