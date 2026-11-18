"""Tests of the coding file and the counts on synthetic rows, without a model. Run: python3 tests/test_contrast.py

Token ids are simulated by characters (a streaming row's ids are the characters of its text, detokenized by joining),
except in the split-character test, which uses UTF-8 bytes as Qwen's byte-level tokens do.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stglossary import coding, prompts  # noqa: E402
from stglossary import contrast as A  # noqa: E402

pb = A.pb
SHA = "a" * 40


# ---------------------------------------------------------------- units

def test_spans_agree_with_contains():
    rng = random.Random(0)
    alpha = "ab AB模型型1-"
    for _ in range(3000):
        text = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 12)))
        form = "".join(rng.choice(alpha) for _ in range(rng.randint(0, 3)))
        sp = coding.spans(text, form)
        assert bool(sp) == coding.contains(text, form), (text, form)
        assert all(text[s:e].lower() == form.lower() for s, e in sp)
    assert coding.spans("模型和模型", "模型") == [(0, 2), (3, 5)]
    assert coding.spans("use BERT and bert", "BERT") == [(4, 8)]  # all-capital form: case-sensitive, as contains


def test_normalise_and_window():
    assert pb.normalise("甲：乙\n\n丙  丁") == "甲:乙 ⏎ 丙 丁"
    text = "一" * 30 + "错词" + "二" * 30
    w = pb.window(text, "错词")
    assert w == "…" + "一" * 20 + "错词" + "二" * 20 + "…"
    # a glossary copy whose arrow lies beyond 20 characters on the same line is kept
    copy = "很长的源术语名字 machine translation system → " + "三" * 25 + "错词。"
    w = pb.window(copy, "错词")
    assert "→" in w and "错词" in w, w
    # ... but not across a line break
    w = pb.window("t → x\n" + "四" * 25 + "错词", "错词")
    assert "→" not in w, w
    # full-width colon is kept (as ':') and two occurrences far apart give two pieces
    w = pb.window("错词" + "五" * 60 + "错词", "错词")
    assert w.count("错词") == 2 and " … " in w
    assert pb.window("没有", "错词") == "没有"


def test_scored_strips_one_fffd_on_cap_only():
    f = "�"
    assert pb.scored(dict(hyp="译文" + f, hit_cap=True)) == "译文"
    assert pb.scored(dict(hyp="译文" + f + f, hit_cap=True)) == "译文" + f  # one only
    assert pb.scored(dict(hyp="译文" + f, hit_cap=False)) == "译文" + f  # not at the cap: untouched
    assert pb.scored(dict(hyp="译文", hit_cap=True)) == "译文"


def test_char_times_split_character_takes_last_token():
    bdet = lambda xs: b"".join(xs).decode("utf-8", errors="replace")  # noqa: E731
    men = "们".encode()
    ids = ["我".encode(), men[:1], men[1:2], men[2:], "错".encode()]
    T = [1.0, 1.0, 2.0, 3.0, 4.0]
    full = bdet(ids)
    ct = A.char_times(full, [(bdet(ids[:k]), T[k - 1]) for k in range(1, 6)])
    assert ct == [1.0, 3.0, 4.0], ct  # 们 is complete only with its third byte, committed at 3.0


def test_hyp_times_tokens_and_steps_agree():
    r = srow("k", "我们的错词。", [(1.0, "", "我们"), (2.0, "我们", "我们的"), (3.0, "我们的错词。", "我们的错词。")])
    t_steps, _ = A.hyp_times(r)
    t_tok, chk = A.hyp_times(r, "".join)
    assert t_steps == t_tok == [2.0, 2.0, 3.0, 3.0, 3.0, 3.0] and chk["agrees_with_steps"]
    # a cap row ending in half a character: the scored hyp drops it, times follow the scored text
    r = srow("k", "我们�", [(1.0, "", "我"), (2.0, "我们�", "我们�")], cap=True)
    r["hyp"] = pb.scored(r)
    assert A.hyp_times(r, "".join)[0] == [2.0, 2.0]


def test_lagging():
    assert A.lagging([1000, 2000, 3000], 3000, 3) == 1000
    assert A.lagging([1000, 2000, 3000], 3000, 6) == 1500  # LAAL: reference of 6 characters
    assert A.lagging([3000, 3000, 3000], 3000, 3) == 3000  # tau = 1: everything at the end
    assert A.lagging([], 3000, 3) is None
    assert A.lagging([1000], 3000, 0) is None


def test_latency_reports_reference_and_hypothesis_al():
    row = dict(hyp="我们好", duration=3.0)
    lat = A.latency(row, [1.0, 2.0, 3.0], "我们 很好吗")  # reference: 5 non-space characters
    assert lat["AL_no_use_ref_len"] == A.lagging([1000, 2000, 3000], 3000, 3) == 1000
    assert lat["AL"] == A.lagging([1000, 2000, 3000], 3000, 5) and lat["ref_chars"] == 5
    assert lat["LAAL"] == lat["AL"]  # max(3, 5) = 5


def test_window_only_raw_matches():
    text = "ＢＥＲＴ" + "一" * 30 + "用BERT。"
    w = pb.window(text, "BERT")
    assert w.count("BERT") == 1 and w.startswith("…"), w  # the full-width copy is not a hit H counts
    assert pb.window("ＢＥＲＴ模型", "BERT") == "BERT模型"  # no raw hit: the whole normalised text
    assert pb.window("甲\n\n  错词  乙", "错词") == "甲 ⏎ 错词 乙"


def test_decide_rows():
    spot = dict(read=10, uptake=10)
    base = dict(C_tf=30, H_st=10, V_own=1, spot=spot, n_unread=0, n_items=116, D=14, k=11.6, N=116, lo=0.01, hi=0.2)
    row = lambda **kw: A.decide(**{**base, **kw})[0]  # noqa: E731
    assert row() == "P1"
    assert row(C_tf=2, H_st=2) == "P0"
    assert row(spot=dict(read=10, uptake=7)) == "P0"
    assert row(n_unread=12) == "P0" and row(n_unread=11) == "P1"
    assert row(D=3, lo=-0.03, hi=0.08) == "P2"
    assert row(D=9, lo=0.001, hi=0.15) == "P3"
    assert row(D=12, lo=-0.01, hi=0.2) == "P3"


def test_decide_names_p0_clauses():
    base = dict(C_tf=30, H_st=10, V_own=1, spot=dict(read=10, uptake=10), n_unread=0, n_items=116, D=14, k=11.6,
                N=116, lo=0.01, hi=0.2)
    r, cl = A.decide(**{**base, "spot": dict(read=7, uptake=7)})  # all 7 read use w, but only 7 were read
    assert r == "P0" and len(cl) == 2 and "only 7 of 10" in cl[0] and "7 of 7" in cl[1], cl
    r, cl = A.decide(**{**base, "spot": dict(read=10, uptake=7)})
    assert r == "P0" and len(cl) == 1 and "spot check" in cl[0]
    r, cl = A.decide(**{**base, "n_unread": 12})
    assert cl == ["unreadable: 12 of 116 items > 10%"]
    r, cl = A.decide(**{**base, "C_tf": 2, "H_st": 2, "n_unread": 12})
    assert len(cl) == 2 and cl[0].startswith("panel test")


def test_decide_p0_uses_own_v():  # the panel test reads V from the streaming system's own kept set
    base = dict(C_tf=5, H_st=5, V_own=6, spot=dict(read=10, uptake=10), n_unread=0, n_items=116, D=14, k=11.6,
                N=116, lo=0.01, hi=0.2)
    assert A.decide(**base)[0] == "P0"  # 5 <= max(6, 2)
    assert A.decide(**{**base, "V_own": 4})[0] == "P1"


# ---------------------------------------------------------------- synthetic run

def H(i: int, base: int) -> str:
    return chr(base + 2 * i) + chr(base + 2 * i + 1)


def srow(key: str, final: str, steps: list[tuple[float, str, str]], cap: bool = False) -> dict:
    """A streaming row in `stream`'s format; ids are characters. steps: (end, committed after, hypothesis)."""
    out, prev = [], ""
    for end, com, hyp in steps:
        out.append(dict(end=end, n_forced=len(prev), forced_ids=list(prev), cont_ids=list(hyp[len(prev):]),
                        cont=hyp[len(prev):], hypothesis=hyp, cut_tokens=0, committed=com, n_committed=len(com)))
        prev = com
    tce = A.char_times(final, [(s["committed"], s["end"]) for s in out])
    return dict(key=key, steps=out, token_commit_end=tce, hyp=final.strip(), hit_cap=cap, duration=steps[-1][0])


def prompt(rec: dict, arm: str) -> str:
    return ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
            + prompts.text_after_source(rec, arm, "speech", "instructed") + "<|im_end|>\n<|im_start|>assistant\n")


def build(tmp: Path, over: dict | None = None, text: dict | None = None) -> dict:
    """20 decision items in 4 talks (plus one non-decision sentence that is not decoded).
    offline: w' in `wrong` for items 0, 1.  streaming: w' in `wrong` for items 0-7 and in `irrelevant` for item 7;
    item 2's w' is committed at 2.0 s (t_end 2.5: early); item 4 has no t_end; item 3's hit will be coded false;
    item 5's irrelevant output starts with w_irr (early reference); item 8 shows w' only in an uncommitted hypothesis;
    item 19's three streaming `none` decodes disagree on w (V = 1); item 10 has cap rows ending in U+FFFD in both.
    `over`: (condition, item, arm) -> output replacing the one above (rep 0 for `none`).
    `text`: write a text-control file (streaming rows with text prompts): dict(drop=[items], excluded=[sids])."""
    over = over or {}
    recs = []
    for i in range(21):
        t, w, ww, wi = f"term{i}", H(i, 0x4E00), H(i, 0x5E00), H(i, 0x6E00)
        recs.append(dict(sid=f"s{i}", talk=f"T{i % 4}", wav="x.wav", en=f"We use term{i} here.", zh="我们在这里使用" + w + "。",
                         item=f"T{i % 4}|term{i}", first=i < 20, t=t, w=w, w_wrong=ww, w_irr=wi,
                         blocks=dict(correct=[["other", "其他"], [t, w]], irrelevant=[["other", "其他"], [f"x{i}", wi]],
                                     wrong=[["other", "其他"], [t, ww]])))
    S = tmp / "sentences.jsonl"
    S.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    items_sha = hashlib.sha256(S.read_bytes()).hexdigest()
    off, st, tx = [], [], []
    for rec in recs[:20]:
        i, w, ww, wi = int(rec["sid"][1:]), rec["w"], rec["w_wrong"], rec["w_irr"]
        for arm, rep in prompts.RUNSETS["main"]:
            k = prompts.key(rec["sid"], arm, rep)
            o = {"none": "我们在这里用它。", "correct": f"我们在这里用{w}。", "irrelevant": "我们在这里用它。",
                 "wrong": (f"我们在这里用{ww}。" if i < 2 else "我们在这里用它。")}[arm]
            s = {"none": (f"我们用{w}。" if (i == 19 and rep == 0) else "我们在这里用它。"), "correct": f"我们在这里用{w}。",
                 "irrelevant": (f"{wi}我们用它。" if i == 5 else f"我们在这里用{ww}。" if i == 7 else "我们在这里用它。"),
                 "wrong": (f"{ww}我们在用。" if i == 2 else f"我们在这里用{ww}。" if i < 8 else "我们在这里用它。")}[arm]
            if rep == 0:
                o, s = over.get(("offline", i, arm), o), over.get(("streaming", i, arm), s)
            ocap = scap = False
            if i == 10 and arm == "wrong":
                o, ocap = o + "�", True
            if i == 10 and arm == "correct":
                s, scap = s + "�", True
            mid = s[:2]
            hyp2 = s if not (i == 8 and arm == "wrong") else f"我们{ww}"
            steps = [(1.0, "", s[:1]), (2.0, mid, hyp2), (3.0, s, s)]
            r = srow(k, s, steps, cap=scap)
            prov = dict(commit=SHA, dirty=False, model="o7", family="omni25", max_new_tokens=256, items_sha256=items_sha,
                        prompt_kind="instructed", prompt=prompt(rec, arm))
            r.update(sid=rec["sid"], arm=arm, rep=rep, t_end=(None if i == 4 else 2.5), **prov)
            st.append(r)
            if text is not None and i not in text.get("drop", ()):
                tr = dict(r, prompt=f"English: {rec['en']}\n" + prompts.text_after_source(rec, arm, "text", "instructed"),
                          excluded_unaligned=list(text.get("excluded", ())))
                tx.append(tr)
            off.append(dict(key=k, sid=rec["sid"], arm=arm, rep=rep, hyp=o, hit_cap=ocap, **prov))
    O, T = tmp / "off.jsonl", tmp / "st.jsonl"
    O.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in off))
    T.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in st))
    if text is not None:
        (tmp / "txt.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tx))
    return dict(sentences=str(S), offline=str(O), streaming=str(T))


def coded_run(tmp: Path, p: dict, false=lambda kr: kr["sid"] == "s3", guess=lambda kr: "offline") -> dict:
    """todo, then a coder (`false`, `guess`: key row -> code / condition guess) and the spot check (all uptake).
    Returns count's keyword arguments."""
    code, key, cc, ck = tmp / "CODE.tsv", tmp / "KEY.tsv", tmp / "CC.tsv", tmp / "CK.tsv"
    for f in (code, key, cc, ck):
        f.unlink(missing_ok=True)
    pb.todo(argparse.Namespace(sentences=p["sentences"], offline=p["offline"], streaming=p["streaming"],
                               out=str(code), key=str(key)))
    keyrows = {r["id"]: r for r in A.eb.read(str(key), pb.KEY_COLS)}
    fill(code, pb.CODE_COLS, lambda r: r.update(false_hit="y" if false(keyrows[r["id"]]) else "n",
                                                condition_guess=guess(keyrows[r["id"]])))
    A.ctodo(argparse.Namespace(ccode=str(cc), ckey=str(ck), **p))
    fill(cc, A.E.CCOLS, lambda r: r.update(uptake="y"))
    return dict(code=str(code), key=str(key), ccode=str(cc), ckey=str(ck), tokenizer=None, text_control=None, **p)


def refuses(f, *args, **kw) -> str:
    try:
        f(*args, **kw)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"{f.__name__} accepted")


def scratch():
    return tempfile.TemporaryDirectory()


def fill(path: Path, cols: list[str], f) -> None:
    rows = A.eb.read(str(path), cols)
    for r in rows:
        f(r)
    A.eb.write(str(path), cols, rows)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = build(tmp)
        code, key, cc, ck = tmp / "CODE.tsv", tmp / "KEY.tsv", tmp / "CC.tsv", tmp / "CK.tsv"
        pb.todo(argparse.Namespace(sentences=p["sentences"], offline=p["offline"], streaming=p["streaming"],
                                   out=str(code), key=str(key)))
        keyrows = {r["id"]: r for r in A.eb.read(str(key), pb.KEY_COLS)}
        coderows = A.eb.read(str(code), pb.CODE_COLS)
        assert len(coderows) == 11  # offline wrong 0,1; streaming wrong 0-7, irrelevant 7; nothing else
        assert {r["system"] for r in keyrows.values()} == {"offline", "streaming"}
        assert all("offline" not in r["window"] and "streaming" not in r["window"] for r in coderows)
        assert all("�" not in r["window"] for r in coderows)
        # coder: item 3's streaming hit is false; every other hit real; guesses all "offline"
        fill(code, pb.CODE_COLS, lambda r: r.update(
            false_hit="y" if (keyrows[r["id"]]["sid"] == "s3") else "n", condition_guess="offline"))
        try:
            pb.todo(argparse.Namespace(sentences=p["sentences"], offline=p["offline"], streaming=p["streaming"],
                                       out=str(code), key=str(key)))
            raise AssertionError("a coded file was overwritten")
        except SystemExit:
            pass
        A.ctodo(argparse.Namespace(ccode=str(cc), ckey=str(ck), **p))
        fill(cc, A.E.CCOLS, lambda r: r.update(uptake="y"))
        kw = coded_run(tmp, p)  # the same coding, made again through the shared helper
        res = A.count(argparse.Namespace(**kw))
        c = res["contrast"]
        assert res["cap_fffd_stripped"] == {"offline": 1, "streaming": 1}, res["cap_fffd_stripped"]
        assert c["N"] == 20 and c["V"] == 1 and c["k"] == 2.0, c
        assert c["A"]["offline"] == {"wrong": 2, "irrelevant": 0}
        assert c["A"]["streaming"] == {"wrong": 7, "irrelevant": 1}  # 8 hits minus item 3 coded false
        assert c["H"] == {"offline": 2, "streaming": 6} and c["D_st"] == 4
        assert c["discordant"] == dict(up=4, down=0)
        lo, hi = c["interval_D_over_N"]
        assert lo <= 4 / 20 <= hi
        assert res["decision_row"] in ("P1", "P3") and res["decision_clauses"]
        assert res["correct_spot_check"] == dict(read=10, uptake=10)
        assert res["unreadable"]["items"] == 0
        assert res["code_pairs"]["both_conditions"] == 2 and res["code_pairs"]["identical_output"] == 2  # items 0, 1
        assert res["setting"]["family"] == "omni25" and len(res["inputs"]) == 6
        assert res["blinding"]["balanced_accuracy"] == 0.5 and not res["blinding"]["failure"]
        ec = res["early_commit"]
        # wrong arm at risk: hits 0,1,2,5,6,7 (3 coded false, 4 unaligned); early: item 2 (committed 2.0 < 2.5)
        assert ec["wrong"]["at_risk"] == 6 and ec["wrong"]["early"] == 1 and ec["wrong"]["in_output_t_unaligned"] == 1
        assert ec["wrong"]["early_items"][0]["sid"] == "s2"
        assert ec["wrong"]["any_hypothesis"] == 9  # items 0-8, uncoded, committed or not
        # string presence (the reference's rule) also keeps item 3's coded-false hit
        assert ec["wrong_string"]["at_risk"] == 7 and ec["wrong_string"]["early"] == 1
        assert ec["reference_compared_with"] == "wrong_string"
        assert ec["irrelevant"]["at_risk"] == 1 and ec["irrelevant"]["early"] == 1
        lat = res["latency"]
        assert set(lat.pop("definitions")) == {"LAAL_ms", "AL_ms", "AL_no_use_ref_len_ms"}
        assert set(lat) == {"none", "correct", "irrelevant", "wrong"} and lat["wrong"]["items"] == 20
        assert all(0 < v["AL_ms"] <= 3000 and v["LAAL_ms"] >= max(v["AL_ms"], v["AL_no_use_ref_len_ms"]) - 1e-9
                   for v in lat.values())
        # a guess for every row is required
        fill(code, pb.CODE_COLS, lambda r: r.update(condition_guess=""))
        try:
            pb.blinding(str(code), str(key))
            raise AssertionError("missing guesses accepted")
        except SystemExit:
            pass
        # perfect guesses are a blinding failure
        fill(code, pb.CODE_COLS, lambda r: r.update(condition_guess=keyrows[r["id"]]["system"]))
        b = pb.blinding(str(code), str(key))
        assert b["balanced_accuracy"] == 1.0 and b["failure"]


def test_unreadable_wrong_outputs_fire_p0():
    with scratch() as d:
        tmp = Path(d)
        over = {("streaming", i, "wrong"): "" for i in range(8, 20)}  # 12 empty
        over[("streaming", 9, "irrelevant")] = "We use it here."  # English only: unreadable
        over[("streaming", 8, "irrelevant")] = "We use it here."  # same item as an empty `wrong`: counted once
        p = build(tmp, over)
        res = A.count(argparse.Namespace(**coded_run(tmp, p)))
        u = res["unreadable"]
        assert u["items"] == 12 and u["of"] == 20 and u["per_arm"]["wrong"]["items"] == 12
        assert u["per_arm"]["irrelevant"]["sids"] == ["s8", "s9"]
        assert res["decision_row"] == "P0" and any(c.startswith("unreadable: 12 of 20") for c in res["decision_clauses"])
        # two unreadable items (10%) do not fire the clause
        p = build(tmp, {("streaming", i, "wrong"): "" for i in (8, 9)})
        res = A.count(argparse.Namespace(**coded_run(tmp, p)))
        assert res["unreadable"]["items"] == 2 and not any("unreadable" in c for c in res["decision_clauses"])


def test_identical_outputs_coded_differently_refused():
    with scratch() as d:
        tmp = Path(d)
        p = build(tmp)  # items 0 and 1: identical `wrong` outputs in both conditions
        kw = coded_run(tmp, p, false=lambda kr: kr["sid"] == "s3" or (kr["sid"] == "s0" and kr["system"] == "streaming"))
        msg = refuses(A.count, argparse.Namespace(**kw))
        assert "byte-identical" in msg and "'s0'" in msg, msg
        # different outputs whose windows are identical, coded differently: accepted and reported
        tail = H(1, 0x5E00) + "。"
        p = build(tmp, {("offline", 1, "wrong"): "庚" * 5 + "甲" * 25 + tail, ("streaming", 1, "wrong"): "戊" * 5 + "甲" * 25 + tail})
        kw = coded_run(tmp, p, false=lambda kr: kr["sid"] == "s3" or (kr["sid"] == "s1" and kr["system"] == "offline"))
        res = A.count(argparse.Namespace(**kw))
        bad = res["code_pairs"]["identical_window_different_code"]
        assert [(b["sid"], b["arm"]) for b in bad] == [("s1", "wrong")], bad


def test_blinding_failure_gives_no_row():
    with scratch() as d:
        tmp = Path(d)
        p = build(tmp)
        res = A.count(argparse.Namespace(**coded_run(tmp, p, guess=lambda kr: kr["system"])))
        assert res["blinding"]["failure"] and res["decision_row"] == "none: blinding failure, recode"
        assert "the counts alone would give" in res["decision_clauses"][0]
        assert res["contrast"]["N"] == 20  # counts still written


def test_settings_and_required_inputs():
    with scratch() as d:
        tmp = Path(d)
        p = build(tmp)
        kw = coded_run(tmp, p)
        rows = [json.loads(l) for l in open(p["offline"])]
        rows[0]["family"] = "omni3"
        Path(p["offline"]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        assert "differ in setting" in refuses(A.count, argparse.Namespace(**kw))
        for r in rows:
            r["family"] = "omni25"
        Path(p["offline"]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        assert A.count(argparse.Namespace(**kw))["setting"]["family"] == "omni25"
        other = tmp / "other.jsonl"
        other.write_text("".join(json.dumps(json.loads(l)) + "\n" for l in open(p["sentences"])))  # same records, ASCII-escaped
        assert "item table" in refuses(A.count, argparse.Namespace(**{**kw, "sentences": str(other)}))


def test_text_control_complete_and_provenance():
    with scratch() as d:
        tmp = Path(d)
        p = build(tmp, text=dict(drop=[0], excluded=[]))
        kw = coded_run(tmp, p)
        kw["text_control"] = str(tmp / "txt.jsonl")
        assert "6 missing" in refuses(A.count, argparse.Namespace(**kw))
        build(tmp, text=dict(drop=[0], excluded=["s0"]))
        tc = A.count(argparse.Namespace(**kw))["text_control"]
        assert tc["items"] == 19 and tc["excluded_unaligned"] == ["s0"]
        build(tmp, text=dict(drop=[], excluded=["s0"]))  # a listed item that was decoded anyway
        assert "6 unexpected" in refuses(A.count, argparse.Namespace(**kw))
        build(tmp, text=dict(drop=[0, 1, 2], excluded=["s0", "s1", "s2"]))  # 15% unaligned
        assert "> 10%" in refuses(A.count, argparse.Namespace(**kw))
        build(tmp, text=dict(drop=[0], excluded=["s0"]))
        rows = [json.loads(l) for l in open(kw["text_control"])]
        rows[3]["dirty"] = True
        Path(kw["text_control"]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        assert "clean commit" in refuses(A.count, argparse.Namespace(**kw))
        for r in rows:
            r["dirty"], r["commit"] = False, "b" * 40
        Path(kw["text_control"]).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        assert "commits" in refuses(A.count, argparse.Namespace(**kw))


def test_streaming_prompt_must_equal_offline():
    off = {"k": dict(prompt="P", prompt_kind="instructed")}
    assert A.same_prompts(off, {"k": dict(prompt="P", prompt_kind="instructed")}) == 1
    try:
        A.same_prompts(off, {"k": dict(prompt="Q", prompt_kind="instructed")})
        raise AssertionError("a different prompt was accepted")
    except SystemExit:
        pass


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} tests passed")
