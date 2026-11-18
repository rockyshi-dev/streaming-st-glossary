"""Tests of the local-agreement policy and prefix grids, without a model. Run: python3 tests/test_stream.py

Token ids are simulated by characters: an id list is a list of one-character strings, detokenized by joining.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stglossary.align import t_end  # noqa: E402
from stglossary.hits import cut as echo_cut  # noqa: E402
from stglossary.stream import Stream, cut_len, la_commit, lcp, speech_ends, strip_stop, text_steps  # noqa: E402

detok = "".join
nocut = lambda x: x  # noqa: E731


def ids(s: str) -> list:
    return list(s)


def run(ends, conts, cut_text=nocut, sources=None) -> Stream:
    """Feed scripted continuations; each continuation is what the model adds after the forced ids."""
    st = Stream(ends, detok, cut_text, sources)
    for c in conts:
        f = st.forced()
        st.feed(ids(c(detok(f)) if callable(c) else c))
    assert st.done
    return st


def test_lcp():
    assert lcp([1, 2, 3], [1, 2, 4]) == [1, 2]
    assert lcp([1, 2], [1, 2, 3]) == [1, 2]
    assert lcp([], [1]) == []
    assert lcp("我们的模型", "我们的方法") == "我们的"


def test_la_commit_needs_committed_prefix():
    assert la_commit(ids("我们"), ids("我们的模型"), ids("我们的方法")) == ids("我们的")
    try:
        la_commit(ids("我们"), ids("你们的"), ids("我们的"))
    except AssertionError:
        return
    raise AssertionError("a hypothesis not starting with the committed ids was accepted")


def test_stream_chinese_never_revises_and_flushes():
    # model outputs per prefix as full strings; the continuation is the part after what was forced
    fulls = ["我们", "我们提出", "我们提出一种", "我们提出一个新方法", "我们提出一种新的方法。"]
    st = Stream([1.0, 2.0, 3.0, 4.0, 4.6], detok, nocut)
    for full in fulls:
        f = detok(st.forced())
        # a model forced with f can only continue it: emulate by keeping f and appending the rest of `full`
        cont = full[len(f):] if full.startswith(f) else full[len(lcp(f, full)):]
        st.feed(ids(cont))
    committed = [s["committed"] for s in st.steps]
    assert committed[0] == ""  # step 1 commits nothing
    assert committed[1] == "我们"  # lcp("我们", "我们提出")
    assert committed[2] == "我们提出"
    for a, b in zip(committed, committed[1:]):
        assert b.startswith(a), (a, b)  # committed text is never revised
    assert st.steps[-1]["committed"] == st.steps[-1]["hypothesis"]  # end-of-source flush
    assert st.hyp == st.steps[-1]["hypothesis"].strip()
    # commit time: 我们 at 2.0 s, 提出 at 3.0 s, 一 at 4.0 s (lcp of 一种 and 一个), the rest at the final prefix
    assert st.first_end[:5] == [2.0, 2.0, 3.0, 3.0, 4.0] and set(st.first_end[5:]) == {4.6}
    assert len(st.first_end) == len(st.committed)


def test_single_prefix_is_offline():
    st = run([3.2], ["这是全部译文。"])
    assert st.steps[0]["n_forced"] == 0 and st.hyp == "这是全部译文。"


def test_echo_cut_before_agreement():
    block = [["parser", "解析器"]]
    cut_text = lambda x: echo_cut(x, block)  # noqa: E731
    st = run([1.0, 2.0], ["我们的解析器\nparser: 解析器", "很快。"], cut_text)
    s0 = st.steps[0]
    assert s0["cut_tokens"] == len("parser: 解析器"), s0
    assert s0["hypothesis"] == "我们的解析器\nparser: 解析器"  # recorded before the cut
    st = run([3.0], ["我们的解析器\nparser: 解析器"], cut_text)
    assert st.hyp == "我们的解析器" and "parser" not in st.hyp
    assert cut_len(ids("abc"), detok, nocut) == 3


def test_strip_stop():
    assert strip_stop([5, 6, 99, 7], [99, 100]) == [5, 6]
    assert strip_stop([5, 6], [99]) == [5, 6]


def test_speech_ends():
    assert speech_ends(16000 * 3 + 100, 16000, 1.0) == [16000, 32000, 48000, 48100]
    assert speech_ends(16000 * 2, 16000, 1.0) == [16000, 32000]  # whole file once, not twice
    assert speech_ends(8000, 16000, 1.0) == [8000]


def test_text_steps():
    en = "We use a parser here."
    words = [[" We", 0.1, 0.4], [" use", 0.4, 0.9], [" a", 0.9, 1.0], [" parser", 1.1, 2.3], [" here.", 2.4, 2.8]]
    ends, srcs = text_steps(words, en, 3.1, 1.0)
    assert ends == [1.0, 3.1] and srcs == ["We use a", en]  # 2.0 adds nothing; 3.0 is the whole sentence -> final
    ends, srcs = text_steps(words, en, 2.5, 1.0)
    assert ends == [1.0, 2.5] and srcs[-1] == en


def test_t_end():
    en = "We use a parser here."
    words = [[" We", 0.1, 0.4], [" use", 0.4, 0.9], [" a", 0.9, 1.0], [" pars", 1.1, 1.8], ["er", 1.8, 2.3],
             [" here.", 2.4, 2.8]]
    assert t_end(en, "parser", words) == 2.3
    assert t_end(en, "Parser", words) == 2.3
    assert t_end(en, "pars", words) is None  # token-boundary match
    assert t_end(en, "parser", []) is None


def test_split_character_is_held_back():
    # byte-level ids: 们 is split over two ids, as Qwen's tokenizer splits some Chinese characters
    bdet = lambda xs: b"".join(xs).decode("utf-8", errors="replace")  # noqa: E731
    wo, men = "我".encode(), "们".encode()
    a, b, c = men[:1], men[1:2], men[2:]
    st = Stream([1.0, 2.0, 3.0, 3.5], bdet, nocut)
    st.feed([wo, a])            # step 1: commits nothing
    st.feed([wo, a, b, c])      # step 2: lcp = [我, a] ends inside 们 -> only 我 committed
    assert st.steps[1]["committed"] == "我" and st.forced() == [wo]
    st.feed([a, b, c])          # step 3: lcp = 我们 complete
    assert st.steps[2]["committed"] == "我们"
    st.feed([])                 # final flush
    assert st.hyp == "我们" and all("\ufffd" not in s["committed"] for s in st.steps)
    assert len(st.first_end) == len(st.committed)



def test_sentence_end_is_held_back_until_flush():
    # agreed "你好。" is committed as "你好" so that later audio can still be translated; the flush commits the rest
    st = run([1.0, 2.0, 3.0, 4.0, 5.0], ["你好。", "你好。", lambda f: "，世界。", lambda f: "，世界。", lambda f: "。"])
    assert st.steps[1]["committed"] == "你好" and st.steps[2]["committed"] == "你好"
    assert st.steps[3]["committed"] == "你好，世界"
    assert st.hyp == "你好，世界。" and len(st.first_end) == len(st.committed)
    # an inner sentence end is committed; only the trailing one waits
    st = run([1.0, 2.0, 3.0], ["甲。乙", "甲。乙丙", lambda f: "丙。"])
    assert st.steps[1]["committed"] == "甲。乙" and st.hyp == "甲。乙丙。"

def test_quote_mark_newline_and_split_char_held_back():
    st = run([1.0, 2.0, 3.0, 4.0], ["说“好”。", "说“好”。", lambda f: "”。然后", lambda f: "”。然后走。"])
    assert st.steps[1]["committed"] == "说“好" and st.steps[2]["committed"] == "说“好"
    assert st.hyp == "说“好”。然后走。"
    st = run([1.0, 2.0, 3.0], ["好。\n", "好。\n", lambda f: "。\n再见"])
    assert st.steps[1]["committed"] == "好" and st.hyp == "好。\n再见"
    # byte-level tokens: X is the head of 们, "Y。" its tail plus a full stop; alone each decodes to U+FFFD.
    # Dropping "Y。" alone would leave "我\ufffd" committed; the one loop drops X too.
    dt = lambda x: "".join(x).replace("XY", "们").replace("X", "\ufffd").replace("Y", "\ufffd")  # noqa: E731
    st = Stream([1.0, 2.0, 3.0], dt, nocut)
    for c in (["我", "X", "Y。"], ["我", "X", "Y。"], ["X", "Y。"]):
        st.feed(c)
    assert st.steps[1]["committed"] == "我" and st.hyp == "我们。"


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            n += 1
    print(f"{n} tests passed")
