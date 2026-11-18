"""Per-system insertion counts over kept items, and the shared constants of the bootstrap.

An item's indicator is read on its first pool sentence (`first`). For each system and item, the outputs are the
three `none` decodes and one decode each of `correct`, `irrelevant` and `wrong`. An item is kept when some `none`
output has Han characters and no `none` output contains w'. H = (kept items whose `wrong` output contains w')
- (kept items whose `irrelevant` output contains w'), after removing hits hand-coded as false (`hand_false`).
"""
from __future__ import annotations

import collections
import math

from .coding import contains, echoed, has_han, is_latin, translation_part

W6 = [-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)]
B = 9999
K_FRAC, CEIL, EQ_MARGIN, CAS_FLOOR = 0.10, 0.85, 0.15, 40


class Fixed(str):
    """A translation that is already echo-cut: no further cut is applied to it. `orig` keeps the model's output,
    which echo counts read."""
    orig: str = ""


def tpart(x: str) -> str:
    return x if isinstance(x, Fixed) else translation_part(x)


def orig(x: str) -> str:
    return x.orig if isinstance(x, Fixed) else x


def han(out, sid, arm, rep=0) -> bool:
    return has_han(tpart(out[(sid, arm, rep)]))


def system_counts(name, out, items, hand_false):
    """Per-system counts over kept items. `items`: list of first-sentence records."""
    def has(sid, arm, rep, form, kind):
        h = contains(tpart(out[(sid, arm, rep)]), form)
        if h and kind == "wrong_form" and (name, sid, arm) in hand_false:
            return False
        return h

    untranslated = [it for it in items if not any(has_han(tpart(out[(it["sid"], "none", r)]))
                                                   for r in range(3))]
    ut = {it["item"] for it in untranslated}
    kept = [it for it in items if it["item"] not in ut
            and not any(has(it["sid"], "none", r, it["w_wrong"], "none") for r in range(3))]
    A = {arm: sum(has(it["sid"], arm, 0, it["w_wrong"], "wrong_form") for it in kept)
         for arm in ("correct", "irrelevant", "wrong")}
    H = A["wrong"] - A["irrelevant"]
    V = sum(len({has(it["sid"], "none", r, it["w"], "w") for r in range(3)}) > 1 for it in kept)
    Mset = [it for it in kept if not is_latin(it["w"])
            and not any(has(it["sid"], "none", r, it["w"], "w") for r in range(3))]
    Cp = (sum(has(it["sid"], "correct", 0, it["w"], "w") for it in Mset)
          - sum(has(it["sid"], "irrelevant", 0, it["w"], "w") for it in Mset))
    reads = Cp > max(V, 2) or H > max(V, 2)
    loss = sum(has(it["sid"], "correct", 0, it["w"], "w") and not has(it["sid"], "wrong", 0, it["w"], "w")
               for it in kept)
    codes = collections.Counter()
    for it in kept:
        a, b = has(it["sid"], "wrong", 0, it["w_wrong"], "wrong_form"), has(it["sid"], "wrong", 0, it["w"], "w")
        codes[{(1, 0): "adopted", (1, 1): "both", (0, 1): "correct", (0, 0): "other"}[(int(a), int(b))]] += 1
    strata = {s: dict(N=sum(1 for it in kept if (is_latin(it["w"]) == (s == "latin"))),
                      A_wrong=sum(has(it["sid"], "wrong", 0, it["w_wrong"], "wrong_form") for it in kept
                                  if is_latin(it["w"]) == (s == "latin")),
                      A_irrelevant=sum(has(it["sid"], "irrelevant", 0, it["w_wrong"], "wrong_form") for it in kept
                                       if is_latin(it["w"]) == (s == "latin")))
              for s in ("latin", "han")}
    echo = {arm: sum(echoed(orig(out[(it["sid"], arm, 0)])) for it in items) for arm in ("none", "correct", "irrelevant", "wrong")}
    no_han_kept = {arm: sum(not han(out, it["sid"], arm) for it in kept) for arm in ("correct", "irrelevant", "wrong")}
    return dict(N=len(kept), excluded=len(items) - len(kept), excluded_untranslated=len(ut), echo_outputs=echo,
                kept_without_han=no_han_kept, A=A, H=H, V=V, M=len(Mset), C_prime=Cp,
                C_prime_over_M=(Cp / len(Mset) if Mset else None), reads=reads, correct_term_loss=loss,
                wrong_arm_codes=dict(codes), strata=strata), {it["item"] for it in kept}
