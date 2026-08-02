"""The leak rule: did the tutor give the answer away?

A ``guards.Veto`` rule, and the reason that wrapper exists. It is deliberately
RULE-BASED rather than a learned judge, because the policy cannot negotiate with
a rule: this is the one term in the reward that has to be non-hackable, since
everything else pays the tutor for being helpful and the shortest helpful thing
is the answer.

Four signals are computed. Only two fire the penalty.

    verbatim     "the answer is copper wire" - gold stated outright        ON
    identifying  names a content word or number belonging to gold and to
                 no distractor and not to the question stem                ON
    overlap      fraction of gold's content words the tutor said           off
    elimination  names the WRONG options so the student can rule them out  off

The two that are off were measured against 1,104 rated turns and are net
negative. ``_content`` drops tokens of two characters or fewer, so a gold of
"8 hours" reduces to {hour} and so does every distractor: on 437 maths items
``overlap`` degenerates into "did the tutor say the unit" and ``elimination``
into "did the tutor say the unit three times". ``elimination`` scores AUC 0.493
- chance - with precision 0.118 against a base rate of 0.208, so it fires
slightly MORE often on turns that do not leak. Dropping both is worth +0.066 F1
held out, and five-fold CV selected this exact rule in 5 of 5 folds. It takes
the flag rate from 17.9% to 10.9% of turns against a true give-away rate of
6.4%, and more than doubles precision on maths (0.191 to 0.476).

That matters more than a detector metric usually would, because the penalty
REPLACES the reward rather than adjusting it. A false positive does not add
noise; it discards everything else measured about that turn.

WHY ``identifying`` AND NOT JUST ``overlap``. ``overlap`` divides by the length
of gold, so it weakens as answers get longer - backwards for a corpus whose
answers are long by design. Measured case: gold "For depriving us ... of the
benefits of Trial by Jury", hint "...couldn't have a fair trial ... anything
about trials and juries?" scores overlap 0.167 and is not flagged. Shorten gold
to "Trial by Jury" and the SAME hint scores 0.500. Three times the signal for an
identical leak. ``identifying`` normalises by the words unique to gold instead.

WHAT IT DELIBERATELY DOES NOT FLAG: a conceptual hint that points at the
mechanism without naming the option ("think about what melts snow" for `heat
lamp`). That is arguably good teaching, and no string rule can adjudicate it -
the specificity contrast is the tool for that grey zone.

STILL OPEN. ``_content`` drops tokens of two characters or fewer, so numeric
golds keep collapsing to their units; and 46 of 77 known misses share no content
word with gold at all, which no string rule reaches.
"""

from __future__ import annotations

import re

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "is",
    "are",
    "was",
    "were",
    "be",
    "and",
    "or",
    "it",
    "its",
    "that",
    "this",
    "these",
    "those",
    "for",
    "on",
    "at",
    "as",
    "with",
    "by",
    "from",
    "than",
    "then",
    "so",
    "if",
    "not",
    "no",
    "do",
    "does",
    "you",
    "your",
    "they",
    "their",
    "what",
    "which",
    "when",
    "how",
    "why",
    "can",
    "will",
    "would",
    "about",
    "into",
    "more",
    "most",
    "some",
    "any",
    "all",
}

# Too common to identify any particular option. Consulted ONLY by the
# identifying-word rule, where a single hit costs -1, so a generic word slipping
# through is an unearned penalty. Deliberately excludes words that can carry
# answers in science items (metal, rock, high, long, ...).
_GENERIC_RAW = {
    "make",
    "made",
    "making",
    "sure",
    "being",
    "been",
    "have",
    "has",
    "had",
    "get",
    "got",
    "take",
    "taken",
    "give",
    "given",
    "come",
    "came",
    "goes",
    "went",
    "keep",
    "put",
    "use",
    "used",
    "using",
    "need",
    "needs",
    "want",
    "know",
    "known",
    "think",
    "thought",
    "look",
    "looks",
    "find",
    "found",
    "help",
    "helps",
    "thing",
    "things",
    "stuff",
    "kind",
    "sort",
    "type",
    "good",
    "bad",
    "nice",
    "great",
    "well",
    "better",
    "best",
    "just",
    "very",
    "really",
    "also",
    "like",
    "likes",
    "other",
    "others",
    "another",
    "same",
    "different",
    "way",
    "ways",
    "one",
    "ones",
    "two",
    "lot",
    "lots",
    "much",
    "many",
    "few",
    "every",
    "each",
    "both",
    "only",
    "even",
    "still",
    "always",
    "never",
    "often",
    "sometimes",
    "usually",
    "maybe",
    "perhaps",
    "should",
    "could",
    "must",
    "might",
    "may",
    "avoid",
    "avoids",
    "become",
    "becomes",
    "there",
    "here",
    "them",
    "him",
    "her",
    "his",
    "she",
    "our",
    "who",
    "whom",
    "something",
    "someone",
    "anything",
    "everything",
    "nothing",
    "person",
    "people",
    "yes",
    # prepositions and relational words long enough to survive the length filter
    "through",
    "across",
    "around",
    "between",
    "within",
    "without",
    "above",
    "below",
    "along",
    "toward",
    "towards",
    "during",
    "before",
    "after",
    # generic nouns/verbs that describe any answer rather than identify one
    "change",
    "changed",
    "changes",
    "create",
    "created",
    "creates",
    "item",
    "items",
    "form",
    "forms",
    "amount",
    "group",
    "groups",
    "place",
    "places",
    "time",
    "times",
    "area",
    "areas",
    "number",
    "part",
    "parts",
    "side",
    "sides",
    "late",
    "early",
    "happen",
    "happens",
    "occur",
    "occurs",
}

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).strip()


def _stem(w: str) -> str:
    """Crude suffix stripping - enough to match weigh/weight, melt/melting."""
    for suffix in ("ing", "edly", "ed", "es", "s", "ly"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w


def _content(s: str) -> set[str]:
    return {_stem(w) for w in _norm(s).split() if w not in _STOP and len(w) > 2}


_GENERIC = {_stem(w) for w in _GENERIC_RAW}


def _matches(gold_word: str, text_words: set[str], min_prefix: int = 4) -> bool:
    """Exact, or one is a prefix of the other at four characters or more.

    Suffix-stripping alone misses pairs like weigh/weight; prefix matching
    catches them without pulling in unrelated words.
    """
    if gold_word in text_words:
        return True
    return any(
        (w.startswith(gold_word) or gold_word.startswith(w)) and min(len(w), len(gold_word)) >= min_prefix
        for w in text_words
    )


def _numbers(s: str) -> set[str]:
    """Canonical numeric values; 7,500 and 7500.0 collapse to one."""
    out: set[str] = set()
    for match in _NUM_RE.findall(str(s)):
        try:
            out.add(f"{float(match.replace(',', '')):g}")
        except ValueError:
            continue
    return out


def identifying_words(gold: str, distractors=(), question: str = "") -> set[str]:
    """Content words that point at THIS option and nothing else.

    A word in gold that appears in no distractor and not in the question stem is
    what lets a student pick the option out. "Jury" in a Bill of Rights item is
    identifying; "many" is not, and neither is anything the question already
    said.
    """
    gold_words = _content(gold)
    if not gold_words:
        return set()
    elsewhere: set[str] = set()
    for d in distractors:
        elsewhere |= _content(d)
    elsewhere |= _content(question)
    return {w for w in gold_words - elsewhere if len(w) >= 4 and w not in _GENERIC}


def identifying_numbers(gold: str, distractors=(), question: str = "") -> set[str]:
    """Numbers in gold and nowhere else in the item.

    Half the corpus is maths, where the answer IS a number and carries no
    content words at all - 170 of 495 items have an empty identifying-word set.
    Excluding the stem's numbers matters: quoting the diameter back to the
    student is teaching, quoting the circumference is the answer.
    """
    gold_numbers = _numbers(gold)
    if not gold_numbers:
        return set()
    elsewhere: set[str] = set()
    for d in distractors:
        elsewhere |= _numbers(d)
    elsewhere |= _numbers(question)
    return gold_numbers - elsewhere


def leak_signals(tutor_text: str, gold: str, distractors=(), question: str = "") -> dict[str, float]:
    """All four signals, whether or not they fire. Log them all; act on two."""
    text_norm = _norm(tutor_text)
    text_words = _content(tutor_text)

    gold_norm = _norm(gold)
    verbatim = 1.0 if gold_norm and gold_norm in text_norm else 0.0

    gold_words = _content(gold)
    overlap = sum(_matches(g, text_words) for g in gold_words) / len(gold_words) if gold_words else 0.0

    words = identifying_words(gold, distractors, question)
    hits = sum(_matches(w, text_words) for w in words)
    numbers = identifying_numbers(gold, distractors, question)
    if numbers:
        hits += len(numbers & _numbers(tutor_text))
    total = len(words) + len(numbers)

    named = 0
    for d in distractors:
        d_words, d_norm = _content(d), _norm(d)
        if (d_norm and d_norm in text_norm) or (d_words and all(_matches(w, text_words) for w in d_words)):
            named += 1

    return {
        "verbatim": verbatim,
        "overlap": overlap,
        "elimination": named / len(distractors) if distractors else 0.0,
        "identifying": hits / total if total else 0.0,
        "identifying_hits": float(hits),
        "identifying_n": float(total),
    }


def leaked(
    tutor_text: str,
    gold: str,
    distractors=(),
    question: str = "",
    ident_hits: int = 1,
    use_overlap: bool = False,
    use_elimination: bool = False,
    overlap_threshold: float = 0.6,
    elimination_threshold: float = 0.5,
) -> bool:
    """The decision. Any enabled signal firing counts as a leak.

    ``ident_hits`` is an absolute count, not a fraction, on purpose: a fraction
    would reintroduce the very length dilution ``identifying`` exists to remove.
    Naming one word that belongs to gold alone is a leak whether gold has one
    such word or six.

    Pass ``use_overlap=True, use_elimination=True`` to reproduce the
    pre-calibration rule - note that leak rates measured under the two rules are
    NOT comparable, since a narrower rule flags fewer turns whatever the policy
    does.
    """
    s = leak_signals(tutor_text, gold, distractors, question)
    return bool(
        s["verbatim"] >= 1.0
        or s["identifying_hits"] >= ident_hits
        or (use_overlap and s["overlap"] >= overlap_threshold)
        or (use_elimination and s["elimination"] >= elimination_threshold)
    )


def gold_text(item: dict) -> str:
    return item["choices"][item["gold_idx"]]


def distractors_of(item: dict) -> list[str]:
    return [c for i, c in enumerate(item["choices"]) if i != item["gold_idx"]]


def leaked_item(tutor_text: str, item: dict, **kwargs) -> bool:
    """``leaked`` against a problem dict."""
    return leaked(tutor_text, gold_text(item), distractors_of(item), question=item.get("question", ""), **kwargs)
