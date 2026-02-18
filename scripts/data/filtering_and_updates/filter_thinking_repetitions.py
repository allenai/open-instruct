#!/usr/bin/env python3
"""
Filter examples with repetitive thinking traces from post-training datasets.

Detects semantic repetition patterns in <think>...</think> traces that indicate
the model is stuck in reasoning loops, strategy cycling, or excessive verification.

Unlike filter_ngram_repetitions.py which catches exact text repetition, this filter
catches cases where the model applies the same *strategy* or *pattern* over and over
with slightly different words.

Six complementary detection strategies are used:
  1. Marker phrase density (counting known loop markers per paragraph)
  2. Top-K vocabulary n-grams (reducing vocab then finding repeated n-grams)
  3. Paragraph-start pattern repetition (low diversity in how paragraphs begin)
  4. POS tag n-grams via spaCy (falls back to lightweight word-class system if unavailable)
  5. Thinking segment action classification (detecting verify/reconsider/wait loops)
  6. Near-duplicate segment detection (content-word bigram Jaccard similarity)

By default, an example is flagged only when 2+ strategies agree (--min-strategies).

Example usage:
    python scripts/data/filtering_and_updates/filter_thinking_repetitions.py \
        --input-dataset allenai/Dolci-Think-SFT-32B \
        --column messages \
        --analyze-only --verbose --debug

    python scripts/data/filtering_and_updates/filter_thinking_repetitions.py \
        --input-dataset allenai/Dolci-Think-SFT-32B \
        --column messages \
        --push-to-hf
"""

import argparse
import re
import sys
from collections import Counter

from datasets import Dataset, Value, load_dataset

import open_instruct.utils as open_instruct_utils
from open_instruct import logger_utils

try:
    import spacy  # pyright: ignore

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


logger = logger_utils.setup_logger(__name__)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_thinking_trace(example: dict, column: str) -> str | None:
    """Extract thinking trace text from between <think> and </think> tags.

    Returns None if no thinking trace is found.
    """
    if column == "messages":
        messages = example.get("messages", [])
        assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
        if not assistant_messages:
            return None
        content = assistant_messages[-1].get("content", "")
    else:
        content = example.get(column, "")

    if not content:
        return None

    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Length-Scaled Thresholds
# ---------------------------------------------------------------------------


def length_scaled_min_repeats(base_min_repeats: int, num_words: int, reference_words: int = 5000) -> int:
    """Scale min_repeats proportionally to text length.

    For texts <= reference_words, returns base unchanged.
    For longer texts, scales linearly (expected n-gram counts scale linearly with length).
    """
    scale = max(1.0, num_words / reference_words)
    return max(base_min_repeats, int(base_min_repeats * scale))


# ---------------------------------------------------------------------------
# Strategy 1: Marker Phrase Density
# ---------------------------------------------------------------------------

MARKER_PATTERNS: dict[str, list[str]] = {
    "wait_reconsider": [r"\bwait\b", r"\bhold on\b", r"\bon second thought\b"],
    "self_directed": [
        r"\blet me think\b",
        r"\blet me try\b",
        r"\blet me check\b",
        r"\blet me verify\b",
        r"\blet me reconsider\b",
        r"\blet me re-?examine\b",
        r"\blet me re-?visit\b",
        r"\blet me re-?think\b",
        r"\blet me re-?do\b",
        r"\blet me re-?calculate\b",
        r"\blet me re-?evaluate\b",
        r"\blet me double[- ]?check\b",
        r"\blet me start over\b",
        r"\blet me go back\b",
    ],
    "hedging": [r"\balternatively\b", r"\bperhaps\b", r"\bmaybe i should\b"],
}

# Compiled patterns (built once at module load)
_COMPILED_MARKERS: dict[str, list[re.Pattern[str]]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats] for cat, pats in MARKER_PATTERNS.items()
}

# Per-category defaults: (threshold_per_paragraph, absolute_threshold)
_MARKER_THRESHOLDS: dict[str, tuple[float, int]] = {
    "wait_reconsider": (0.4, 30),
    "self_directed": (0.5, 40),
    "hedging": (0.3, 25),
}

_COMBINED_DENSITY_THRESHOLD = 0.8


def marker_phrase_density(text: str, threshold_scale: float = 1.0) -> tuple[bool, dict]:
    """Count marker phrases and flag if density is too high.

    Args:
        text: The thinking trace text.
        threshold_scale: Multiply thresholds by this value (>1 = more lenient).

    Returns:
        (flagged, details) where details has per-category counts and densities.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    num_paragraphs = max(len(paragraphs), 1)

    category_counts: dict[str, int] = {}
    for cat, patterns in _COMPILED_MARKERS.items():
        count = 0
        for pat in patterns:
            count += len(pat.findall(text))
        category_counts[cat] = count

    total_count = sum(category_counts.values())
    total_density = total_count / num_paragraphs

    details = {
        "category_counts": category_counts,
        "num_paragraphs": num_paragraphs,
        "total_count": total_count,
        "total_density": total_density,
    }

    # Check per-category thresholds
    # Require BOTH high density AND a minimum absolute count to avoid
    # false positives on short traces where one marker inflates density.
    _MIN_CATEGORY_COUNT = 5
    flagged = False
    triggered_categories = []
    for cat, count in category_counts.items():
        density = count / num_paragraphs
        thresh_density, thresh_abs = _MARKER_THRESHOLDS[cat]
        thresh_density *= threshold_scale
        thresh_abs = int(thresh_abs * threshold_scale)
        if (density > thresh_density and count >= _MIN_CATEGORY_COUNT) or count > thresh_abs:
            flagged = True
            triggered_categories.append(cat)

    # Check combined density (also require minimum absolute count)
    _MIN_COMBINED_COUNT = 10
    if total_density > _COMBINED_DENSITY_THRESHOLD * threshold_scale and total_count >= _MIN_COMBINED_COUNT:
        flagged = True
        triggered_categories.append("combined")

    details["triggered_categories"] = triggered_categories
    return flagged, details


# ---------------------------------------------------------------------------
# Strategy 2: Top-K Vocabulary N-grams
# ---------------------------------------------------------------------------

# Default configurations: (K, n, min_repeats)
# min_repeats are base values; they get length-scaled inside topk_vocab_ngram_repetition.
DEFAULT_TOPK_CONFIGS: list[tuple[int, int, int]] = [
    (10, 3, 12),  # Very coarse, short n-grams, high threshold
    (10, 4, 8),  # Original, now length-scaled
    (10, 6, 4),  # Coarse vocab, longer patterns
    (20, 4, 8),  # Raised from min_repeats=5
    (20, 5, 5),  # Medium granularity
    (20, 7, 3),  # Long patterns, medium vocab
    (50, 5, 4),  # Original, now length-scaled
    (50, 6, 3),  # Near-original vocab, 6-grams
    (50, 8, 3),  # Near-original vocab, catches repeated reasoning chains
]


def topk_vocab_ngram_repetition(text: str, k: int, n: int, min_repeats: int) -> tuple[bool, dict]:
    """Replace words outside top-K vocabulary with '_', then find repeated n-grams.

    Args:
        text: The thinking trace text.
        k: Keep only the top-K most frequent words.
        n: N-gram size.
        min_repeats: Minimum repetition count to flag.

    Returns:
        (flagged, details) with repeated n-grams and their counts.
    """
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < n:
        return False, {}

    # Length-scale min_repeats so long texts aren't unfairly penalized
    effective_min_repeats = length_scaled_min_repeats(min_repeats, len(words))

    word_freq = Counter(words)
    top_k_words = {w for w, _ in word_freq.most_common(k)}

    # Replace non-top-k words with placeholder
    reduced = [w if w in top_k_words else "_" for w in words]

    # Build n-grams
    ngrams = [tuple(reduced[i : i + n]) for i in range(len(reduced) - n + 1)]
    ngram_counts = Counter(ngrams)

    # Filter: skip all-placeholder n-grams
    meaningful_repeats = {
        " ".join(ng): count
        for ng, count in ngram_counts.items()
        if count >= effective_min_repeats and not all(w == "_" for w in ng)
    }

    details = {
        "k": k,
        "n": n,
        "min_repeats": min_repeats,
        "effective_min_repeats": effective_min_repeats,
        "num_words": len(words),
        "top_k_words": sorted(top_k_words),
        "repeated_ngrams": dict(sorted(meaningful_repeats.items(), key=lambda x: -x[1])[:10]),
    }
    return len(meaningful_repeats) > 0, details


def topk_vocab_combined(text: str, configs: list[tuple[int, int, int]] | None = None) -> tuple[bool, dict]:
    """Run top-K vocab n-gram detection with multiple configurations.

    Returns (flagged, details) where flagged is True if any config triggers.
    """
    if configs is None:
        configs = DEFAULT_TOPK_CONFIGS

    all_details = {}
    any_flagged = False
    for k, n, min_rep in configs:
        flagged, details = topk_vocab_ngram_repetition(text, k, n, min_rep)
        key = f"k{k}_n{n}"
        all_details[key] = details
        if flagged:
            any_flagged = True

    return any_flagged, all_details


# ---------------------------------------------------------------------------
# Strategy 3: Paragraph-Start Pattern Repetition
# ---------------------------------------------------------------------------


def paragraph_start_repetition(
    text: str, prefix_lengths: tuple[int, ...] = (2, 3), min_paragraphs: int = 5
) -> tuple[bool, dict]:
    """Check if many paragraphs share the same opening words.

    Args:
        text: The thinking trace text.
        prefix_lengths: How many starting words to consider.
        min_paragraphs: Minimum number of paragraphs to analyze.

    Returns:
        (flagged, details) with diversity and start-ratio per prefix length.
    """
    # Split on both single and double newlines so we handle both formats
    paragraphs = [p.strip() for p in re.split(r"\n\n?", text) if p.strip()]
    if len(paragraphs) < min_paragraphs:
        return False, {"num_paragraphs": len(paragraphs), "skipped": True}

    results: dict[int, dict] = {}
    for plen in prefix_lengths:
        prefixes = []
        for p in paragraphs:
            words = p.lower().split()[:plen]
            if words:
                prefixes.append(tuple(words))

        if not prefixes:
            continue

        prefix_counts = Counter(prefixes)
        most_common_prefix, most_common_count = prefix_counts.most_common(1)[0]
        start_ratio = most_common_count / len(paragraphs)
        diversity = len(prefix_counts) / len(paragraphs)

        results[plen] = {
            "most_common_prefix": " ".join(most_common_prefix),
            "most_common_count": most_common_count,
            "start_ratio": start_ratio,
            "diversity": diversity,
        }

    details = {"num_paragraphs": len(paragraphs), "prefix_results": results}

    # Flag conditions
    flagged = (
        results.get(2, {}).get("start_ratio", 0) > 0.4
        or results.get(3, {}).get("start_ratio", 0) > 0.3
        or results.get(2, {}).get("diversity", 1) < 0.3
    )
    return flagged, details


# ---------------------------------------------------------------------------
# Strategy 4: Word-Class N-grams
# ---------------------------------------------------------------------------

FUNCTION_WORDS = frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "him",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "their",
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "of",
        "about",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "must",
        "and",
        "but",
        "or",
        "if",
        "when",
        "not",
        "no",
        "as",
        "than",
        "then",
        "there",
        "here",
        "what",
        "which",
        "who",
        "how",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "some",
        "any",
        "up",
        "out",
        "just",
        "also",
        "very",
        "too",
        "only",
        "into",
        "over",
        "after",
        "before",
        "between",
        "through",
    }
)

THINKING_MARKERS = frozenset(
    {
        "wait",
        "hmm",
        "actually",
        "let",
        "think",
        "reconsider",
        "check",
        "verify",
        "try",
        "approach",
        "alternatively",
        "perhaps",
        "maybe",
        "consider",
        "revisit",
        "rethink",
        "okay",
        "right",
        "so",
        "well",
        "hold",
        "oh",
        "hm",
        "oops",
        "no",
        "yes",
        "alright",
    }
)

_NUMBER_RE = re.compile(r"^\d")


def word_class(w: str) -> str:
    """Classify a word into M(arker), F(unction), N(umber), or C(ontent)."""
    wl = w.lower()
    if wl in THINKING_MARKERS:
        return "M"
    if wl in FUNCTION_WORDS:
        return "F"
    if _NUMBER_RE.match(w):
        return "N"
    return "C"


def word_class_ngrams(text: str, n: int = 6, min_repeats: int = 6) -> tuple[bool, dict]:
    """Classify words by type and find repeated word-class n-grams.

    Focuses on n-grams containing at least one Marker word.
    Uses n=6 (3,367 marker-containing patterns) instead of n=4 (175 patterns)
    to avoid false positives on long texts.

    Args:
        text: The thinking trace text.
        n: N-gram size.
        min_repeats: Base minimum repeat count to flag (length-scaled).

    Returns:
        (flagged, details) with repeated word-class n-grams.
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return False, {}

    # Length-scale min_repeats
    effective_min_repeats = length_scaled_min_repeats(min_repeats, len(words))

    classes = [word_class(w) for w in words]

    ngrams = [tuple(classes[i : i + n]) for i in range(len(classes) - n + 1)]
    ngram_counts = Counter(ngrams)

    # Focus on n-grams with at least one marker
    marker_repeats = {
        "".join(ng): count for ng, count in ngram_counts.items() if count >= effective_min_repeats and "M" in ng
    }

    details = {
        "n": n,
        "min_repeats": min_repeats,
        "effective_min_repeats": effective_min_repeats,
        "num_words": len(words),
        "repeated_class_ngrams": dict(sorted(marker_repeats.items(), key=lambda x: -x[1])[:10]),
    }
    return len(marker_repeats) > 0, details


# Optional: spaCy-based POS tagging (only used with --use-pos-tagging)
_spacy_nlp = None


def _get_spacy_nlp():
    global _spacy_nlp
    if SPACY_AVAILABLE:
        _spacy_nlp = spacy.load("en_core_web_sm")  # pyright: ignore

    return _spacy_nlp


def pos_tag_ngrams(text: str, n: int = 5, min_repeats: int = 10, max_chars: int = 50000) -> tuple[bool, dict]:
    """Use spaCy POS tagging to find repeated POS n-grams.

    Requires spaCy and en_core_web_sm model (``uv run --extra filtering``).
    Uses n=5 (1.4M patterns) instead of n=4 (83K patterns) to reduce false positives.

    Args:
        text: The thinking trace text.
        n: N-gram size.
        min_repeats: Base minimum repeat count to flag (length-scaled).
        max_chars: Truncate text to this many chars for performance.

    Returns:
        (flagged, details) with repeated POS n-grams.
    """
    nlp = _get_spacy_nlp()
    assert nlp is not None

    # Truncate very long texts for performance
    if len(text) > max_chars:
        text = text[:max_chars]

    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc if not token.is_space]

    if len(pos_tags) < n:
        return False, {}

    # Length-scale min_repeats
    effective_min_repeats = length_scaled_min_repeats(min_repeats, len(pos_tags))

    ngrams = [tuple(pos_tags[i : i + n]) for i in range(len(pos_tags) - n + 1)]
    ngram_counts = Counter(ngrams)

    repeated = {" ".join(ng): count for ng, count in ngram_counts.items() if count >= effective_min_repeats}

    details = {
        "n": n,
        "min_repeats": min_repeats,
        "effective_min_repeats": effective_min_repeats,
        "num_tokens": len(pos_tags),
        "repeated_pos_ngrams": dict(sorted(repeated.items(), key=lambda x: -x[1])[:10]),
    }
    return len(repeated) > 0, details


# ---------------------------------------------------------------------------
# Shared: Transition Splitting
# ---------------------------------------------------------------------------

_TRANSITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^wait[,.\s]",
        r"^but wait\b",
        r"^actually[,.\s]",
        r"^hmm[,.\s]",
        r"^hm[,.\s]",
        r"^let me (?:re|try|check|verify|think|start|go back)",
        r"^alternatively\b",
        r"^maybe I should\b",
        r"^perhaps\b",
        r"^no[,.\s]",
        r"^on second thought\b",
        r"^hold on\b",
        r"^i think\b",
        r"^so(?:,|\s+(?:the|let|i|we|if))",
        r"^okay[,.\s]",
        r"^alright[,.\s]",
    ]
]


def _split_at_transitions(text: str) -> list[str]:
    """Split text into segments at transition boundaries.

    Splits on both single and double newlines, then groups paragraphs
    at transition markers (Wait, Let me, Actually, etc.).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n?", text) if p.strip()]

    segments: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        is_transition = any(pat.match(para) for pat in _TRANSITION_PATTERNS)
        if is_transition and current:
            segments.append("\n\n".join(current))
            current = [para]
        else:
            current.append(para)
    if current:
        segments.append("\n\n".join(current))

    return segments


# ---------------------------------------------------------------------------
# Strategy 5: Thinking Segment Action Classification
# ---------------------------------------------------------------------------

TRANSITION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^wait[,.\s]",
        r"^but wait\b",
        r"^actually[,.\s]",
        r"^hmm[,.\s]",
        r"^hm[,.\s]",
        r"^let me (?:re|try|check|verify|think|start|go back)",
        r"^alternatively\b",
        r"^maybe I should\b",
        r"^perhaps\b",
        r"^no[,.\s]",
        r"^on second thought\b",
        r"^hold on\b",
        r"^i think\b",
        r"^so(?:,|\s+(?:the|let|i|we|if))",
        r"^okay[,.\s]",
        r"^alright[,.\s]",
    ]
]

ACTION_CATEGORIES: dict[str, re.Pattern[str]] = {
    "TRY": re.compile(r"(?:try|attempt|solve|approach|method|start)", re.IGNORECASE),
    "VERIFY": re.compile(r"(?:check|verify|confirm|double.check|test|validate)", re.IGNORECASE),
    "RECONSIDER": re.compile(r"(?:reconsider|rethink|revisit|think again|re-?examine)", re.IGNORECASE),
    "WAIT": re.compile(r"(?:wait|hold on|but wait|stop)", re.IGNORECASE),
    "CALCULATE": re.compile(r"(?:calculat|comput|evaluat|simplif)", re.IGNORECASE),
    "CONCLUDE": re.compile(r"(?:therefore|thus|so the answer|the answer is|conclude)", re.IGNORECASE),
}


def classify_thinking_segments(
    text: str,
    min_segments: int = 3,
    loop_ratio_threshold: float = 0.6,
    diversity_threshold: float = 0.5,
    min_action_repeat: int = 3,
) -> tuple[bool, dict]:
    """Split thinking trace at transition boundaries and classify by action type.

    Detects looping patterns where VERIFY/RECONSIDER/WAIT dominate.

    Args:
        text: The thinking trace text.
        min_segments: Minimum segments to analyze.
        loop_ratio_threshold: Fraction of segments that must be loop actions to flag.
        diversity_threshold: Maximum action diversity (unique/total) to flag.
        min_action_repeat: Minimum repetitions of any single action to flag.

    Returns:
        (flagged, details) with segment classification info.
    """
    segments = _split_at_transitions(text)

    if len(segments) < min_segments:
        return False, {"num_segments": len(segments), "skipped": True}

    # Classify each segment
    action_sequence: list[frozenset[str]] = []
    for seg in segments:
        actions: set[str] = set()
        for action, pattern in ACTION_CATEGORIES.items():
            if pattern.search(seg):
                actions.add(action)
        action_sequence.append(frozenset(actions) if actions else frozenset({"OTHER"}))

    # Metrics
    loop_actions = {"VERIFY", "RECONSIDER", "WAIT"}
    loop_count = sum(1 for a in action_sequence if a & loop_actions)
    loop_ratio = loop_count / len(action_sequence)

    unique_actions = len(set(action_sequence))
    action_diversity = unique_actions / len(action_sequence)

    action_counts = Counter(action_sequence)
    max_action_repeat = max(action_counts.values()) if action_counts else 0

    details = {
        "num_segments": len(segments),
        "loop_ratio": round(loop_ratio, 3),
        "action_diversity": round(action_diversity, 3),
        "max_action_repeat": max_action_repeat,
    }

    flagged = (
        loop_ratio > loop_ratio_threshold
        and action_diversity < diversity_threshold
        and max_action_repeat >= min_action_repeat
    )
    return flagged, details


# ---------------------------------------------------------------------------
# Strategy 6: Near-Duplicate Segment Detection
# ---------------------------------------------------------------------------


def _segment_fingerprint(segment: str) -> frozenset[tuple[str, str]]:
    """Content-word bigram fingerprint for a text segment.

    Removes function words and short words (<3 chars) to focus on topical content.
    """
    words = re.findall(r"[a-z]+", segment.lower())
    content = [w for w in words if w not in FUNCTION_WORDS and len(w) > 2]
    if len(content) < 2:
        return frozenset()
    return frozenset((content[i], content[i + 1]) for i in range(len(content) - 1))


def near_duplicate_segments(
    text: str,
    min_segments: int = 5,
    min_segment_words: int = 50,
    jaccard_threshold: float = 0.3,
    duplicate_pair_ratio: float = 0.2,
    window_size: int = 20,
    max_segments: int = 200,
) -> tuple[bool, dict]:
    """Detect near-duplicate segments using content-word bigram Jaccard similarity.

    Splits at transition boundaries, fingerprints qualifying segments, and checks
    whether too many segment pairs within a sliding window are near-duplicates.

    Args:
        text: The thinking trace text.
        min_segments: Minimum qualifying segments to analyze.
        min_segment_words: Minimum words for a segment to qualify.
        jaccard_threshold: Jaccard similarity threshold for a near-duplicate pair.
        duplicate_pair_ratio: Fraction of pairs that must be near-duplicates to flag.
        window_size: Sliding window size for pairwise comparison.
        max_segments: Cap segments for performance.

    Returns:
        (flagged, details) with near-duplicate pair statistics.
    """
    segments = _split_at_transitions(text)
    qualifying = [s for s in segments if len(re.findall(r"[a-z]+", s.lower())) >= min_segment_words]

    if len(qualifying) < min_segments:
        return False, {"num_qualifying_segments": len(qualifying), "skipped": True}

    # Limit segments for performance
    if len(qualifying) > max_segments:
        qualifying = qualifying[:max_segments]

    # Compute fingerprints
    fingerprints = [_segment_fingerprint(s) for s in qualifying]

    # Count near-duplicate pairs in sliding window
    total_pairs = 0
    duplicate_pairs = 0

    for i in range(len(fingerprints)):
        window_end = min(i + window_size, len(fingerprints))
        for j in range(i + 1, window_end):
            fp_i, fp_j = fingerprints[i], fingerprints[j]
            if not fp_i or not fp_j:
                continue
            total_pairs += 1
            intersection = len(fp_i & fp_j)
            union = len(fp_i | fp_j)
            jaccard = intersection / union if union > 0 else 0
            if jaccard >= jaccard_threshold:
                duplicate_pairs += 1

    ratio = duplicate_pairs / total_pairs if total_pairs > 0 else 0

    details = {
        "num_qualifying_segments": len(qualifying),
        "total_pairs_checked": total_pairs,
        "duplicate_pairs": duplicate_pairs,
        "duplicate_pair_ratio": round(ratio, 3),
    }

    return ratio >= duplicate_pair_ratio, details


# ---------------------------------------------------------------------------
# Combined Detection
# ---------------------------------------------------------------------------


def detect_thinking_repetition(
    thinking_text: str,
    min_strategies: int = 2,
    marker_threshold_scale: float = 1.0,
    topk_configs: list[tuple[int, int, int]] | None = None,
    use_pos_tagging: bool = False,
) -> tuple[bool, str, dict]:
    """Run all detection strategies and combine results.

    Args:
        thinking_text: Text between <think> and </think> tags.
        min_strategies: How many strategies must agree to flag.
        marker_threshold_scale: Scaling factor for marker thresholds.
        topk_configs: Top-K vocabulary configurations.
        use_pos_tagging: If True, use spaCy POS tagging for Strategy 4.

    Returns:
        (has_repetition, reason_string, all_details)
    """
    if not thinking_text or len(thinking_text.strip()) < 100:
        return False, "", {"skipped": "text too short"}

    scores: dict[str, dict] = {}
    reasons: list[str] = []

    # Strategy 1: Marker phrase density
    marker_flagged, marker_details = marker_phrase_density(thinking_text, marker_threshold_scale)
    scores["marker_density"] = marker_details
    if marker_flagged:
        reasons.append("marker_density")

    # Strategy 2: Top-K vocab n-grams
    topk_flagged, topk_details = topk_vocab_combined(thinking_text, topk_configs)
    scores["topk_vocab"] = topk_details
    if topk_flagged:
        reasons.append("topk_vocab")

    # Strategy 3: Paragraph start patterns
    para_flagged, para_details = paragraph_start_repetition(thinking_text)
    scores["paragraph_starts"] = para_details
    if para_flagged:
        reasons.append("paragraph_starts")

    # Strategy 4: Word class or POS n-grams
    if use_pos_tagging:
        wc_flagged, wc_details = pos_tag_ngrams(thinking_text)
        scores["pos_tag_ngrams"] = wc_details
        if wc_flagged:
            reasons.append("pos_tag_ngrams")
    else:
        wc_flagged, wc_details = word_class_ngrams(thinking_text)
        scores["word_class_ngrams"] = wc_details
        if wc_flagged:
            reasons.append("word_class_ngrams")

    # Strategy 5: Segment action classification
    seg_flagged, seg_details = classify_thinking_segments(thinking_text)
    scores["segment_classification"] = seg_details
    if seg_flagged:
        reasons.append("segment_classification")

    # Strategy 6: Near-duplicate segment detection
    dup_flagged, dup_details = near_duplicate_segments(thinking_text)
    scores["near_duplicate_segments"] = dup_details
    if dup_flagged:
        reasons.append("near_duplicate_segments")

    has_repetition = len(reasons) >= min_strategies
    reason = ", ".join(reasons) if reasons else ""

    return has_repetition, reason, scores


# ---------------------------------------------------------------------------
# Dataset Processing
# ---------------------------------------------------------------------------


def process_example(
    example: dict,
    column: str,
    index: int,
    min_strategies: int = 2,
    marker_threshold_scale: float = 1.0,
    topk_configs: list[tuple[int, int, int]] | None = None,
    use_pos_tagging: bool = True,
) -> dict:
    """Process a single example through all detection strategies.

    Returns the example dict with added fields:
        - has_thinking_repetition: bool
        - thinking_repetition_reason: str
        - thinking_repetition_strategies_triggered: int
        - hit_marker_density: bool
        - hit_topk_vocab: bool
        - hit_paragraph_starts: bool
        - hit_pos_or_wordclass: bool  (POS tagging or word-class, depending on mode)
        - hit_segment_classification: bool
        - hit_near_duplicate_segments: bool
    """
    _empty = {
        "has_thinking_repetition": False,
        "thinking_repetition_reason": "",
        "thinking_repetition_strategies_triggered": 0,
        "hit_marker_density": False,
        "hit_topk_vocab": False,
        "hit_paragraph_starts": False,
        "hit_pos_or_wordclass": False,
        "hit_segment_classification": False,
        "hit_near_duplicate_segments": False,
    }

    thinking_text = extract_thinking_trace(example, column)
    if thinking_text is None:
        return _empty

    has_rep, reason, _details = detect_thinking_repetition(
        thinking_text,
        min_strategies=min_strategies,
        marker_threshold_scale=marker_threshold_scale,
        topk_configs=topk_configs,
        use_pos_tagging=use_pos_tagging,
    )

    reasons_set = set(reason.split(", ")) if reason else set()

    return {
        "has_thinking_repetition": bool(has_rep),
        "thinking_repetition_reason": str(reason) if reason else "",
        "thinking_repetition_strategies_triggered": len(reasons_set) if reasons_set else 0,
        "hit_marker_density": "marker_density" in reasons_set,
        "hit_topk_vocab": "topk_vocab" in reasons_set,
        "hit_paragraph_starts": "paragraph_starts" in reasons_set,
        "hit_pos_or_wordclass": "pos_tag_ngrams" in reasons_set or "word_class_ngrams" in reasons_set,
        "hit_segment_classification": "segment_classification" in reasons_set,
        "hit_near_duplicate_segments": "near_duplicate_segments" in reasons_set,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Filter examples with repetitive thinking traces (semantic repetition detection)"
    )
    parser.add_argument("--input-dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument(
        "--output-dataset", type=str, default=None, help="Output dataset name (default: <input>-thinking-filtered)"
    )
    parser.add_argument(
        "--column", type=str, default="messages", help="Column containing messages (default: messages)"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--debug", action="store_true", help="Process only first 1000 examples")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode to avoid downloading the full dataset. Takes only the first --num-examples rows.",
    )
    parser.add_argument(
        "--num-examples", type=int, default=100, help="Number of examples to take in streaming mode (default: 100)"
    )
    parser.add_argument("--push-to-hf", action="store_true", help="Push filtered dataset to HuggingFace Hub")
    parser.add_argument("--num-proc", type=int, default=None, help="Number of processes (default: auto)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed analysis of flagged examples")
    parser.add_argument(
        "--min-strategies",
        type=int,
        default=2,
        help="Minimum number of strategies that must agree to flag (default: 2, use 1 for aggressive)",
    )
    parser.add_argument(
        "--no-pos-tagging",
        action="store_true",
        help="Disable spaCy POS tagging and use lightweight word-class system instead. "
        "Disable spaCy POS tagging and use lightweight word-class system instead. "
        "By default, spaCy POS tagging is used (requires uv run --extra filtering).",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze and print statistics, don't filter")
    parser.add_argument(
        "--marker-threshold-scale",
        type=float,
        default=1.0,
        help="Scale marker phrase density thresholds (>1 = more lenient, <1 = stricter, default: 1.0)",
    )
    parser.add_argument(
        "--topk-k-values",
        type=str,
        default="10,20,50",
        help="Comma-separated K values for top-K vocab strategy (default: 10,20,50). "
        "Ignored if --topk-configs is provided.",
    )
    parser.add_argument(
        "--topk-configs",
        type=str,
        default=None,
        help="Custom top-K configs as comma-separated K:n:min_repeats triples "
        "(e.g. '10:4:8,20:5:5,50:6:3'). Overrides --topk-k-values and DEFAULT_TOPK_CONFIGS.",
    )
    args = parser.parse_args()

    # Determine POS tagging mode (default: on, unless --no-pos-tagging)
    use_pos_tagging = not args.no_pos_tagging
    if use_pos_tagging:
        if not SPACY_AVAILABLE:
            logger.warning(
                "spaCy not installed. Falling back to word-class system.\n"
                "To enable POS tagging, run with: uv run --extra filtering"
            )
            use_pos_tagging = False
        else:
            try:
                _get_spacy_nlp()
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. Falling back to word-class system.\n"
                    "To enable POS tagging, run with: uv run --extra filtering"
                )
                use_pos_tagging = False

    num_proc = args.num_proc or open_instruct_utils.max_num_processes()

    # Parse top-K configs
    if args.topk_configs:
        # Explicit K:n:min_repeats triples override everything
        topk_configs = []
        for triple in args.topk_configs.split(","):
            parts = triple.strip().split(":")
            if len(parts) != 3:
                parser.error(f"Invalid topk-configs triple: '{triple}'. Expected K:n:min_repeats.")
            topk_configs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    else:
        # Use defaults (the expanded DEFAULT_TOPK_CONFIGS) unless --topk-k-values overrides
        if args.topk_k_values != "10,20,50":
            k_values = [int(k.strip()) for k in args.topk_k_values.split(",")]
            topk_configs = []
            for k in k_values:
                if k <= 10:
                    topk_configs.append((k, 4, 8))
                elif k <= 30:
                    topk_configs.append((k, 4, 5))
                else:
                    topk_configs.append((k, 5, 4))
        else:
            topk_configs = None  # Use DEFAULT_TOPK_CONFIGS

    # Load dataset
    logger.info(f"Loading dataset: {args.input_dataset}")
    if args.streaming:
        logger.info(f"Streaming mode: taking first {args.num_examples} examples")
        stream = load_dataset(args.input_dataset, split=args.split, streaming=True)
        rows = list(stream.take(args.num_examples))
        dataset = Dataset.from_list(rows)
        logger.info(f"Loaded {len(dataset)} examples via streaming")
    else:
        dataset = load_dataset(args.input_dataset, split=args.split, num_proc=num_proc)
        if args.debug:
            dataset = dataset.select(range(min(1000, len(dataset))))
            logger.info(f"Debug mode: using first {len(dataset)} examples")

    original_count = len(dataset)
    logger.info(f"Original dataset size: {original_count}")

    # Process dataset
    new_features = dataset.features.copy()
    new_features["has_thinking_repetition"] = Value("bool")
    new_features["thinking_repetition_reason"] = Value("string")
    new_features["thinking_repetition_strategies_triggered"] = Value("int32")
    new_features["hit_marker_density"] = Value("bool")
    new_features["hit_topk_vocab"] = Value("bool")
    new_features["hit_paragraph_starts"] = Value("bool")
    new_features["hit_pos_or_wordclass"] = Value("bool")
    new_features["hit_segment_classification"] = Value("bool")
    new_features["hit_near_duplicate_segments"] = Value("bool")

    logger.info("Detecting thinking trace repetitions...")
    dataset_with_flags = dataset.map(
        lambda x, idx: process_example(
            x,
            column=args.column,
            index=idx,
            min_strategies=args.min_strategies,
            marker_threshold_scale=args.marker_threshold_scale,
            topk_configs=topk_configs,
            use_pos_tagging=use_pos_tagging,
        ),
        num_proc=num_proc,
        with_indices=True,
        desc="Detecting thinking repetitions",
        features=new_features,
    )

    # Compute statistics
    flagged_dataset = dataset_with_flags.filter(lambda x: x["has_thinking_repetition"])
    flagged_count = len(flagged_dataset)

    logger.info(f"Flagged: {flagged_count} / {original_count} ({flagged_count / max(original_count, 1) * 100:.2f}%)")

    # Per-strategy breakdown
    reason_counts: Counter[str] = Counter()
    for example in flagged_dataset:
        for reason in example["thinking_repetition_reason"].split(", "):
            if reason:
                reason_counts[reason] += 1

    logger.info("Strategy breakdown (among flagged examples):")
    for reason, count in reason_counts.most_common():
        logger.info(f"  {reason}: {count}")

    # Distribution of strategy counts
    strategy_count_dist: Counter[int] = Counter()
    for example in dataset_with_flags:
        strategy_count_dist[example["thinking_repetition_strategies_triggered"]] += 1

    logger.info("Strategy trigger distribution (all examples):")
    for n_strategies in sorted(strategy_count_dist):
        count = strategy_count_dist[n_strategies]
        logger.info(f"  {n_strategies} strategies: {count} examples ({count / original_count * 100:.2f}%)")

    # Verbose: print details of flagged examples
    if args.verbose and flagged_count > 0:
        logger.info(f"\nShowing first {min(10, flagged_count)} flagged examples:")
        for i, example in enumerate(flagged_dataset.select(range(min(10, flagged_count)))):
            thinking = extract_thinking_trace(example, args.column)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Example {i} | Reason: {example['thinking_repetition_reason']}")
            if thinking:
                preview = thinking[:2000] + ("..." if len(thinking) > 2000 else "")
                logger.info(f"Thinking trace preview:\n{preview}")

    if args.analyze_only:
        logger.info("Analyze-only mode: not filtering or pushing.")
        return

    # Determine output name
    if args.output_dataset:
        output_name = args.output_dataset
    else:
        if "/" in args.input_dataset:
            org, name = args.input_dataset.split("/", 1)
            output_name = f"{org}/{name}-thinking-filtered"
        else:
            output_name = f"{args.input_dataset}-thinking-filtered"

    if args.push_to_hf:
        # Push annotated dataset (keeps all heuristic columns for inspection)
        logger.info(f"Pushing annotated dataset to: {output_name}")
        dataset_with_flags.push_to_hub(output_name, private=True, num_proc=num_proc)
        logger.info("Done!")
    else:
        logger.info(f"Would push to: {output_name} (use --push-to-hf to actually push)")


if __name__ == "__main__":
    main()
