"""Render Dolci-Think-RL math, instruction-following and code prompts for Megatron GRPO.

The multi-domain RL study trains single-domain specialists and domain mixes on the
Olmo 3 RL prompts (``allenai/Dolci-Think-RL-7B``). Each source row names one Open
Instruct verifier in ``dataset`` (``math``, ``ifeval``, ``code`` or ``code_stdio``) with
its target in ``ground_truth``; its ``prompt`` is the single user turn flattened as
``user: <content>``. This script keeps those three domains, renders the user turn with the
learner's chat template (thinking off), holds out a fixed evaluation set per domain, and
writes Miles ``data.prompt_data`` / ``data.eval_prompt_data`` JSONL plus the
``verifiers.json`` registry for ``data.reward_config``:

- ``train_<domain>.jsonl`` and ``eval_<domain>.jsonl`` for each domain;
- ``mix_<name>.jsonl`` for each ``--mix``, sampled without repetition at the given ratio;
- ``verifiers.json``, which scores each row with the verifier Open Instruct uses for it;
- ``manifest.json`` with the source revision, row counts, drops and file hashes.

``--reward-fixes`` applies the MOPD retrain audit's data and scoring fixes (data v3):

- math prompts that ask for no answer format get a boxed-answer instruction, rows with
  empty targets are dropped, and ``math`` is scored by
  ``math_rewards.MathVerifyVerifier`` (Math-Verify on the boxed answer);
- instruction-following rows that cannot be fully satisfied (contradictory or invalid
  constraints) or that the one-letter answer "a" fully passes are dropped, and
  ``ifeval`` is scored all-or-nothing by ``if_rewards.StrictIFVerifier``;
- code rows whose tests can never pass or exceed the service's payload limit are dropped,
  prompts that never name the tested function or (stdio) never ask for Python get a
  sentence that does, and grader failures are masked from the loss instead of scored 0;
- a prompt that appears with different targets keeps one row when the targets agree (math
  targets Math-Verify reads as equal; for code, the copy with the most tests) and is dropped
  otherwise, and training excludes rows whose normalized user text matches a held-out row.

Example (inside the Miles image, writing to WEKA):

    python scripts/miles/prepare_dolci_rl_prompts.py \\
        --tokenizer /weka/.../assets-pr1924/Qwen3.5-9B-c20223623576 \\
        --source allenai/Dolci-Think-RL-7B@<40-character revision> \\
        --mix all3=math:1,ifeval:1,code:1 --output /weka/.../data/dolci-rl-v1
"""

import argparse
import ast
import base64
import binascii
import hashlib
import json
import random
import re
import zlib
from pathlib import Path

import datasets
from transformers import AutoTokenizer

from open_instruct import logger_utils
from open_instruct.miles.rewards import code_rewards, if_rewards, math_rewards

logger = logger_utils.setup_logger(__name__)

DOMAINS = {"math": ("math",), "ifeval": ("ifeval",), "code": ("code", "code_stdio")}
"""Study domain -> the Open Instruct verifier names its source rows carry."""
COLUMNS = ["id", "prompt", "ground_truth", "dataset", "dataset_source", "original_dataset"]
DEFAULT_CODE_API_URL = "https://p9f1719l7f.execute-api.us-west-2.amazonaws.com/prod/test_program"
_USER_PREFIX = "user: "
_OTHER_TURN = re.compile(r"\n(?:assistant|system|user): ")


def verifier_registry(code_api_url, reward_fixes=False):
    """The trusted ``data.reward_config``: Open Instruct's verifier for each source verifier name.

    Code follows the Olmo 3 RL recipe: the pass rate when at least 99% of the tests pass, else 0.
    With ``reward_fixes``, math is scored by Math-Verify on the boxed answer, instruction
    following is scored all-or-nothing and code grader failures are masked from the loss.
    """
    code = {"api_url": code_api_url, "pass_rate_reward_threshold": 0.99}
    math = {"factory": "open_instruct.ground_truth_utils.MathVerifier"}
    ifeval = {"factory": "open_instruct.ground_truth_utils.IFEvalVerifier"}
    if reward_fixes:
        math = {"factory": "open_instruct.miles.rewards.math_rewards.MathVerifyVerifier"}
        code["failure_policy"] = "mask"
        ifeval = {"factory": "open_instruct.miles.rewards.if_rewards.StrictIFVerifier"}
    return {
        "math": math,
        "ifeval": ifeval,
        "code": {"factory": "open_instruct.miles.rewards.code_rewards.CodeVerifier", "config": code},
        "code_stdio": {
            "factory": "open_instruct.miles.rewards.code_rewards.CodeVerifier",
            "config": {**code, "stdio": True},
        },
    }


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def user_content(prompt):
    """The single user turn of a flattened ``user: ...`` prompt, or None for anything else."""
    if not isinstance(prompt, str) or not prompt.startswith(_USER_PREFIX):
        return None
    content = prompt[len(_USER_PREFIX) :]
    if not content.strip() or _OTHER_TURN.search(content):
        return None
    return content


def verifier_target(name, ground_truth):
    """The Miles verifier target for one source row's Open Instruct ``ground_truth`` entry."""
    if name == "math":
        if not isinstance(ground_truth, str) or not ground_truth.strip():
            raise ValueError("math target must be a nonempty string")
        return ground_truth
    if name == "ifeval":
        # IFEvalVerifier parses this repr itself: one constraint bundle with parallel lists.
        bundle = ast.literal_eval(ground_truth)[0]
        if isinstance(bundle, str):
            bundle = json.loads(bundle)
        ids, kwargs = bundle["instruction_id"], bundle["kwargs"]
        if not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs):
            raise ValueError("ifeval target needs matching nonempty instruction_id and kwargs lists")
        return ground_truth
    # code and code_stdio: the execution service takes the list of tests itself.
    try:
        tests = json.loads(ground_truth)
    except json.JSONDecodeError:
        return compressed_tests(ground_truth)
    if not isinstance(tests, list) or not tests:
        raise ValueError(f"{name} target must be a nonempty list of tests")
    return tests


def compressed_tests(ground_truth):
    """A base64 zlib-compressed pickled test list, passed to the execution service unchanged.

    The service's ``decode_tests`` unpacks this form (``code_utils.encode_tests``); some
    Dolci code sources ship it. Preparation checks the encoding but never unpickles it.
    """
    try:
        raw = zlib.decompress(base64.b64decode(ground_truth, validate=True))
    except (binascii.Error, zlib.error) as error:
        raise ValueError("code target is neither a JSON test list nor compressed tests") from error
    if not raw.startswith(b"\x80"):
        raise ValueError("compressed code target is not a pickle")
    return ground_truth


def fix_prompt(name, content, target):
    """(content, fixes) with the ``--reward-fixes`` prompt fixes applied, or (None, reason) to drop the row."""
    if name == "math":
        problems = math_rewards.math_row_problems({"ground_truth": target})
        if problems:
            return None, f"math_{problems[0]}"
        fixed = math_rewards.add_boxed_instruction(content)
        return fixed, ["boxed_instruction"] if fixed != content else []
    if name == "ifeval":
        problems = if_rewards.if_row_problems({"prompt": content, "ground_truth": target})
        return (None, f"ifeval_{problems[0].split(':')[0]}") if problems else (content, [])
    problems = code_rewards.code_target_problems(name, target, content)
    blocking = [p for p in problems if p not in code_rewards.FIXABLE_PROBLEMS and p != "non_assert_test"]
    if blocking:
        return None, f"code_{blocking[0]}"
    fixes = []
    if "function_name_missing" in problems:
        function_name = code_rewards.function_name_from_tests(code_rewards.decode_tests_safely(target))
        content = code_rewards.add_function_name_hint(content, function_name)
        fixes.append("function_name_hint")
    if "language_unspecified" in problems:
        content = code_rewards.add_python_hint(content)
        fixes.append("python_hint")
    return content, fixes


def render_row(row, tokenizer, source, reward_fixes=False):
    """One prepared Miles row, or (None, reason) when the source row is outside the study."""
    names = row["dataset"]
    if not isinstance(names, list) or len(names) != 1:
        return None, "not_single_verifier"
    name = names[0]
    domain = next((domain for domain, members in DOMAINS.items() if name in members), None)
    if domain is None:
        return None, "other_domain"
    content = user_content(row["prompt"])
    if content is None:
        return None, "not_single_user_turn"
    truths = row["ground_truth"]
    if not isinstance(truths, list) or len(truths) != 1:
        return None, "not_single_target"
    try:
        target = verifier_target(name, truths[0])
    except (ValueError, SyntaxError, KeyError, IndexError, TypeError):
        return None, "malformed_target"
    fixes = None
    label = truths[0]
    if reward_fixes:
        content, fixes = fix_prompt(name, content, target)
        if content is None:
            return None, fixes
        if name == "ifeval":
            # State the count relation the prompt implies, so the stored target matches the prompt.
            target = label = if_rewards.normalize_if_target(content, target)
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {
        "input": prompt,
        "label": label,
        "metadata": {
            "source_dataset": source["source"],
            "source_revision": source["revision"],
            "source_id": row["id"],
            "domain": domain,
            "dataset_source": row["dataset_source"],
            "original_dataset": row["original_dataset"],
            "query": content,
            "verifiers": [{"name": name, "target": target, "weight": 1.0}],
            **({"prompt_fixes": fixes} if fixes is not None else {}),
        },
    }, domain


def resolve_duplicates(rows):
    """The row to keep among the prepared copies of one prompt, or None when their targets disagree.

    Math copies whose targets Math-Verify reads as the same answer keep the first. Code copies
    carry test lists for the same problem, often one a subset of another, so the copy with the
    most tests is kept.
    """
    labels = [row["label"] for row in rows]
    if all(label == labels[0] for label in labels):
        return rows[0]
    domains = {row["metadata"]["domain"] for row in rows}
    if domains == {"math"} and all(math_rewards.targets_equivalent(labels[0], label) for label in labels[1:]):
        return rows[0]
    if domains == {"code"}:
        return max(rows, key=lambda row: len(code_rewards.decode_tests_safely(row["metadata"]["verifiers"][0]["target"]) or []))
    return None


def normalized_query(row):
    """The user text lowercased with punctuation and whitespace runs collapsed, for leakage checks."""
    return " ".join(re.sub(r"[^\w\s]", " ", row["metadata"]["query"].lower()).split())


def split_from_eval(rows, held_out, seed, key=None):
    """Training rows for a fixed, previously prepared held-out set: the seeded shuffle minus held-out prompts."""
    key = key or (lambda row: row["input"])
    keys = {key(row) for row in held_out}
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return [row for row in shuffled if key(row) not in keys]


def split_domain(rows, eval_count, seed, key=None):
    """Deterministic (eval, train) split; train excludes any prompt that is also held out.

    ``key`` (default: the rendered input) decides which training rows count as held out.
    """
    key = key or (lambda row: row["input"])
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    held_out = shuffled[:eval_count]
    keys = {key(row) for row in held_out}
    return held_out, [row for row in shuffled[eval_count:] if key(row) not in keys]


def parse_mix(text):
    """``NAME=domain:weight,...`` -> (name, {domain: weight})."""
    name, _, spec = text.partition("=")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or not spec:
        raise SystemExit(f"--mix expects NAME=domain:weight,...; got {text!r}")
    weights = {}
    for part in spec.split(","):
        domain, _, weight = part.partition(":")
        if domain not in DOMAINS or domain in weights:
            raise SystemExit(f"--mix {name}: unknown or repeated domain {domain!r}")
        weights[domain] = float(weight or 1)
        if weights[domain] <= 0:
            raise SystemExit(f"--mix {name}: weights must be positive")
    return name, weights


def mix_rows(train, weights, seed):
    """The largest mix at the given ratio that repeats no prompt, shuffled."""
    scale = min(len(train[domain]) / weight for domain, weight in weights.items())
    counts = {domain: int(scale * weight) for domain, weight in weights.items()}
    rng = random.Random(seed)
    rows = [row for domain, count in counts.items() for row in rng.sample(train[domain], count)]
    rng.shuffle(rows)
    return rows, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="Learner tokenizer: prepared local checkpoint or HF id")
    parser.add_argument("--tokenizer-revision", default=None, help="Immutable revision for an HF tokenizer id")
    parser.add_argument("--source", required=True, help="HF dataset as repo@40-character-revision")
    parser.add_argument("--eval-per-domain", type=int, default=256, help="Held-out prompts per domain")
    parser.add_argument(
        "--eval-from",
        default=None,
        help="Prepared directory whose eval_<domain>.jsonl files are reused unchanged (overrides --eval-per-domain)",
    )
    parser.add_argument("--mix", action="append", default=[], metavar="NAME=DOMAIN:WEIGHT,...")
    parser.add_argument("--max-prompt-tokens", type=int, default=2048, help="Drop longer rendered prompts")
    parser.add_argument("--code-api-url", default=DEFAULT_CODE_API_URL, help="Code execution /test_program URL")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reward-fixes", action="store_true", help="Apply the MOPD retrain audit's data and scoring fixes (v3)"
    )
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args()

    repo, _, revision = args.source.partition("@")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit("--source needs an immutable 40-character revision")
    if not args.code_api_url.endswith("/test_program"):
        raise SystemExit("--code-api-url must end in /test_program; the stdio verifier derives its URL from it")
    mixes = dict(parse_mix(text) for text in args.mix)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} exists; choose a fresh output directory")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    source = {"source": repo, "revision": revision}
    dataset = datasets.load_dataset(repo, revision=revision, split="train", columns=COLUMNS)
    by_domain = {domain: [] for domain in DOMAINS}
    drops = {}
    copies = {}
    for row in dataset:
        prepared, outcome = render_row(row, tokenizer, source, args.reward_fixes)
        if prepared is None:
            drops[outcome] = drops.get(outcome, 0) + 1
            continue
        if prepared["input"] in copies:
            drops["duplicate_prompt"] = drops.get("duplicate_prompt", 0) + 1
            copies[prepared["input"]].append(prepared)
            continue
        if len(tokenizer.encode(prepared["input"], add_special_tokens=False)) > args.max_prompt_tokens:
            drops["prompt_too_long"] = drops.get("prompt_too_long", 0) + 1
            continue
        copies[prepared["input"]] = [prepared]
        by_domain[outcome].append(prepared)
    resolved_targets = {}
    if args.reward_fixes:
        for domain, rows in by_domain.items():
            kept = []
            for row in rows:
                group = copies[row["input"]]
                chosen = resolve_duplicates(group)
                if chosen is None:
                    drops["conflicting_targets"] = drops.get("conflicting_targets", 0) + 1
                    continue
                if len({copy["label"] for copy in group}) > 1:
                    resolved_targets[domain] = resolved_targets.get(domain, 0) + 1
                kept.append(chosen)
            by_domain[domain] = kept

    files = {}
    train = {}
    eval_from = {}
    for domain, rows in by_domain.items():
        key = normalized_query if args.reward_fixes else None
        if args.eval_from:
            raw = (Path(args.eval_from) / f"eval_{domain}.jsonl").read_bytes()
            held_out = [json.loads(line) for line in raw.decode().splitlines()]
            eval_from[f"eval_{domain}"] = _sha(raw)
            train[domain] = split_from_eval(rows, held_out, args.seed, key)
        else:
            held_out, train[domain] = split_domain(rows, args.eval_per_domain, args.seed, key)
        files[f"eval_{domain}"] = held_out
        files[f"train_{domain}"] = train[domain]
    mix_counts = {}
    for name, weights in mixes.items():
        files[f"mix_{name}"], mix_counts[name] = mix_rows(train, weights, args.seed)

    output.mkdir(parents=True)
    manifest = {
        **source,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "template_sha256": _sha(tokenizer.get_chat_template().encode()),
        "enable_thinking": False,
        "seed": args.seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "reward_fixes": args.reward_fixes,
        "eval_from": {"path": args.eval_from, "sha256": eval_from} if args.eval_from else None,
        "dropped": dict(sorted(drops.items())),
        "resolved_duplicate_targets": dict(sorted(resolved_targets.items())),
        "mixes": {name: {"weights": mixes[name], "counts": mix_counts[name]} for name in mixes},
        "files": {},
    }
    for stem, rows in files.items():
        for index, row in enumerate(rows):
            row["metadata"]["prepared_sample_id"] = f"{stem}:{index}"
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        (output / f"{stem}.jsonl").write_bytes(raw)
        verifiers = {}
        for row in rows:
            name = row["metadata"]["verifiers"][0]["name"]
            verifiers[name] = verifiers.get(name, 0) + 1
        manifest["files"][stem] = {"records": len(rows), "verifiers": verifiers, "sha256": _sha(raw)}
        logger.info("%s: %d prompts %s", stem, len(rows), verifiers)
    registry = verifier_registry(args.code_api_url, args.reward_fixes)
    (output / "verifiers.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Dropped %s; wrote %s", manifest["dropped"], output)


if __name__ == "__main__":
    main()
