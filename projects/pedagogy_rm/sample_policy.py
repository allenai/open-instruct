"""Draw turns from a policy on held-out questions, and score them with the probe.

    python projects/pedagogy_rm/sample_policy.py generate \
        --policy allenai/OLMo-2-1124-7B-Instruct --out data/samples/base.json
    python projects/pedagogy_rm/sample_policy.py score \
        --units data/samples/base.json

WHY THIS IS NOT A HOOK INSIDE THE TRAINING LOOP. The question worth answering is
whether the probe's number means anything about teaching, and that cannot be asked
from inside the loop maximising it: both the training reward and the held-out reward
are computed by the same head, so both rise together whether the policy learned to
teach or learned what this particular head likes. Sampling from the outside gives a
file that a human can read and that label_agents.py can rate, which is the only
evidence that separates the two.

THE OUTPUT IS A LABELLING UNITS FILE, deliberately. It carries the same keys as
data/label_slices/*.json, so label_agents.py rates policy output with no changes and
agreement.py compares those ratings to anything else on the same scales. A bespoke
format here would have meant a converter, and a converter is where the turn a human
read stops being the turn the probe scored.

TWO SUBCOMMANDS BECAUSE OF ONE GPU. vLLM reserves a fraction of the whole card up
front and the probe's encoder wants about 9GB of what is left; freeing vLLM inside a
live process means tearing down its distributed state, which is fragile enough that
a crash there would look like a scoring bug. Separate processes cost one model load
and are worth it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from projects.pedagogy_rm.plugin import SIGNS, PedagogyHead

UNIT_KEYS = ("question", "choices", "gold", "student_before", "subject", "grade", "turn_index")


def read_prompts(path: str, limit: int) -> list[dict]:
    with open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:limit] if limit else rows


def unit_id(messages: list[dict], text: str, index: int) -> str:
    """Stable across runs of the same policy, distinct across samples of one prompt.

    Hashed rather than sequential so that a units file merged with another does not
    silently reuse an id, which is what label_agents.py keys its output on.
    """
    material = json.dumps(messages, sort_keys=True) + text + str(index)
    return "s" + hashlib.blake2b(material.encode(), digest_size=6).hexdigest()


def generate(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    rows = read_prompts(args.prompts, args.limit)
    tokenizer = AutoTokenizer.from_pretrained(args.policy)
    prompts = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True) for row in rows
    ]

    lora = None
    engine_kwargs = {}
    if args.adapter:
        from vllm.lora.request import LoRARequest  # noqa: PLC0415

        engine_kwargs = {"enable_lora": True, "max_lora_rank": args.lora_rank}
        lora = LoRARequest("policy", 1, args.adapter)

    llm = LLM(
        model=args.policy,
        dtype="bfloat16",
        gpu_memory_utilization=args.vllm_util,
        max_model_len=args.max_model_len,
        **engine_kwargs,
    )
    # n>1 draws several turns for one prompt, which is what makes a mean per policy an
    # estimate rather than one roll of the dice. The temperature matches training: a
    # policy scored greedily is not the policy the reward was computed on.
    sampling = SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens)
    outputs = llm.generate(prompts, sampling, lora_request=lora)

    units = []
    for row, output in zip(rows, outputs, strict=True):
        item = json.loads(row["ground_truth"])
        for index, candidate in enumerate(output.outputs):
            turn = candidate.text.strip()
            if not turn:
                continue
            units.append(
                {
                    "id": unit_id(row["messages"], turn, index),
                    **{key: item.get(key) for key in UNIT_KEYS},
                    "tutor_turn": turn,
                    "style": args.tag,
                    "temperature": args.temperature,
                    "sample_index": index,
                    "messages": row["messages"],
                }
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"units": units, "policy": args.policy, "adapter": args.adapter, "tag": args.tag}, handle, indent=1)
    print(f"wrote {args.out}: {len(units)} turns from {len(rows)} prompts")


def score(args: argparse.Namespace) -> None:
    import numpy as np  # noqa: PLC0415

    with open(args.units) as handle:
        blob = json.load(handle)
    units = blob["units"]

    head = PedagogyHead(head=args.head, device=args.device, batch_size=args.batch_size)
    # The contexts are the prompt messages exactly as the RL rows carry them, which
    # build_rl_data.py built with extract_hidden.context_messages. Rebuilding them here
    # from the question and the student turn would be a second implementation of the one
    # thing that has to agree with how the head was fitted.
    contexts = [(unit["messages"], unit["tutor_turn"]) for unit in units]
    pooled = head.states(contexts)

    for dim in head.dims:
        spec = head.meta["dimensions"][dim]
        weight = head.weights[dim]
        x = (np.stack(pooled[(spec["pooling"], spec["layer"])]) - weight["mean"]) / weight["scale"]
        raw = x @ weight["coef"] + weight["intercept"]
        for unit, value in zip(units, raw, strict=True):
            unit.setdefault("probe", {})[dim] = float(np.clip(value, spec["lo"], spec["hi"]))

    for unit in units:
        # The same aggregation plugin.score_group returns as `score`, so a number read
        # here is the number the policy was trained against rather than a relative of it.
        signed = [SIGNS[dim] * unit["probe"][dim] for dim in head.dims]
        unit["probe"]["total"] = float(sum(signed) / len(signed))

    blob["dimensions"] = head.dims
    with open(args.units, "w") as handle:
        json.dump(blob, handle, indent=1)

    print(f"scored {len(units)} turns in {args.units}")
    for dim in [*head.dims, "total"]:
        values = np.array([unit["probe"][dim] for unit in units])
        print(f"  {dim:11} mean {values.mean():+.3f}  sd {values.std():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="draw turns from a policy")
    gen.add_argument("--prompts", default="data/rl/eval.jsonl", help="held-out by question, as build_rl_data split it")
    gen.add_argument("--policy", default="allenai/OLMo-2-1124-7B-Instruct")
    gen.add_argument("--adapter", default="", help="a PEFT adapter directory, for a LoRA policy")
    gen.add_argument("--lora-rank", type=int, default=32)
    gen.add_argument("--tag", default="base", help="what this policy is, carried into every unit")
    gen.add_argument("--out", default="data/samples/base.json")
    gen.add_argument("--limit", type=int, default=200)
    gen.add_argument("--samples", type=int, default=2)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--max-model-len", type=int, default=2048)
    gen.add_argument("--vllm-util", type=float, default=0.60)
    gen.set_defaults(func=generate)

    sco = sub.add_parser("score", help="add probe scores to a units file, in place")
    sco.add_argument("--units", required=True)
    sco.add_argument("--head", default="data/head.npz")
    sco.add_argument("--device", default="cuda")
    sco.add_argument("--batch-size", type=int, default=16)
    sco.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
