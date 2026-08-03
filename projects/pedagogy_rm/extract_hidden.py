"""OLMo hidden states for each labelled tutor turn.

    python -m projects.pedagogy_rm.extract_hidden \
        --units data/label_slices/slice_1.json --out data/hidden.npz

THE STATES MUST MATCH THE ROLLOUT. The whole premise is a reward that is free
during training because the policy already computed it. That only holds if the
vectors probed here are the ones a rollout would produce, so this rebuilds the
exact chat context ``generate.py`` used, appends the tutor turn as the
assistant message, and reads the states over the TUTOR TURN'S tokens only.
Encoding the turn as a bare string instead would be easier, cheaper, and would
measure something the trainer can never reproduce.

TWO POOLINGS, SEVERAL LAYERS, DECIDED LATER. Which layer carries a property is
not knowable in advance and is cheap to keep open: 600 turns x 8 layers x 4096
dims x 2 poolings is about 160MB in fp16. Deciding now would be guessing, and
re-running the extraction after every guess is the expensive path.

    last   the final tutor-turn token. What a value head would see.
    mean   averaged over the tutor turn. Steadier, and usually stronger for
           properties spread across a sentence rather than resolved at its end.

The previous project used only a final hidden state with a linear head, which is
one cell of this grid - and the cell least likely to work for properties like
"is this targeted at the student's error".
"""

from __future__ import annotations

import argparse
import json

TEACHER_SYSTEM = """You are a tutor helping a student with a test question. \
The student cannot see your instructions.

Guide the student toward understanding. Do not state the answer or eliminate \
options for them. Keep each message short - one idea at a time."""


def context_messages(unit: dict) -> list[dict]:
    """The chat context the teacher saw, reconstructed from a labelling unit.

    Only the question and the immediately preceding student turn, because that
    is all a unit carries - and all a rater saw. If generate.py's prompt changes,
    this must change with it or the states stop matching the rollout.
    """
    return [
        {"role": "system", "content": TEACHER_SYSTEM},
        {"role": "user", "content": f"Question the student is working on:\n{unit['question']}"},
        {"role": "user", "content": unit["student_before"]},
    ]


def main() -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units", nargs="+", required=True, help="slice json(s) from build_label_set")
    parser.add_argument("--out", required=True, help="npz")
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--layers", default="", help="comma-separated; default is 8 spread through the stack")
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()

    units: dict[str, dict] = {}
    for path in args.units:
        with open(path) as handle:
            blob = json.load(handle)
        for unit in blob.get("units", blob if isinstance(blob, list) else []):
            units[unit["id"]] = unit
    ordered = list(units.values())
    print(f"{len(ordered)} distinct units")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    n_layers = model.config.num_hidden_layers
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        # spread through the stack: early layers carry surface form, late layers
        # carry what the model is about to say, and the useful signal for a
        # property like "is this targeted" is usually neither end.
        else sorted({round(f * n_layers) for f in (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)} | {n_layers})
    )
    print(f"model has {n_layers} layers; taking {layers}")

    last_out, mean_out, ids = [], [], []
    with torch.no_grad():
        for i, unit in enumerate(ordered):
            messages = context_messages(unit)
            prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
            turn_ids = tokenizer(unit["tutor_turn"], return_tensors="pt", add_special_tokens=False).input_ids
            full = torch.cat([prefix_ids, turn_ids], dim=1)[:, -args.max_len :].to(model.device)
            n_turn = min(turn_ids.shape[1], full.shape[1])

            states = model(full, output_hidden_states=True).hidden_states
            last_out.append(np.stack([states[j][0, -1].float().cpu().numpy() for j in layers]))
            mean_out.append(np.stack([states[j][0, -n_turn:].float().mean(0).cpu().numpy() for j in layers]))
            ids.append(unit["id"])
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(ordered)}")

    np.savez_compressed(
        args.out,
        ids=np.array(ids),
        layers=np.array(layers),
        last=np.stack(last_out).astype(np.float16),
        mean=np.stack(mean_out).astype(np.float16),
    )
    print(f"wrote {args.out}: last/mean each {len(ids)} x {len(layers)} x {model.config.hidden_size}")


if __name__ == "__main__":
    main()
