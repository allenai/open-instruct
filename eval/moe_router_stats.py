import argparse
import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import torch

from eval.utils import load_hf_lm_and_tokenizer, dynamic_import_function


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MoE router stats over eval prompts.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--evals", type=str, nargs="+", required=True, help="Eval groups: mmlu_0shot, mmlu_5shot, bbh_direct, bbh_cot, gsm_direct, gsm_cot")
    parser.add_argument("--data_root", type=str, default="/data", help="Base data mount as used by evals")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=2, help="Top-k experts to count per token")
    parser.add_argument("--layers", type=str, default="all", help="Comma-separated layer indices or 'all'")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format")
    parser.add_argument("--hf_revision", type=str, default=None)
    return parser.parse_args()


def as_int_list(spec: str, max_layers: int) -> List[int]:
    if spec == "all":
        return list(range(max_layers))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def collect_router_topk(model, input_ids: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Returns top-k expert indices per token per layer.
    Output shape: [batch, seq_len, top_k, num_layers]
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_router_logits=True)
    router_logits = outputs["router_logits"]  # tuple per layer: [batch, seq_len, num_experts]
    # Stack into [batch, seq, experts, layers]
    stacked = torch.stack([layer.to(input_ids.device) for layer in router_logits], dim=-1)
    weights = torch.softmax(stacked, dim=2)
    topk = torch.topk(weights, k=top_k, dim=2).indices  # [batch, seq, top_k, layers]
    return topk


def update_counts(layer_counters: Dict[int, Counter], topk_indices: torch.Tensor, layer_filter: List[int]):
    # topk_indices: [batch, seq, top_k, layers]
    num_layers = topk_indices.shape[-1]
    layers = layer_filter if layer_filter else list(range(num_layers))
    for layer in layers:
        ids = topk_indices[..., layer].reshape(-1).tolist()
        layer_counters[layer].update(ids)


def load_prompts_for_eval(eval_name: str, data_root: str, use_chat: bool, chat_fn, tokenizer) -> List[str]:
    """
    Mirror prompt construction used by existing eval runners for a minimal subset:
    - mmlu_0shot, mmlu_5shot
    - bbh_direct, bbh_cot
    - gsm_direct, gsm_cot
    Only builds prompts, no evaluation/scoring.
    """
    prompts: List[str] = []

    if eval_name in ("bbh_direct", "bbh_cot"):
        import glob, json, random
        task_files = glob.glob(os.path.join(data_root, "bbh", "*.json"))
        cot_dir = os.path.join(data_root, "cot-prompts")
        cot_prompts: Dict[str, str] = {}
        for path in glob.glob(os.path.join(cot_dir, "*.txt")):
            task = os.path.basename(path).split(".")[0]
            with open(path, "r") as f:
                task_prompt = "".join(f.readlines()[2:])
            if eval_name == "bbh_direct":
                fields = task_prompt.split("\n\n")
                new_fields = []
                for field in fields:
                    if field.startswith("Q:"):
                        assert "So the answer is" in field and "\nA:" in field
                        answer = field.split("So the answer is")[-1].strip()
                        question = field.split("\nA:")[0].strip()
                        new_fields.append(question + "\nA: " + answer)
                    else:
                        new_fields.append(field)
                task_prompt = "\n\n".join(new_fields)
            cot_prompts[task] = task_prompt
        for task_file in task_files:
            task = os.path.basename(task_file).split(".")[0]
            with open(task_file, "r") as f:
                examples = json.load(f)["examples"]
            # match submit_eval default cap
            examples = examples[:40]
            task_prompt = cot_prompts[task]
            if use_chat:
                for ex in examples:
                    base = task_prompt.strip() + "\n\nQ: " + ex["input"]
                    messages = [{"role": "user", "content": base}]
                    prompt = chat_fn(messages, tokenizer, add_bos=False)
                    prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
                    prompts.append(prompt)
            else:
                prompts += [task_prompt.strip() + "\n\nQ: " + ex["input"] + "\nA:" for ex in examples]

    elif eval_name in ("gsm_direct", "gsm_cot"):
        import json, re
        path = os.path.join(data_root, "gsm", "test.jsonl")
        examples = []
        with open(path) as fin:
            for line in fin:
                ex = json.loads(line)
                examples.append({
                    "question": ex["question"],
                    "answer": ex["answer"].split("####")[1].strip()
                })
        # default n_shot=8 in submit_eval, but prompts here only need the question
        from eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS
        demonstrations = []
        if eval_name == "gsm_direct":
            for d in GSM_EXAMPLARS[:8]:
                demonstrations.append("Quesion: " + d["question"] + "\n" + "Answer: " + d["short_answer"])  # typo preserved to match loader
        else:
            for d in GSM_EXAMPLARS[:8]:
                demonstrations.append("Question: " + d["question"] + "\n" + "Answer: " + d["cot_answer"])
        prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        if use_chat:
            for ex in examples[:200]:
                messages = [{"role": "user", "content": prefix + "Question: " + ex["question"].strip()}]
                prompt = chat_fn(messages, tokenizer, add_bos=False)
                prompt += "Answer:" if prompt[-1] in ["\n", " "] else " Answer:"
                prompts.append(prompt)
        else:
            prompts += [prefix + "Question: " + ex["question"].strip() + "\nAnswer:" for ex in examples[:200]]

    elif eval_name in ("mmlu_0shot", "mmlu_5shot"):
        # Build per-subject prompts matching eval/mmlu/run_eval.py logic
        import pandas as pd
        from eval.mmlu.run_eval import format_example, gen_prompt
        ntrain = 0 if eval_name == "mmlu_0shot" else 5
        test_dir = os.path.join(data_root, "mmlu", "test")
        dev_dir = os.path.join(data_root, "mmlu", "dev")
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(test_dir) if f.endswith("_test.csv")])
        for subject in subjects:
            dev_df = pd.read_csv(os.path.join(dev_dir, subject + "_dev.csv"), header=None)[: ntrain]
            test_df = pd.read_csv(os.path.join(test_dir, subject + "_test.csv"), header=None)
            for i in range(test_df.shape[0]):
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, ntrain)
                prompt = train_prompt + prompt_end
                if use_chat:
                    messages = [{"role": "user", "content": prompt}]
                    prompt = chat_fn(messages, tokenizer, add_bos=False)
                    if prompt[-1] in ["\n", " "]:
                        prompt += "The answer is:"
                    else:
                        prompt += " The answer is:"
                prompts.append(prompt)
    else:
        raise ValueError(f"Unsupported eval name: {eval_name}")

    return prompts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        revision=args.hf_revision,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
        use_fast_tokenizer=True,
    )
    model.eval()

    chat_fn = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    index = {}
    for eval_name in args.evals:
        prompts = load_prompts_for_eval(eval_name, args.data_root, args.use_chat_format, chat_fn, tokenizer)
        layer_counters: Dict[int, Counter] = defaultdict(Counter)

        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            tok = tokenizer(batch_prompts, padding="longest", return_tensors="pt")
            input_ids = tok.input_ids
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            topk = collect_router_topk(model, input_ids=input_ids, top_k=args.top_k)
            # Determine layer filter lazily on first batch using returned num layers
            layer_filter = None
            if args.layers != "all":
                layer_filter = as_int_list(args.layers, topk.shape[-1])
            update_counts(layer_counters, topk.cpu(), layer_filter)

        # Persist
        out_path = os.path.join(args.output_dir, f"{eval_name}_router_counts.json")
        with open(out_path, "w") as f:
            json.dump({str(k): dict(v) for k, v in layer_counters.items()}, f)
        index[eval_name] = out_path
        print(f"Wrote router counts for {eval_name} -> {out_path}")

    with open(os.path.join(args.output_dir, "index.json"), "w") as f:
        json.dump(index, f)
    print("Done.")


if __name__ == "__main__":
    main()
