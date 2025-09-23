import argparse
import json
import os
import pickle as pkl
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch

from eval.utils import load_hf_lm_and_tokenizer, dynamic_import_function


# Adapted from OLMoE run_routing_analysis.py
def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=8
) -> float:
    """
    Compute load balancing loss from router logits.
    Adapted from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=1)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(-2)) / len(gate_logits)
    return overall_loss * num_experts


class RouterStatsCollector:
    """
    Collects router statistics during model evaluation.
    Compatible with OLMoE analysis format.
    """
    
    def __init__(self, model_name: str = "olmoe", top_k: int = 8, 
                 track_token_expert_mapping: bool = False,
                 selected_layers: Optional[List[int]] = None):
        self.model_name = model_name
        self.top_k = top_k
        self.track_token_expert_mapping = track_token_expert_mapping
        self.selected_layers = selected_layers or [0, 7, 15]  # Default layers for OLMoE
        
        # Core counters
        self.layer_counters: Dict[int, Counter] = defaultdict(Counter)
        self.crosslayer_counters: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
        
        # Token-expert mapping (optional)
        if track_token_expert_mapping:
            self.eid2token_layer0 = defaultdict(Counter)
            self.eid2token_layer7 = defaultdict(Counter)
            self.eid2token_layer15 = defaultdict(Counter)
        
        # Auxiliary metrics
        self.aux_losses = []
        self.total_token_count = 0
        
    def update(self, input_ids: torch.Tensor, router_logits: Tuple[torch.Tensor, ...], 
               model_num_experts: int):
        """
        Update counters with router logits from a forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] - input token ids
            router_logits: tuple of [seq_len, num_experts] per layer
            model_num_experts: number of experts in the model
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert router logits to numpy for processing
        router_logits_np = [l.detach().cpu().numpy() for l in router_logits]
        
        # Get top-k expert assignments per token, per layer
        # Shape: [seq_len, top_k, num_layers]
        exp_ids = np.stack([
            np.argsort(-logits, -1)[:, :self.top_k].tolist() 
            for logits in router_logits_np
        ], -1)
        
        # Update token count
        self.total_token_count += seq_len * batch_size
        
        # Compute auxiliary loss
        aux_loss = load_balancing_loss_func(
            router_logits, model_num_experts, self.top_k
        )
        self.aux_losses.append(aux_loss.cpu().item())
        
        # Update layer counters
        for layer in range(exp_ids.shape[2]):
            if self.selected_layers is None or layer in self.selected_layers:
                exp_counts = Counter(exp_ids[:, :, layer].flatten())
                self.layer_counters[layer].update(exp_counts)
        
        # Update cross-layer counters for selected layer pairs
        for layer_i in range(exp_ids.shape[2] - 1):
            for layer_j in range(layer_i + 1, exp_ids.shape[2]):
                if ((self.selected_layers is None or (layer_i in self.selected_layers and layer_j in self.selected_layers)) and
                    (layer_i, layer_j) in [(0, 7), (7, 15)]):  # Only track specific pairs for compatibility
                    exps_counts = Counter(zip(exp_ids[:, :, layer_i].flatten(), 
                                            exp_ids[:, :, layer_j].flatten()))
                    self.crosslayer_counters[(layer_i, layer_j)].update(exps_counts)
        
        # Update token-expert mapping if enabled
        if self.track_token_expert_mapping:
            input_ids_np = input_ids[0].detach().cpu().numpy().tolist()
            
            for token_idx, experts in enumerate(exp_ids[:, :, 0]):  # Layer 0
                for e in experts:
                    self.eid2token_layer0[e][input_ids_np[token_idx]] += 1
                    
            for token_idx, experts in enumerate(exp_ids[:, :, 7]):  # Layer 7
                for e in experts:
                    self.eid2token_layer7[e][input_ids_np[token_idx]] += 1
                    
            for token_idx, experts in enumerate(exp_ids[:, :, 15]):  # Layer 15
                for e in experts:
                    self.eid2token_layer15[e][input_ids_np[token_idx]] += 1
    
    def save_results(self, output_dir: str, domain: str):
        """
        Save results in OLMoE-compatible format.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save expert counts for selected layers
        expert_counts_path = Path(output_dir) / "expert_counts"
        expert_counts_path.mkdir(parents=True, exist_ok=True)
        
        with open(expert_counts_path / f"{domain}.pkl", "wb") as f:
            pkl.dump([
                self.layer_counters[0], 
                self.layer_counters[7], 
                self.layer_counters[15]
            ], f)
        
        # Save cross-layer counts
        crosslayer_counts_path = Path(output_dir) / "expert_counts_crosslayer"
        crosslayer_counts_path.mkdir(parents=True, exist_ok=True)
        
        with open(crosslayer_counts_path / f"{domain}.pkl", "wb") as f:
            pkl.dump([
                self.crosslayer_counters[(0, 7)], 
                self.crosslayer_counters[(7, 15)]
            ], f)
        
        # Save token-expert mapping if available
        if self.track_token_expert_mapping:
            eid2token_path = Path(output_dir) / "eid2token"
            eid2token_path.mkdir(parents=True, exist_ok=True)
            
            with open(eid2token_path / f"{domain}.pkl", "wb") as f:
                pkl.dump([
                    self.eid2token_layer0, 
                    self.eid2token_layer7, 
                    self.eid2token_layer15
                ], f)
        
        # Save summary statistics
        summary = {
            "total_tokens": self.total_token_count,
            "avg_aux_loss": np.mean(self.aux_losses) if self.aux_losses else 0.0,
            "num_batches": len(self.aux_losses),
            "expert_usage_per_layer": {
                str(layer): dict(counter) for layer, counter in self.layer_counters.items()
            }
        }
        
        with open(Path(output_dir) / f"{domain}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    def print_summary(self):
        """Print summary statistics."""
        print(f"Total tokens processed: {self.total_token_count}")
        print(f"Average auxiliary loss: {np.mean(self.aux_losses) if self.aux_losses else 0.0:.4f}")
        
        for layer in sorted(self.layer_counters.keys()):
            total = sum(self.layer_counters[layer].values())
            print(f"\nLayer {layer} expert usage:")
            for eid, count in self.layer_counters[layer].most_common(10):
                print(f"  Expert {eid}: {count/total*100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MoE router stats over eval prompts.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--evals", type=str, nargs="+", required=True, help="Eval groups: mmlu_0shot, mmlu_5shot, bbh_direct, bbh_cot, gsm_direct, gsm_cot")
    parser.add_argument("--data_root", type=str, default="/data", help="Base data mount as used by evals")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=8, help="Top-k experts to count per token (8 for OLMoE, 2 for Mixtral)")
    parser.add_argument("--layers", type=str, default="0,7,15", help="Comma-separated layer indices or 'all'")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_chat_format", action="store_true")
    parser.add_argument("--chat_formatting_function", type=str, default="eval.templates.create_prompt_with_tulu_chat_format")
    parser.add_argument("--hf_revision", type=str, default=None)
    parser.add_argument("--track_token_expert_mapping", action="store_true", help="Track which tokens are assigned to which experts")
    parser.add_argument("--model_name", type=str, default="olmoe", help="Model name for output directory structure")
    parser.add_argument("--max_tokens", type=int, default=204800, help="Maximum tokens to process per eval")
    return parser.parse_args()


def as_int_list(spec: str, max_layers: int) -> List[int]:
    if spec == "all":
        return list(range(max_layers))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def collect_router_logits(model, input_ids: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Collect router logits from model forward pass.
    Returns tuple of [batch, seq_len, num_experts] per layer.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_router_logits=True)
    return outputs["router_logits"]


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
