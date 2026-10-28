"""Extract frozen question/answer representations with the starting SGLang policy.

This pilot uses final reference answers, not worked solutions. Training rows are
never rewritten. Each output is keyed by the prepared prompt identity and format.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sglang
from miles.utils import processing_utils
from olmo_sglang import register


def identity(row):
    meta = row["metadata"]
    return f"{meta['prepared_sample_id']}:{meta['prompt_format']}"


def representation_input(row, tokenizer):
    prefix = tokenizer.encode(row["metadata"]["query"] + "\n\n", add_special_tokens=False)
    answer = tokenizer.encode(str(row["label"]), add_special_tokens=False)
    if not prefix or not answer:
        raise ValueError("Empty question or answer")
    return prefix + answer, len(prefix)


def unpack(output, ids, start):
    meta = output["meta_info"]
    hidden = np.asarray(meta["hidden_states"], dtype=np.float32).reshape(
        -1, np.asarray(meta["hidden_states"]).shape[-1]
    )
    if hidden.shape[0] != len(ids) or not np.isfinite(hidden).all():
        raise ValueError(f"Expected {len(ids)} final hidden states, got {hidden.shape}")
    logprobs = meta["input_token_logprobs"]
    # SGLang starts this list at logprob_start_len; requesting zero preserves
    # position zero (null) and permits direct token-by-token alignment checks.
    if len(logprobs) != len(ids) or [x[1] for x in logprobs] != ids:
        raise ValueError("Input log probability alignment mismatch")
    values = np.array([x[0] for x in logprobs[start:]], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite reference answer likelihood")
    return hidden.mean(axis=0, dtype=np.float32), float(-values.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    rows = rows[: args.limit] if args.limit else rows
    if len({identity(r) for r in rows}) != len(rows):
        raise ValueError("Duplicate prepared prompt ids")
    register()
    tokenizer = processing_utils.load_tokenizer(args.model, trust_remote_code=True)
    encoded = [representation_input(row, tokenizer) for row in rows]
    if max(len(ids) for ids, _ in encoded) > 8192:
        raise ValueError("Unexpected input length: do not silently truncate")
    started = time.monotonic()
    engine = sglang.Engine(
        model_path=args.model,
        tokenizer_path=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        tp_size=1,
        context_length=8192,
        mem_fraction_static=0.7,
        attention_backend="triton",
        disable_radix_cache=True,
        mamba_radix_cache_strategy="no_buffer",
        disable_overlap_schedule=True,
        sampling_backend="pytorch",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        chunked_prefill_size=8192,
        max_running_requests=8,
        enable_return_hidden_states=True,
        random_seed=17,
    )
    startup = time.monotonic() - started

    def generate(entries):
        return engine.generate(
            input_ids=[ids for ids, _ in entries],
            sampling_params={"temperature": 0, "max_new_tokens": 1},
            return_hidden_states=True,
            return_logprob=True,
            logprob_start_len=0,
        )

    try:
        # Same prompt alone and in a batch must produce the same representation.
        check = encoded[:2]
        single = [unpack(generate([entry])[0], *entry) for entry in check]
        batched = [unpack(result, *entry) for result, entry in zip(generate(check), check, strict=True)]
        errors = []
        for (a, anll), (b, bnll) in zip(single, batched, strict=True):
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if cosine < 0.999 or abs(anll - bnll) > 0.05:
                raise ValueError(f"Batch consistency failed: cosine={cosine}, NLL delta={anll - bnll}")
            errors.append({"cosine": cosine, "nll_abs_delta": abs(anll - bnll)})
        qualification = {"startup_seconds": startup, "batch_consistency": errors}
        (out / "qualification.json").write_text(json.dumps(qualification, indent=2))
        print(json.dumps({"qualification": qualification}), flush=True)
        vectors, records = [], []
        started = time.monotonic()
        for offset in range(0, len(rows), args.batch_size):
            entries = encoded[offset : offset + args.batch_size]
            results = generate(entries)
            for row, entry, result in zip(rows[offset : offset + args.batch_size], entries, results, strict=True):
                vector, nll = unpack(result, *entry)
                vectors.append(vector)
                records.append(
                    {
                        "prompt_id": identity(row),
                        "row_index": len(records),
                        "difficulty": nll,
                        "input_tokens": len(entry[0]),
                        "answer_tokens": len(entry[0]) - entry[1],
                        "input_ids_sha256": hashlib.sha256(json.dumps(entry[0]).encode()).hexdigest(),
                    }
                )
            if offset % 100 == 0:
                print(json.dumps({"completed": len(records), "seconds": time.monotonic() - started}), flush=True)
        np.save(out / "embeddings.npy", np.stack(vectors))
        (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        manifest = {
            "schema_version": 1,
            "model": args.model,
            "rows": len(records),
            "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
            "representation": "raw query + two newlines + final reference answer; final postnorm layer mean",
            "difficulty": "mean final-answer token NLL conditional on raw query",
            "startup_seconds": startup,
            "extraction_seconds": time.monotonic() - started,
            "batch_size": args.batch_size,
            "qualification": qualification,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
