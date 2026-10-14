"""Score two Qwen3.5 stopping-token candidates on a frozen diagnostic panel.

Run only against an idle teacher owned by this investigation. No generation,
weight changes, or training requests are issued. The panel is selected, not a
representative estimate of model-wide EOS preferences.
"""

import argparse
import hashlib
import json
import math
import urllib.request
from pathlib import Path


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    query = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(query, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.panel.read_bytes()
    rows = json.loads(raw)
    if len(rows) > 12 or sum(len(row["input_ids"]) for row in rows) > 200_000:
        raise ValueError("Diagnostic cap exceeded")
    result = {
        "panel_sha256": hashlib.sha256(raw).hexdigest(),
        "model_info": request(args.url.rstrip("/") + "/get_model_info"),
        "rows": [],
    }
    for row in rows:
        ids = row["input_ids"]
        if len(ids) < 2 or ids[-1] != 248046:
            raise ValueError("Expected a complete response ending with Qwen3.5 im_end")
        response = request(
            args.url.rstrip("/") + "/generate",
            {
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
                "return_logprob": True,
                "logprob_start_len": len(ids) - 2,
                "token_ids_logprob": [248044, 248046],
            },
        )
        metadata = response["meta_info"]
        sampled = metadata["input_token_logprobs"][-1]
        candidates = {int(entry[1]): float(entry[0]) for entry in metadata["input_token_ids_logprobs"][-1]}
        if sampled[1] != 248046 or set(candidates) != {248044, 248046}:
            raise ValueError("Unexpected teacher score alignment")
        if not all(math.isfinite(value) for value in candidates.values()):
            raise ValueError("Nonfinite teacher stop scores")
        result["rows"].append(
            {
                **{key: value for key, value in row.items() if key != "input_ids"},
                "input_tokens": len(ids),
                "teacher_im_end_logprob": candidates[248046],
                "teacher_endoftext_logprob": candidates[248044],
                "teacher_total_two_stop_probability": sum(math.exp(value) for value in candidates.values()),
                "im_end_rescore_minus_saved": candidates[248046] - row["saved_teacher_sampled_logprob"],
            }
        )
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
