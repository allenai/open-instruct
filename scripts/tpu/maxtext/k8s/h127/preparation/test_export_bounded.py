"""Compare bounded export with the established exporter for both checkpoint layouts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
import torch
from export_bounded import tensor_hashes
from orbax import checkpoint as ocp
from safetensors.torch import save_file


def main():
    root = Path("/tmp/h127-export-test")
    root.mkdir(exist_ok=False)
    rng = np.random.default_rng(17)

    def value(*shape):
        return rng.normal(size=shape).astype(np.float32)

    def dense(x, y):
        return {"kernel": value(x, y)}

    def layer(index):
        lp = {name: {"scale": value(4)} for name in ("attn_in_norm", "attn_out_norm", "ffn_in_norm", "ffn_out_norm")}
        if index in (7, 15):
            lp["mixer"] = {
                "attention": {
                    "query": {"kernel": value(4, 2, 4)},
                    "key": {"kernel": value(4, 2, 2)},
                    "value": {"kernel": value(4, 2, 2)},
                    "out": {"kernel": value(2, 2, 4)},
                    "query_norm": {"scale": value(2, 2)},
                    "key_norm": {"scale": value(2, 2)},
                    "ssmax_scale": value(2),
                }
            }
        else:
            lp["mixer"] = {
                "A_log": value(2),
                "dt_bias": value(4),
                "w_q": dense(4, 4),
                "w_k": dense(4, 4),
                "w_v": dense(4, 8),
                "f_proj_1": dense(4, 2),
                "f_proj_2": dense(2, 4),
                "w_b": dense(4, 2),
                "g_proj_1": dense(4, 4),
                "g_proj_2": {"kernel": value(4, 8), "bias": value(8)},
                "w_out": dense(8, 4),
                "q_conv": value(4, 4),
                "k_conv": value(4, 4),
                "v_conv": value(8, 4),
                "o_norm": {"scale": value(4)},
            }
        width = 8 if index == 0 else 4
        lp["shared_ffn"] = {"wi_0": dense(4, width), "wi_1": dense(4, width), "wo": dense(width, 4)}
        if index:
            lp.update(
                latent_down=dense(4, 2),
                latent_up=dense(2, 4),
                moe_block={"gate": dense(4, 2), "wi_0": value(2, 2, 4), "wi_1": value(2, 2, 4), "wo": value(2, 4, 2)},
            )
        return lp

    tree = {
        "token_embedder": {"embedding": value(100352, 4)},
        "decoder": {
            "embedding_norm": {"scale": value(4)},
            "decoder_norm": {"scale": value(4)},
            "logits_dense": dense(4, 100352),
            "layers_0": {f"layer_{i}": layer(i) for i in range(8)},
            "scanned_blocks": {
                f"layer_{i}": jax.tree.map(lambda x: np.expand_dims(x, 1), layer(i + 8)) for i in range(8)
            },
        },
    }
    ref = root / "reference"
    ref.mkdir()
    (ref / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": 100352,
                "num_hidden_layers": 16,
                "num_attention_heads": 2,
                "head_dim": 2,
                "layer_types": ["full_attention" if i in (7, 15) else "linear_attention" for i in range(16)],
            }
        )
    )
    source = Path("/work/export_maxtext_to_hf.py").read_text()
    # Bootstrap only the fixture's target shape table; the test then invokes the
    # unchanged established exporter independently on both saved layouts.
    source = source[: source.index("assert set(hf) == set(target)")]
    start = source.index('index_path = os.path.join(ref, "model.safetensors.index.json")')
    end = source.index('with open(os.path.join(ref, "config.json"))', start)
    source = source[:start] + "target = {}\n" + source[end:]
    ckpt = ocp.PyTreeCheckpointer()
    layouts = {"base": {"params": {"params": tree}}, "tunix": jax.tree.map(lambda x: {"value": x}, tree)}
    layouts["tunix"]["decoder"]["rngs"] = {
        "count": np.array(3, dtype=np.uint32),
        "key": np.array([0, 1], dtype=np.uint32),
    }
    for name, state in layouts.items():
        params = root / name
        ckpt.save(str(params), state)
        if name == "base":
            original_argv = sys.argv
            sys.argv = ["bootstrap", str(params), str(ref), str(root / "unused")]
            ns = {}
            exec(compile(source, "<fixture shapes>", "exec"), ns)
            sys.argv = original_argv
            save_file(
                {k: torch.zeros(v.shape, dtype=torch.bfloat16) for k, v in ns["hf"].items()},
                str(ref / "model.safetensors"),
            )
            (ref / "reference-shapes.json").write_text(json.dumps({k: list(v.shape) for k, v in ns["hf"].items()}))
        original, bounded = root / f"original-{name}", root / f"bounded-{name}"
        subprocess.run(
            [sys.executable, "/work/export_maxtext_to_hf.py", str(params), str(ref), str(original)], check=True
        )
        (ref / "expected-seeded-bf16-sha256.json").write_text(
            json.dumps(tensor_hashes(original / "model.safetensors"))
        )
        subprocess.run(
            [
                sys.executable,
                "/work/export_bounded.py",
                str(params),
                str(ref),
                str(bounded),
                "/tokenizer",
                "--check-base",
            ],
            check=True,
        )
        hashes = [hashlib.sha256((p / "model.safetensors").read_bytes()).hexdigest() for p in (original, bounded)]
        assert hashes[0] == hashes[1], (name, hashes)
        print("EXPORT-EQUIVALENT", name, hashes[0], flush=True)


if __name__ == "__main__":
    main()
