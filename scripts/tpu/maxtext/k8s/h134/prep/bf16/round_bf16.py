"""H134 B0 bf16 start (ledger-manager 2026-10-26: B1/B2 are bf16-only exports, so all three bases start from bf16).
Rounds each fp32 master (.npy from dump_olmocore_masters.py) to bf16 with round-to-nearest-even (torch .to(torch.bfloat16)),
records the sha256 of the rounded bf16 bytes, and writes the exact fp32 upcast (the MaxText master format) as .npy for
convert_olmocore_to_maxtext.py.   python round_bf16.py <fp32 masters dir> <out dir> <manifest.json>"""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
src, out, man = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
def one(f):
    x = np.load(os.path.join(src, f))
    b = torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)
    sha = hashlib.sha256(b.view(torch.int16).numpy().tobytes()).hexdigest()
    u = b.to(torch.float32).numpy()
    assert u.shape == x.shape and np.array_equal(torch.from_numpy(u).to(torch.bfloat16).view(torch.int16).numpy(), b.view(torch.int16).numpy())
    np.save(os.path.join(out, f), u)
    return f[:-4], {"shape": list(x.shape), "sha256_bf16": sha, "max_abs_round": float(np.abs(u - x).max()), "changed_frac": float((u != x).mean())}
fs = sorted(f for f in os.listdir(src) if f.endswith(".npy"))
with ProcessPoolExecutor(16) as ex: res = dict(ex.map(one, fs))
tot = hashlib.sha256("".join(f"{k}:{v['sha256_bf16']}\n" for k, v in sorted(res.items())).encode()).hexdigest()
json.dump({"method": "torch .to(torch.bfloat16) (round-to-nearest-even) of OLMo-core's fp32 masters; fp32 upcast written for stage 2", "n": len(res), "sha256_of_manifest": tot, "tensors": res}, open(man, "w"), indent=1)
print(f"rounded {len(res)} tensors; manifest sha {tot[:16]}; max |round| {max(v['max_abs_round'] for v in res.values()):.3e}")
