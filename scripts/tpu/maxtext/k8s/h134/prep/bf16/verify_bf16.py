"""H134 B0: re-read the written fp32 upcasts (masters-bf16) and check that each .to(torch.bfloat16) reproduces the rounded
source bits recorded in bf16_manifest.json (the strip fork's bit-check pattern). Writes verify_bf16.json; exit 1 on any miss.
  python verify_bf16.py <masters-bf16 dir> <bf16_manifest.json> <out.json>"""
import hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch
d, man, out = sys.argv[1:4]
M = json.load(open(man))["tensors"]
def one(k):
    u = torch.from_numpy(np.ascontiguousarray(np.load(os.path.join(d, k + ".npy"))))
    b = u.to(torch.bfloat16)
    ok = torch.equal(b.to(torch.float32), u) and hashlib.sha256(b.view(torch.int16).numpy().tobytes()).hexdigest() == M[k]["sha256_bf16"]
    return k, ok
with ProcessPoolExecutor(16) as ex: res = dict(ex.map(one, sorted(M)))
bad = sorted(k for k, v in res.items() if not v)
json.dump({"n": len(res), "bad": bad, "pass": not bad}, open(out, "w"), indent=1)
print(f"verify: {len(res) - len(bad)}/{len(res)} tensors re-read exactly bf16 and match the manifest sha"); sys.exit(1 if bad else 0)
