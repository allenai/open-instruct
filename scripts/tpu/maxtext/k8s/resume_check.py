"""Check that a resumed TPU SFT run trains on the same batches as the original pass.

total_weights is a function of the batch only, so after a correct resume it must equal the original pass's value at
the same step; losses should agree to bf16 noise because the weights were restored from the same checkpoint.

    python resume_check.py <original host-0 log> <resumed host-0 log>
"""

import re
import sys

PAT = re.compile(r"completed step: (\d+),.*?total_weights: (\d+), loss: ([0-9.]+)")


def steps(path):
    out = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = PAT.search(line)
            if m:
                out[int(m[1])] = (int(m[2]), float(m[3]))
    return out


orig, resumed = steps(sys.argv[1]), steps(sys.argv[2])
common = sorted(set(orig) & set(resumed))
if not common:
    sys.exit(f"no overlapping steps (resumed has {min(resumed, default=None)}..{max(resumed, default=None)})")
w_match = sum(orig[s][0] == resumed[s][0] for s in common)
dl = [abs(orig[s][1] - resumed[s][1]) for s in common]
print(
    f"steps {common[0]}..{common[-1]} ({len(common)}): total_weights equal {w_match}/{len(common)}; "
    f"|dloss| mean {sum(dl) / len(dl):.4f} max {max(dl):.4f}"
)
for s in common[:3]:
    print(f"  step {s}: orig {orig[s]} resumed {resumed[s]}")
sys.exit(0 if w_match == len(common) else 1)
