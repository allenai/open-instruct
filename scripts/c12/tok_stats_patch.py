"""Patch open_instruct/numpy_dataset_conversion.py in place so _aggregate_stats sums trainable tokens per sample in
chunks. The original np.add.reduceat(labels_mm, starts, dtype=np.int64) casts the whole label mask to int64 first
(8x its size in RAM); this computes the same per-sample sums (zero-length samples give 0, as the caller enforces)
from chunked int64 prefix sums at the sample boundaries."""
import sys
p = sys.argv[1]
s = open(p).read()
old = "trainable_counts = np.add.reduceat(labels_mm, starts, dtype=np.int64)"
new = "trainable_counts = _chunked_segment_sums(labels_mm, starts)"
assert s.count(old) == 1, "reduceat line not found exactly once"
func = '''

def _chunked_segment_sums(arr, starts, chunk=1 << 27):
    """Per-sample sums of arr over [starts[i], starts[i+1]) (last runs to len(arr)), int64, chunked."""
    n = len(arr)
    bounds = np.append(np.asarray(starts, dtype=np.int64), n)
    prefix = np.zeros(len(bounds), dtype=np.int64)
    order = np.argsort(bounds, kind="stable")
    sorted_b = bounds[order]
    running = 0
    j = 0
    for c in range(0, n + 1, chunk):
        e = min(c + chunk, n)
        cs = np.cumsum(np.asarray(arr[c:e], dtype=np.int64)) if e > c else np.zeros(0, dtype=np.int64)
        while j < len(sorted_b) and sorted_b[j] < e + (1 if e == n else 0):
            b = sorted_b[j]
            prefix[order[j]] = running + (cs[b - c - 1] if b > c else 0)
            j += 1
        if e > c:
            running += int(cs[-1])
    return np.diff(prefix)
'''
s = s.replace(old, new).replace("\ndef _aggregate_stats(", func + "\n\ndef _aggregate_stats(", 1)
open(p, "w").write(s)
print("PATCHED", p)
