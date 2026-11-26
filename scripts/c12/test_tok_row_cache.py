"""Check scripts/c12/tok_row_cache_patch.py: the row cache must not change one output byte.

    uv run --no-project --python 3.12 --with numpy==2.2.6 --with pyarrow==23.0.0 --with datasets==4.8.5 \
        --with transformers==5.4.0 --with tokenizers==0.22.2 --with torch==2.11.0 --with jinja2==3.1.6 --with rich \
        python scripts/c12/test_tok_row_cache.py

(the image's locked versions; torch and rich are imported by dataset_transformation.py). Takes open_instruct/ at
37a119ffd from this repo's git history, applies the launcher's five tokenize patches in its order (sed fix,
tok_stats_patch.py, tok_count_patch.py, tok_collect_patch.py, tok_row_cache_patch.py) and compiles the results. The
"original" tree has the first four, the "patched" tree all five; only open_instruct/utils.py (ray, beaker, wandb) is
replaced by a stub. Each run is a subprocess that calls the real convert_hf_to_numpy_sft with the real tokenizer
(allenai/dolma2-tokenizer-olmo35 at 8b97170, its olmo35 chat template, <think> and </think> promoted) on synthetic
parquet mixes, in a fresh Hugging Face datasets cache, with BEAKER_ASSIGNED_CPU_COUNT=8 and fork, as on Linux.

Every patched run is compared with an original run of the same mix and arguments: every file of the numpy cache
(statistics without the timestamp and directory), every file of the saved transformed dataset, and every Arrow file
the original left in its Hugging Face cache (the parquet conversion, the dataset_source map, the tokenize map's
output shards and the filter's indices), byte for byte. Exits non-zero on any mismatch.
"""

import json
import os
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_REV = "37a119ffd"
TOKENIZER = "allenai/dolma2-tokenizer-olmo35"
TOKENIZER_REV = "8b9717061fae09d5be814373d189919a62a9a00d"
SED_OLD = "train_dataset_iter = train_dataset.select(range(min(start_idx, total_samples), total_samples))"
SED_NEW = SED_OLD + " if start_idx < total_samples else train_dataset.select([])"
FAILS: list[str] = []
TIMES: dict[str, dict] = {}

UTILS_STUB = '''"""Test stub for open_instruct/utils.py (the real one imports ray, beaker and wandb)."""
import os


def max_num_processes() -> int:
    return 4


def hf_whoami():
    return {"name": "test"}


def maybe_update_beaker_description(**kwargs):
    return None
'''

RUNNER = r'''
import json, os, pathlib, sys, time
args = json.loads(sys.argv[1])
sys.path.insert(0, args["tree"])
import multiprocess
multiprocess.set_start_method("fork", force=True)
from open_instruct import dataset_transformation as dt, numpy_dataset_conversion as ndc

T = {"get_dataset_v1": 0.0}
real = dt.get_dataset_v1
def timed(dc, tc):
    t = time.perf_counter()
    try:
        return real(dc, tc)
    finally:
        T["get_dataset_v1"] += time.perf_counter() - t
dt.get_dataset_v1 = timed

if args.get("identity"):
    from open_instruct import token_row_cache as trc
    tc = dt.TokenizerConfig(tokenizer_name_or_path=args["tokenizer"], tokenizer_revision=args["revision"],
                            chat_template_name=args["template"], reserved_slot_tokens=args["slots"])
    fn_name = args.get("fn", "sft_tulu_tokenize_and_truncate_v1")
    fn = dt.TRANSFORM_FNS[fn_name][0]
    kw = {"tokenizer": tc.tokenizer, **dt.sft_tokenize_fn_args(args["max_seq_length"], args.get("ols", "keep"))}
    ident = trc.compute_identity(fn, fn_name, kw, tc, args.get("cache_version", dt.DATASET_CACHE_VERSION))
    print("RUNNER-IDENTITY " + trc.identity_digest(ident))
    sys.exit(0)

tc = dt.TokenizerConfig(tokenizer_name_or_path=args["tokenizer"], tokenizer_revision=args["revision"],
                        chat_template_name=args["template"], reserved_slot_tokens=args["slots"])
t0 = time.perf_counter()
ndc.convert_hf_to_numpy_sft(
    output_dir=pathlib.Path(args["out"]), dataset_mixer_list=args["mix"], dataset_mixer_list_splits=["train"], tc=tc,
    dataset_transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
    transform_fn_args=[dt.sft_tokenize_fn_args(args["max_seq_length"], "keep"), {}],
    dataset_target_columns=list(dt.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE), max_seq_length=args["max_seq_length"],
    dataset_local_cache_dir=args["local_cache"], shuffle_seed=33333, resume=True, visualize=False,
    row_aligned_parts=True)
T["total"] = time.perf_counter() - t0
print("RUNNER-RESULT " + json.dumps(T))
'''


# ---------------------------------------------------------------- sources


def build_sources(work: pathlib.Path) -> dict[str, pathlib.Path]:
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE_REV, "open_instruct"], check=True,
                             capture_output=True).stdout
    trees = {}
    base = work / "base"
    base.mkdir()
    subprocess.run(["tar", "-x", "-C", str(base)], input=archive, check=True)
    orig = work / "orig"
    shutil.copytree(base, orig)
    f = orig / "open_instruct/numpy_dataset_conversion.py"
    s = f.read_text()
    assert s.count(SED_OLD) == 1
    f.write_text(s.replace(SED_OLD, SED_NEW))
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], check=True, capture_output=True, text=True).stdout
    assert "PATCHED" in run(HERE / "tok_stats_patch.py", f)
    assert "COUNT-PATCHED" in run(HERE / "tok_count_patch.py", orig)
    assert "COLLECT-PATCHED" in run(HERE / "tok_collect_patch.py", f)
    patched = work / "patched"
    shutil.copytree(orig, patched)
    assert "ROWCACHE-PATCHED" in run(HERE / "tok_row_cache_patch.py", patched)
    # The row cache patch refuses a second application and applies to unpatched 37a119ffd too (it touches no line
    # of the other patches).
    again = subprocess.run([sys.executable, str(HERE / "tok_row_cache_patch.py"), str(patched)], capture_output=True)
    assert again.returncode != 0, "row cache patch applied twice"
    alone = work / "rowcache_only"
    shutil.copytree(base, alone)
    run(HERE / "tok_row_cache_patch.py", alone)
    for tree in (orig, patched, alone):
        for p in (tree / "open_instruct").glob("*.py"):
            py_compile.compile(str(p), doraise=True)
    dtp = (patched / "open_instruct/dataset_transformation.py").read_text()
    assert "token_row_cache.cached_map(" in dtp and "dataset_statistics as token_statistics" in dtp
    assert (patched / "open_instruct/token_row_cache.py").read_bytes() == (HERE / "tok_row_cache_patch.py").read_bytes()
    nd = (patched / "open_instruct/numpy_dataset_conversion.py").read_text()
    assert SED_NEW in nd and "_chunked_segment_sums(labels_mm, starts)" in nd and "_prepare_arrow_batch(" in nd
    for tree in (orig, patched):
        (tree / "open_instruct/utils.py").write_text(UTILS_STUB)
    print("sources: sed + stats + count + collect + row cache patches applied in launcher order; all compile")
    trees["orig"], trees["patched"] = orig, patched
    return trees


# ---------------------------------------------------------------- synthetic data


WORDS = ("the of and to in is that for it as with was on be by this are or from at which an not have but "
         "model token data answer think step proof value function return list error result because therefore "
         "x y z n k sum prime integer matrix vector python code test case 0 1 2 3 4 5 6 7 8 9 = + - * / ( ) { } "
         "naïve café 東京 数据 — … émoji 🙂 <think> </think> <|im_end|> \\n").split()


def text(rng, n_words):
    w = rng.choice(len(WORDS), n_words)
    out = " ".join(WORDS[i] for i in w)
    if rng.random() < 0.2:
        out = out.replace(" . ", ".\n\n")
    return out


def length(rng, mean):
    return max(1, int(rng.lognormal(np.log(mean), 0.8)))


def dolci_rows(rng, n, mean_words):
    """Reasoning chats: role, content, reasoning_content (nullable)."""
    rows = []
    for i in range(n):
        msgs = []
        if rng.random() < 0.3:
            msgs.append({"role": "system", "content": text(rng, length(rng, 30)), "reasoning_content": None})
        for t in range(1 if rng.random() < 0.8 else int(rng.integers(2, 4))):
            msgs.append({"role": "user", "content": text(rng, length(rng, mean_words // 4)), "reasoning_content": None})
            if rng.random() < 0.5:
                msgs.append({"role": "assistant", "content": text(rng, length(rng, mean_words // 2)),
                             "reasoning_content": text(rng, length(rng, mean_words))})
            else:
                msgs.append({"role": "assistant", "reasoning_content": None,
                             "content": "<think>" + text(rng, length(rng, mean_words)) + "</think>"
                             + text(rng, length(rng, mean_words // 2))})
        rows.append({"id": f"dolci-{i}", "messages": msgs})
    return rows


def chat_rows(rng, n, mean_words):
    """Plain chats: role and content only, a different struct type from the Dolci rows."""
    rows = []
    for i in range(n):
        msgs = []
        for t in range(int(rng.integers(1, 4))):
            msgs.append({"role": "user", "content": text(rng, length(rng, mean_words // 3))})
            msgs.append({"role": "assistant", "content": text(rng, length(rng, mean_words))})
        rows.append({"id": f"chat-{i}", "messages": msgs})
    return rows


def tool_rows(rng, n, mean_words):
    """Tool calls with JSON-string arguments, environment turns and a tools column (a JSON string or null)."""
    rows = []
    schema = json.dumps([{"type": "function", "function": {"name": "search", "description": "Search.",
                          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
                         {"type": "function", "function": {"name": "calc", "parameters": {"type": "object"}}}])
    for i in range(n):
        msgs = [{"role": "user", "content": text(rng, length(rng, mean_words // 3)), "tool_calls": None}]
        for t in range(int(rng.integers(1, 3))):
            args = {"q": text(rng, 5), "k": int(rng.integers(1, 9))} if rng.random() < 0.5 else {"k": 3, "q": "x y"}
            msgs.append({"role": "assistant", "content": text(rng, length(rng, 20)),
                         "tool_calls": [{"function": {"name": "search", "arguments": json.dumps(args)}}]})
            msgs.append({"role": "environment", "content": text(rng, length(rng, mean_words // 2)), "tool_calls": None})
        msgs.append({"role": "assistant", "content": text(rng, length(rng, mean_words)), "tool_calls": None})
        rows.append({"id": f"tool-{i}", "messages": msgs, "tools": schema if rng.random() < 0.8 else None})
    return rows


def add_specials(rng, rows, kind):
    """Rows the filter drops (no assistant turn; a prompt so long that truncation leaves no assistant token), exact
    duplicates, and empty contents."""
    proto = rows[0]["messages"][0]
    blank = {k: None for k in proto}
    n = len(rows)
    for j in range(6):  # no assistant turn: every label is -100, sft_tulu_filter_v1 drops the row
        m = [{**blank, "role": "user", "content": text(rng, 40)}]
        if j % 2:
            m = [{**blank, "role": "system", "content": "sys"}] + m
        rows.insert(int(rng.integers(0, n)), {**{k: None for k in rows[0]}, "id": f"{kind}-nolabel-{j}", "messages": m})
    for j in range(2):  # ~70k tokens of prompt: truncated at 65,536 before the answer, so dropped too
        m = [{**blank, "role": "user", "content": text(rng, 70000)}, {**blank, "role": "assistant", "content": "ok"}]
        rows.insert(int(rng.integers(0, n)), {**{k: None for k in rows[0]}, "id": f"{kind}-long-{j}", "messages": m})
    for j in range(20):  # exact duplicates (tokenized once)
        rows.insert(int(rng.integers(0, n)), dict(rows[int(rng.integers(0, n))]))
    rows[3]["messages"][-1] = {**rows[3]["messages"][-1], "content": ""}
    if "tools" in rows[0]:
        for r in rows:
            r.setdefault("tools", None)
    return rows


def write_parquet(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)


def make_data(data: pathlib.Path, scale: float, words: int = 300):
    rng = np.random.default_rng(20261113)
    data.mkdir()
    n_dolci = int(3000 * scale)
    dolci = add_specials(rng, dolci_rows(rng, n_dolci, words), "dolci")
    chat = add_specials(rng, chat_rows(rng, int(600 * scale), 150), "chat")
    tool = add_specials(rng, tool_rows(rng, int(600 * scale), 120), "tool")
    write_parquet(data / "dolci.parquet", dolci)
    write_parquet(data / "chat.parquet", chat)
    write_parquet(data / "tool.parquet", tool)
    # The swap analogue: the final assistant turn replaced in a quarter of the rows, 3% of rows dropped.
    rng2 = np.random.default_rng(7)
    swapped, changed = [], 0
    for r in dolci:
        if rng2.random() < 0.03:
            continue
        if rng2.random() < 0.25 and r["messages"][-1]["role"] == "assistant":
            m = [dict(x) for x in r["messages"]]
            m[-1]["content"] = "<think>" + text(rng2, length(rng2, words)) + "</think>" + text(rng2, length(rng2, words // 2))
            m[-1]["reasoning_content"] = None
            r = {**r, "messages": m}
            changed += 1
        swapped.append(r)
    write_parquet(data / "dolci-swapped.parquet", swapped)
    print(f"data: dolci {len(dolci):,} rows, chat {len(chat):,}, tool {len(tool):,}; dolci-swapped {len(swapped):,} "
          f"rows with {changed:,} final turns replaced")
    mix_a = [str(data / "dolci.parquet"), "1.0", str(data / "chat.parquet"), "2.5", str(data / "tool.parquet"), "1.0"]
    mix_b = [str(data / "dolci-swapped.parquet"), "1.0"] + mix_a[2:]
    return {"A": mix_a, "B": mix_b, "changed": changed}


# ---------------------------------------------------------------- running and comparing


class Run:
    def __init__(self, name, work, tree, mix, *, max_seq_length=65536, template="olmo35", store=None, reuse_hf=None,
                 slot="main", env=None):
        self.name = name
        self.dir = work / "runs" / name
        self.dir.mkdir(parents=True)
        # Every run uses the same HF_DATASETS_CACHE path (its slot), because datasets fingerprints a loaded dataset
        # by its file paths; after the run the slot is moved to self.hf_cache for the comparison. reuse_hf moves an
        # earlier run's cache back into the slot first.
        self.slot = WORK / "slots" / slot
        self.reuse_hf = reuse_hf
        self.hf_cache = self.dir / "hf"
        self.out = self.dir / "numpy"
        self.local = self.dir / "local"
        self.tree = tree
        # Every run imports open_instruct through the same path, WORK/src (a symlink to the tree): datasets
        # fingerprints the dataset_source lambda by its code object, which includes the file name.
        self.args = {"tree": str(WORK / "src"), "mix": mix, "max_seq_length": max_seq_length, "template": template,
                     "slots": ["<think>", "</think>"], "tokenizer": TOKENIZER, "revision": TOKENIZER_REV,
                     "out": str(self.out), "local_cache": str(self.local)}
        self.env = {**os.environ, "HF_DATASETS_CACHE": str(self.slot), "BEAKER_ASSIGNED_CPU_COUNT": "8",
                    "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES", "TOKENIZERS_PARALLELISM": "false"}
        self.env.pop("TOK_ROW_CACHE", None)
        if store is not None:
            self.env["TOK_ROW_CACHE"] = str(store)
        self.env.update(env or {})
        self.proc = None

    def start(self):
        link = WORK / "src"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(self.tree)
        assert not self.slot.exists(), f"slot {self.slot} in use"
        self.slot.parent.mkdir(exist_ok=True)
        if self.reuse_hf is not None:
            shutil.move(str(self.reuse_hf.hf_cache), str(self.slot))
        self.log = open(self.dir / "log.txt", "w")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen([sys.executable, str(WORK / "runner.py"), json.dumps(self.args)], env=self.env,
                                     stdout=self.log, stderr=subprocess.STDOUT, cwd=str(self.dir))
        return self

    def wait(self):
        rc = self.proc.wait()
        self.wall = time.perf_counter() - self.t0
        shutil.move(str(self.slot), str(self.hf_cache))
        self.log.close()
        self.text = (self.dir / "log.txt").read_text()
        if rc != 0:
            print(self.text[-6000:])
            raise SystemExit(f"run {self.name} failed with {rc}")
        self.result = json.loads(re.findall(r"RUNNER-RESULT (.*)", self.text)[-1])
        self.summaries = [json.loads(x) for x in re.findall(r"TOK-ROW-CACHE SUMMARY (.*)", self.text)]
        TIMES[self.name] = {"wall": self.wall, **self.result}
        return self

    def run(self):
        return self.start().wait()

    def total(self, key):
        return sum(s.get(key, 0) for s in self.summaries)


def files_of(root: pathlib.Path, pattern="*") -> dict[str, bytes]:
    out = {}
    for p in sorted(root.rglob(pattern)):
        if p.is_file() and ".lock" not in p.name and not p.name.startswith("tmp"):
            out[str(p.relative_to(root))] = p.read_bytes()
    return out


def numpy_outputs(out_dir: pathlib.Path) -> dict[str, bytes]:
    res = {}
    for rel, b in files_of(out_dir).items():
        name = pathlib.Path(rel).name
        if name == "dataset_statistics.json":
            d = json.loads(b)
            d.pop("timestamp"), d.pop("output_directory")
            b = json.dumps(d, sort_keys=True).encode()
        elif name == "dataset_statistics.txt":
            b = b"\n".join(x for x in b.split(b"\n") if not x.startswith((b"Generated:", b"Output Directory:")))
        res[rel] = b
    return res


def compare(label, ref: Run, got: Run):
    problems = []
    a, b = numpy_outputs(ref.out), numpy_outputs(got.out)
    if a != b:
        problems.append(f"numpy cache differs in {sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}")
    a, b = files_of(ref.local), files_of(got.local)
    if a != b:
        problems.append(f"transformed dataset differs in {sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))}")
    a = files_of(ref.hf_cache, "*.arrow")
    b = files_of(got.hf_cache, "*.arrow")
    missing = sorted(k for k in a if b.get(k) != a[k])
    if missing:
        problems.append(f"Hugging Face cache files differ or are missing: {missing}")
    n_tok_files = sum(1 for k in a if "cache-" in k)
    if problems:
        FAILS.append(label)
        print(f"FAIL {label}: " + "; ".join(problems))
    else:
        rows = sum(len(v) for k, v in numpy_outputs(ref.out).items() if k.endswith(".npy"))
        print(f"ok   {label} ({len(files_of(ref.out))} numpy files, {len(files_of(ref.local))} transformed-dataset "
              f"files, {len(a)} HF Arrow files incl. {n_tok_files} map outputs; {rows:,} npy bytes identical)")


def expect(label, cond, detail=""):
    if cond:
        print(f"ok   {label}{(' (' + detail + ')') if detail else ''}")
    else:
        FAILS.append(label)
        print(f"FAIL {label} {detail}")


def counts(run: Run) -> str:
    return (f"hits {run.total('hits'):,}, tokenized {run.total('tokenized'):,}, fallbacks {run.total('fallbacks'):,}, "
            f"rows {run.total('rows'):,}")


def shard_files(store: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in store.rglob("shards/*.arrow"))


# ---------------------------------------------------------------- cases


def main():
    global WORK
    keep = os.environ.get("KEEP_WORK")
    tmp = tempfile.mkdtemp(prefix="trc-test-")
    WORK = pathlib.Path(tmp)
    try:
        cases(WORK)
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"work dir kept at {tmp}")
    import datasets, transformers, tokenizers, torch  # noqa: E401

    print(f"datasets {datasets.__version__}, transformers {transformers.__version__}, tokenizers "
          f"{tokenizers.__version__}, torch {torch.__version__}, pyarrow {pa.__version__}, numpy {np.__version__}")
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("ALL PASSED")


def cases(work: pathlib.Path):
    trees = build_sources(work)
    (work / "runner.py").write_text(RUNNER)
    mixes = make_data(work / "data", float(os.environ.get("TEST_SCALE", "1")), int(os.environ.get("TEST_WORDS", "300")))
    A, B = mixes["A"], mixes["B"]
    O, P = trees["orig"], trees["patched"]
    stores = work / "stores"
    stores.mkdir()
    s1 = stores / "s1"

    # Identity: the same arguments give the same digest in two processes; any change gives a new one.
    ident = {}
    for name, extra in (("base", {}), ("base-again", {}), ("seq2048", {"max_seq_length": 2048}),
                        ("terminate", {"ols": "terminate"}), ("template-olmo", {"template": "olmo"}),
                        ("no-slots", {"slots": None}), ("last-turn", {"fn": "last_turn_tulu_tokenize_and_truncate_v1"}),
                        ("cache-v9", {"cache_version": "v9"})):
        args = {"tree": str(P), "identity": True, "tokenizer": TOKENIZER, "revision": TOKENIZER_REV,
                "template": "olmo35", "slots": ["<think>", "</think>"], "max_seq_length": 65536, **extra}
        env = dict(os.environ)
        out = subprocess.run([sys.executable, str(work / "runner.py"), json.dumps(args)], env=env, capture_output=True,
                             text=True)
        if out.returncode:
            print(out.stdout[-3000:], out.stderr[-3000:])
            raise SystemExit("identity run failed")
        ident[name] = re.findall(r"RUNNER-IDENTITY (\w+)", out.stdout)[-1]
    others = [v for k, v in ident.items() if k not in ("base", "base-again")]
    expect("identity is stable across processes and changes with every argument, the template, the slots, the "
           "transform and DATASET_CACHE_VERSION",
           ident["base"] == ident["base-again"] and len(set(others)) == len(others) and ident["base"] not in others,
           f"{len(ident)} digests")

    ref_a = Run("orig-A", work, O, A).run()
    ref_b = Run("orig-B", work, O, B).run()
    if os.environ.get("TEST_BENCH"):  # timing only: original, cold, warm and partial
        cold = Run("cold-A", work, P, A, store=s1).run()
        warm = Run("warm-A", work, P, A, store=s1).run()
        part = Run("partial-B", work, P, B, store=s1).run()
        for label, ref, got in (("cold", ref_a, cold), ("warm", ref_a, warm), ("partial", ref_b, part)):
            compare(f"bench {label}: patched vs original", ref, got)
        report_times(ref_a, cold, warm, part, s1)
        return

    # 1. Cold cache: everything is tokenized once (duplicates once), and the store fills.
    cold = Run("cold-A", work, P, A, store=s1).run()
    compare("cold cache, mix A: patched vs original", ref_a, cold)
    unique_rows = cold.total("tokenized")
    expect("cold cache tokenizes every distinct row once and serves every row", cold.total("hits") == 0
           and cold.total("fallbacks") == 0 and unique_rows < cold.total("rows") and cold.total("served") == cold.total("rows"),
           counts(cold))
    cold_shards = [p.name for p in shard_files(s1)]
    expect("cold cache stored shards", len(cold_shards) > 0, f"{len(cold_shards)} shards")

    # The Hugging Face map cache still short-circuits: a rerun in the same datasets cache loads the map output.
    again = Run("cold-A-same-hf-cache", work, P, A, store=s1, reuse_hf=cold).run()
    compare("rerun in the same Hugging Face cache: patched vs original", ref_a, again)
    expect("rerun in the same Hugging Face cache loads the map output without the row cache",
           not again.summaries and "the Hugging Face map cache already holds" in again.text)

    # 2. Warm cache, identical mix: all hits, nothing tokenized.
    warm = Run("warm-A", work, P, A, store=s1).run()
    compare("warm cache, identical mix A: patched vs original", ref_a, warm)
    expect("warm cache serves every row from the store", warm.total("tokenized") == 0 and warm.total("fallbacks") == 0
           and warm.total("hits") == warm.total("rows"), counts(warm))

    # 3. Partial hits: the swap analogue (a quarter of the Dolci final turns replaced, 3% of rows dropped).
    part = Run("partial-B", work, P, B, store=s1).run()
    compare("partial hits, mix B (swapped final turns): patched vs original", ref_b, part)
    expect("partial mix tokenizes only the rows with a replaced final turn",
           part.total("tokenized") == mixes["changed"] and part.total("fallbacks") == 0, counts(part))

    # 4. Filtered rows behave identically whether tokenized or served (checked by the statistics in every compare).
    stats = json.loads((ref_a.out / "dataset_statistics.json").read_text())
    dropped = sum(d["instances_filtered_during_transformation"] for d in stats["per_dataset_statistics"])
    expect("the filter drops the same rows from served rows (no-assistant and over-length rows)", dropped >= 3 * 8,
           f"{dropped} rows filtered in mix A, identical in the cold, warm and partial runs")

    # 5. A different max_seq_length or chat template is a different identity: zero hits.
    for name, kw in (("seq2048", {"max_seq_length": 2048}), ("template-olmo", {"template": "olmo"})):
        ref = Run(f"orig-A-{name}", work, O, A, **kw).run()
        got = Run(f"patched-A-{name}", work, P, A, store=s1, **kw).run()
        compare(f"{name}, mix A: patched vs original", ref, got)
        expect(f"{name} is a new identity: zero hits", got.total("hits") == 0 and got.total("tokenized") == unique_rows,
               counts(got))

    # 6. Damaged store: bytes flipped inside stored token data (caught by the digest), a truncated shard (unreadable,
    # skipped), and a junk file with a shard name. Output identical; damaged rows are tokenized again.
    s1c = stores / "s1-damaged"
    shutil.copytree(s1, s1c)
    shards = sorted((p for p in shard_files(s1c) if p.name in cold_shards), key=lambda p: -p.stat().st_size)
    flipped = flip_token_bytes(shards[0], rows=5)
    trunc = shards[1]
    trunc.write_bytes(trunc.read_bytes()[: trunc.stat().st_size // 2])
    (shards[0].parent / ("0" * 64 + ".arrow")).write_bytes(b"not an arrow file")
    dam = Run("damaged-A", work, P, A, store=s1c).run()
    compare("damaged store, mix A: patched vs original", ref_a, dam)
    expect("flipped token bytes are caught by the digest and those rows tokenized", flipped > 0 and dam.total("fallbacks") >= flipped,
           f"{flipped} rows damaged; {counts(dam)}")
    expect("the truncated and junk shards are skipped as unreadable",
           dam.total("store_unreadable_shards") >= 2 and dam.total("tokenized") > 0)
    dam2 = Run("damaged-A-again", work, P, A, store=s1c).run()
    compare("damaged store, second run: patched vs original", ref_a, dam2)
    expect("the truncated shard was replaced by the re-tokenized rows; only the flipped rows still fall back",
           dam2.total("tokenized") == 0 and dam2.total("fallbacks") == dam.total("fallbacks"), counts(dam2))

    # Unreachable root and the kill switch: the original map, identical output, nothing written.
    unr = Run("unreachable-A", work, P, A, store=work / "no-such-mount" / "token-row-cache").run()
    compare("unreachable cache root: patched vs original", ref_a, unr)
    expect("unreachable cache root falls back loudly to the original map",
           "SETUP FAILED" in unr.text and not unr.summaries and not (work / "no-such-mount").exists())
    off = Run("off-A", work, P, A, store="off").run()
    compare("TOK_ROW_CACHE=off: patched vs original", ref_a, off)
    expect("TOK_ROW_CACHE=off runs the original map", "disabled by TOK_ROW_CACHE" in off.text and not off.summaries)

    # 7. Two concurrent writers on an empty store, then both mixes warm.
    s2 = stores / "s2"
    ref_c1 = Run("orig-A-slot-c1", work, O, A, slot="c1").run()
    ref_c2 = Run("orig-B-slot-c2", work, O, B, slot="c2").run()
    c1 = Run("concurrent-A", work, P, A, store=s2, slot="c1").start()
    c2 = Run("concurrent-B", work, P, B, store=s2, slot="c2").start()
    c1.wait(), c2.wait()
    compare("concurrent writer 1, mix A: patched vs original", ref_c1, c1)
    compare("concurrent writer 2, mix B: patched vs original", ref_c2, c2)
    expect("the concurrent writers both tokenized the shared rows", c1.total("tokenized") > 0 and c2.total("tokenized") > 0,
           f"A: {counts(c1)}; B: {counts(c2)}")
    leftovers = [p for p in s2.rglob("*") if p.is_file() and (p.name.startswith(("tmp", ".tmp", ".probe", "incoming")))]
    expect("concurrent writers leave no temporary or unsealed files", not leftovers, str(leftovers[:3]))
    wa = Run("concurrent-then-warm-A", work, P, A, store=s2).run()
    wb = Run("concurrent-then-warm-B", work, P, B, store=s2).run()
    compare("after concurrent writers, warm mix A: patched vs original", ref_a, wa)
    compare("after concurrent writers, warm mix B: patched vs original", ref_b, wb)
    expect("after concurrent writers both mixes are all hits", wa.total("tokenized") == 0 and wb.total("tokenized") == 0
           and wa.total("fallbacks") == 0 and wb.total("fallbacks") == 0, f"A: {counts(wa)}; B: {counts(wb)}")

    report_times(ref_a, cold, warm, part, s1)


def report_times(ref_a, cold, warm, part, s1):
    print("\ntimes (get_dataset_v1 = adding dataset_source + tokenize map + filter, summed over the three sources; "
          "total = convert_hf_to_numpy_sft)")
    for name in ("orig-A", "cold-A", "warm-A", "orig-B", "partial-B"):
        t = TIMES[name]
        print(f"  {name:12s} get_dataset_v1 {t['get_dataset_v1']:7.1f} s   total {t['total']:7.1f} s")
    for r in (cold, warm, part):
        ms = sum(s.get("miss_map_s", 0) for s in r.summaries)
        print(f"  {r.name:12s} row cache: setup {sum(s['prep_s'] for s in r.summaries):.1f} s (of which miss "
              f"tokenize map {ms:.1f} s), final map {sum(s['assemble_s'] for s in r.summaries):.1f} s; {counts(r)}")
    st = shard_files(s1)
    print(f"  store s1: {len(st)} shards, {sum(p.stat().st_size for p in st) / 1e6:.1f} MB; map cache files (dataset_source, tokenize, filter) in "
          f"the HF cache of orig-A: {sum(p.stat().st_size for p in ref_a.hf_cache.rglob('cache-*.arrow')) / 1e6:.1f} MB")


def flip_token_bytes(path: pathlib.Path, rows: int) -> int:
    """Overwrite the first 16 token ids of `rows` compact rows inside the file; return how many rows were damaged."""
    import pyarrow.ipc as ipc

    table = ipc.open_stream(pa.memory_map(str(path))).read_all()
    ids = table.column("__trc_ids")
    blob = bytearray(path.read_bytes())
    done = 0
    for r in range(0, table.num_rows, max(1, table.num_rows // (rows * 2))):
        v = np.asarray(ids[r].values, dtype=np.uint32)
        if table.column("__trc_mode")[r].as_py() != 0 or len(v) < 32:
            continue
        pat = v[:16].tobytes()
        at = blob.find(pat)
        if at < 0 or blob.find(pat, at + 1) >= 0:
            continue
        blob[at : at + 64] = (v[:16] ^ np.uint32(1)).tobytes()
        done += 1
        if done == rows:
            break
    path.write_bytes(bytes(blob))
    return done


if __name__ == "__main__":
    main()
