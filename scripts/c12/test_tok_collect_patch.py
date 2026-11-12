"""Check scripts/c12/tok_collect_patch.py against the 37a119ffd collect loop it replaces.

    uv run --no-project --with numpy --with pyarrow --with datasets python scripts/c12/test_tok_collect_patch.py

Takes open_instruct/ at 37a119ffd from this repo's git history (the image's source), applies the launcher's tokenize
patches in the launcher's order (sed fix, tok_stats_patch.py, tok_count_patch.py, tok_collect_patch.py), compiles the
patched files, then runs convert_hf_to_numpy_sft from the original (sed + stats patch) and the patched module on
synthetic tokenized datasets, with open_instruct's other modules stubbed. Every case asserts that the three partial
files (as _aggregate_stats sees them) and every output file (parts, metadata, statistics minus the timestamp and
directory) are byte-identical to an uninterrupted original run. Ends with a speed comparison of the two collect loops
on a dataset with the qwen-c12 mean row length. Exits non-zero on any mismatch.
"""

import contextlib
import io
import json
import logging
import pathlib
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
import importlib.util

import numpy as np
import pyarrow as pa
import datasets

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_REV = "37a119ffd"
VOCAB = 100278  # dolma2-tokenizer-olmo35 vocab size: token ids are uint32
SED_OLD = "train_dataset_iter = train_dataset.select(range(min(start_idx, total_samples), total_samples))"
SED_NEW = SED_OLD + " if start_idx < total_samples else train_dataset.select([])"


class Interrupted(Exception):
    pass


# ---------------------------------------------------------------- open_instruct stubs


LOG: list[str] = []


class _ListHandler(logging.Handler):
    def emit(self, record):
        LOG.append(record.getMessage())


def _install_stubs():
    pkg = types.ModuleType("open_instruct")
    pkg.__path__ = []
    dt = types.ModuleType("open_instruct.dataset_transformation")
    dt.INPUT_IDS_KEY = "input_ids"
    dt.ATTENTION_MASK_KEY = "attention_mask"
    dt.LABELS_KEY = "labels"
    dt.MASKED_TOKEN_VALUE = -100
    dt.DATASET_ORIGIN_KEY = "dataset_source"
    dt.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE = ["input_ids", "attention_mask", "labels", "dataset_source"]
    dt.TokenizerConfig = object
    dt.CURRENT = None
    dt.get_cached_dataset_tulu_with_statistics = lambda **kw: (dt.CURRENT, {"per_dataset_stats": []})
    lu = types.ModuleType("open_instruct.logger_utils")

    def setup_logger(name):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not lg.handlers:
            lg.addHandler(_ListHandler())
        return lg

    lu.setup_logger = setup_logger
    ut = types.ModuleType("open_instruct.utils")
    ut.maybe_update_beaker_description = lambda **kw: None
    pkg.dataset_transformation, pkg.logger_utils, pkg.utils = dt, lu, ut
    sys.modules.update({"open_instruct": pkg, "open_instruct.dataset_transformation": dt,
                        "open_instruct.logger_utils": lu, "open_instruct.utils": ut})
    return dt


DT = _install_stubs()


class _Tok:
    vocab_size = VOCAB
    bos_token_id = 100257
    pad_token_id = 100277
    eos_token_id = 100257

    def save_pretrained(self, path):
        pathlib.Path(path, "tokenizer_config.json").write_text("{}")


class _TC:
    tokenizer = _Tok()
    tokenizer_name_or_path = "allenai/dolma2-tokenizer-olmo35"
    chat_template_name = "olmo35"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- building the patched sources


def build_sources(work: pathlib.Path) -> dict[str, pathlib.Path]:
    tree = work / "tree"
    tree.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", BASE_REV, "open_instruct"], check=True,
                             capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    f = tree / "open_instruct/numpy_dataset_conversion.py"
    out = {"base": work / "base.py"}
    shutil.copy(f, out["base"])
    # The launcher's sed, as a literal replacement (the pattern has no regex metacharacters that match differently).
    s = f.read_text()
    assert s.count(SED_OLD) == 1
    f.write_text(s.replace(SED_OLD, SED_NEW))
    assert "else train_dataset.select([])" in f.read_text()
    subprocess.run([sys.executable, str(HERE / "tok_stats_patch.py"), str(f)], check=True)
    out["orig"] = work / "orig.py"
    shutil.copy(f, out["orig"])
    subprocess.run([sys.executable, str(HERE / "tok_count_patch.py"), str(tree)], check=True)
    subprocess.run([sys.executable, str(HERE / "tok_collect_patch.py"), str(f)], check=True)
    out["patched"] = work / "patched.py"
    shutil.copy(f, out["patched"])
    for p in (f, tree / "open_instruct/dataset_transformation.py", tree / "open_instruct/dataset_statistics.py"):
        py_compile.compile(str(p), doraise=True)
    t = f.read_text()
    assert "_chunked_segment_sums(labels_mm, starts)" in t and SED_NEW in t and "_prepare_arrow_batch(" in t
    assert 'with_format("numpy")' not in t
    # The collect patch also applies to the unpatched file, so the order of the in-file patches does not matter.
    alone = work / "collect_only.py"
    shutil.copy(out["base"], alone)
    subprocess.run([sys.executable, str(HERE / "tok_collect_patch.py"), str(alone)], check=True, capture_output=True)
    py_compile.compile(str(alone), doraise=True)
    print("sources: sed + stats + count + collect patches applied in launcher order; all compile")
    return out


# ---------------------------------------------------------------- synthetic datasets


SOURCES = ["dolci-qwen", "nemotron-tool", "nemotron-sci", "ifchat"]


def make_dataset(rng, n_rows, mean_len, *, int_type=pa.int64(), n_chunks=4, special=True, bad_attention_row=None):
    lengths = np.minimum(rng.lognormal(np.log(mean_len), 1.0, n_rows).astype(np.int64), 65536)
    if special:
        lengths[rng.choice(n_rows, 12, replace=False)] = 0
        lengths[rng.choice(n_rows, 6, replace=False)] = 65536
        lengths[rng.choice(n_rows, 6, replace=False)] = 1
    offsets = np.zeros(n_rows + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    tokens = rng.integers(0, VOCAB, total, dtype=np.int64)
    tokens[rng.integers(0, max(total, 1), min(50, total))] = VOCAB - 1
    labels = tokens.copy()
    kind = rng.integers(0, 10, n_rows)
    for i in range(n_rows):
        a, b = offsets[i], offsets[i + 1]
        if b == a:
            continue
        if kind[i] == 0:  # nothing trainable
            labels[a:b] = -100
        elif kind[i] == 1:  # everything trainable
            continue
        else:  # a masked prompt plus masked spans inside the answer
            labels[a:a + int(rng.integers(0, b - a + 1))] = -100
            for _ in range(int(rng.integers(0, 4))):
                s = int(rng.integers(a, b))
                labels[s:min(b, s + int(rng.integers(1, 200)))] = -100
    attention = np.ones(total, np.int64)
    if bad_attention_row is not None:
        assert lengths[bad_attention_row] > 0
        attention[offsets[bad_attention_row] + lengths[bad_attention_row] // 2] = 0
    source = np.array(SOURCES)[rng.integers(0, len(SOURCES), n_rows)]

    def lst(values):
        return pa.ListArray.from_arrays(pa.array(offsets.astype(np.int32)), pa.array(values).cast(int_type))

    full = pa.table({"input_ids": lst(tokens), "attention_mask": lst(attention), "labels": lst(labels),
                     "dataset_source": pa.array(source)})
    cuts = np.linspace(0, n_rows, n_chunks + 1).astype(int)
    table = pa.concat_tables([full.slice(cuts[i], cuts[i + 1] - cuts[i]) for i in range(n_chunks)])
    return datasets.Dataset(datasets.table.InMemoryTable(table)), total


# ---------------------------------------------------------------- running a conversion


PARTIALS = ("_tokens.partial.bin", "_labels.partial.bin", "_boundaries.partial.bin")


def run(mod, out_dir: pathlib.Path, ds, *, resume=True, stop_after_flushes=None, stop_at_stats=False,
        collect_rows=None, workers=None, depth=None):
    """Run convert_hf_to_numpy_sft; return (partial files as _aggregate_stats saw them, collect seconds)."""
    DT.CURRENT = ds
    seen = {}
    flushes = [0]
    real_flush, real_stats = mod._flush_partial_files, mod._aggregate_stats
    knobs = {k: v for k, v in (("_COLLECT_BATCH_ROWS", collect_rows), ("_COLLECT_WORKERS", workers),
                                ("_COLLECT_DEPTH", depth)) if v is not None}
    saved = {k: getattr(mod, k) for k in knobs}

    def flush(*fhs):
        real_flush(*fhs)
        flushes[0] += 1
        if stop_after_flushes is not None and flushes[0] == stop_after_flushes:
            raise Interrupted

    def stats(**kw):
        for name in PARTIALS:
            seen[name] = (out_dir / name).read_bytes()
        if stop_at_stats:
            raise Interrupted
        return real_stats(**kw)

    mod._flush_partial_files, mod._aggregate_stats = flush, stats
    for k, v in knobs.items():
        setattr(mod, k, v)
    LOG.clear()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.convert_hf_to_numpy_sft(
                output_dir=out_dir, dataset_mixer_list=["mix", "1.0"], dataset_mixer_list_splits=["train"], tc=_TC(),
                dataset_transform_fn=[], transform_fn_args=[], dataset_target_columns=DT.TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE,
                max_seq_length=65536, shuffle_seed=33333, resume=resume, row_aligned_parts=True)
    finally:
        mod._flush_partial_files, mod._aggregate_stats = real_flush, real_stats
        for k, v in saved.items():
            setattr(mod, k, v)
    secs = [float(m.group(1)) for m in (re.search(r"samples in ([\d.]+)s", x) for x in LOG) if m]
    return seen, (secs[0] if secs else None)


def outputs(out_dir: pathlib.Path) -> dict[str, bytes]:
    """Every file the conversion leaves, with the run-specific timestamp and directory removed from the statistics."""
    res = {}
    for p in sorted(out_dir.rglob("*")):
        if not p.is_file():
            continue
        b = p.read_bytes()
        if p.name == "dataset_statistics.json":
            d = json.loads(b)
            d.pop("timestamp"), d.pop("output_directory")
            b = json.dumps(d, sort_keys=True).encode()
        elif p.name == "dataset_statistics.txt":
            b = b"\n".join(x for x in b.split(b"\n") if not x.startswith((b"Generated:", b"Output Directory:")))
        res[str(p.relative_to(out_dir))] = b
    return res


def interrupted_dir(mod, work, name, ds, flushes, *, torn=None) -> pathlib.Path:
    d = work / name
    try:
        run(mod, d, ds, stop_after_flushes=flushes)
        raise AssertionError("run was not interrupted")
    except Interrupted:
        pass
    if torn == "tail":  # a write cut off after the last flush: half a row of tokens and labels, half a pair
        with open(d / PARTIALS[0], "ab") as f:
            f.write(b"\x07" * 4 * 333)
        with open(d / PARTIALS[1], "ab") as f:
            f.write(b"\x01" * 333)
        with open(d / PARTIALS[2], "ab") as f:
            f.write(b"\x05" * 8)
    elif torn == "lag":  # boundaries behind the data files: drop the last 37 pairs and 3 bytes
        p = d / PARTIALS[2]
        p.write_bytes(p.read_bytes()[: -(37 * 16 + 3)])
    return d


def copy_dir(src: pathlib.Path, dst: pathlib.Path) -> pathlib.Path:
    shutil.copytree(src, dst)
    return dst


FAILS = []


def check(label, got_partials, got_out, ref_partials, ref_out):
    ok = got_partials == ref_partials and got_out == ref_out
    if not ok:
        bad = [k for k in ref_partials if got_partials.get(k) != ref_partials[k]]
        bad += [k for k in set(ref_out) | set(got_out) if got_out.get(k) != ref_out.get(k)]
        FAILS.append(label)
        print(f"FAIL {label}: differs in {sorted(bad)}")
    else:
        n = sum(len(v) for v in ref_partials.values())
        print(f"ok   {label} ({len(ref_out)} output files, {n:,} partial bytes identical)")


def correctness(mods, work: pathlib.Path):
    orig, pat, base = mods["orig"], mods["patched"], mods["base"]
    rng = np.random.default_rng(1234)
    for int_type in (pa.int64(), pa.int32()):
        tag = f"[{int_type}]"
        ds, total = make_dataset(rng, 5000, 300, int_type=int_type)
        print(f"dataset {tag}: {len(ds):,} rows, {total:,} tokens, {ds.data.column('input_ids').num_chunks} chunks")
        w = work / str(int_type)
        ref_p, _ = run(orig, w / "ref", ds)
        ref_o = outputs(w / "ref")
        assert len(ref_p[PARTIALS[0]]) == 4 * total and len(ref_p[PARTIALS[2]]) == 16 * len(ds)

        p, _ = run(pat, w / "pat", ds)
        check(f"{tag} uninterrupted: patched vs original", p, outputs(w / "pat"), ref_p, ref_o)
        for rows, workers, depth in ((1, 3, 5), (7, 1, 1), (999, 2, 3), (4096, 16, 32)):
            d = w / f"pat-rows{rows}"
            p, _ = run(pat, d, ds, collect_rows=rows, workers=workers, depth=depth)
            check(f"{tag} uninterrupted, patched batches of {rows} rows, {workers} workers, {depth} ahead", p,
                  outputs(d), ref_p, ref_o)

        # Original interrupted after 2 of 5 batches (2,000 rows), with a torn tail; resumed by each version.
        src = interrupted_dir(orig, w, "int-orig", ds, 2, torn="tail")
        for name, mod in (("original", orig), ("patched", pat)):
            d = copy_dir(src, w / f"int-orig-res-{name}")
            p, _ = run(mod, d, ds)
            check(f"{tag} original interrupted at 2,000 rows + torn tail, resumed by {name}", p, outputs(d), ref_p, ref_o)
        d = copy_dir(interrupted_dir(orig, w, "lag-orig", ds, 3, torn="lag"), w / "lag-orig-res")
        p, _ = run(pat, d, ds)
        check(f"{tag} original interrupted at 3,000 rows, boundaries lagging, resumed by patched", p, outputs(d),
              ref_p, ref_o)

        # Patched interrupted after its first 1,024-row batch, and after 3 batches of 7 rows; resumed by each version.
        src = interrupted_dir(pat, w, "int-pat", ds, 1, torn="tail")
        for name, mod in (("original", orig), ("patched", pat)):
            d = copy_dir(src, w / f"int-pat-res-{name}")
            p, _ = run(mod, d, ds)
            check(f"{tag} patched interrupted at 1,024 rows + torn tail, resumed by {name}", p, outputs(d), ref_p, ref_o)
        d = w / "int-pat7"
        try:
            run(pat, d, ds, stop_after_flushes=3, collect_rows=7)
        except Interrupted:
            pass
        assert (d / PARTIALS[2]).stat().st_size == 21 * 16
        p, _ = run(pat, d, ds)
        check(f"{tag} patched interrupted at 21 rows, resumed by patched", p, outputs(d), ref_p, ref_o)

        # All rows collected, then preempted in the statistics stage; resumed by each version.
        src = w / "all-orig"
        try:
            run(orig, src, ds, stop_at_stats=True)
        except Interrupted:
            pass
        for name, mod in (("original", orig), ("patched", pat)):
            d = copy_dir(src, w / f"all-res-{name}")
            p, _ = run(mod, d, ds)
            check(f"{tag} all rows collected, resumed by {name}", p, outputs(d), ref_p, ref_o)
        d = copy_dir(src, w / "all-res-base")
        try:
            run(base, d, ds)
            print(f"info {tag} unpatched 37a119ffd resumes the all-collected case")
        except Exception as e:  # noqa: BLE001
            print(f"info {tag} unpatched 37a119ffd fails the all-collected resume ({type(e).__name__}); the sed fix is needed")

    # A row whose attention mask has a 0: both raise AssertionError with the same rows on disk.
    ds, _ = make_dataset(rng, 3000, 300, special=False, bad_attention_row=2500)
    shuffled_pos = int(np.flatnonzero(np.array(ds.shuffle(seed=33333)._indices.column(0)) == 2500)[0])
    left = {}
    for name, mod in (("original", orig), ("patched", pat)):
        d = work / f"badattn-{name}"
        try:
            run(mod, d, ds, resume=False)
            raise SystemExit(f"{name} accepted a bad attention mask")
        except AssertionError as e:
            assert "Expected all attention mask values to be 1" in str(e)
        left[name] = {n: (d / n).read_bytes() for n in PARTIALS}
    rows_left = len(left["original"][PARTIALS[2]]) // 16
    check(f"bad attention mask at shuffled row {shuffled_pos}: same partial files after the failure "
          f"({rows_left} rows kept)", left["patched"], {}, left["original"], {})
    assert rows_left == shuffled_pos + 1


def speed(mods, work: pathlib.Path):
    rng = np.random.default_rng(7)
    # qwen-c12: 31.7B tokens over ~4.0M rows, ~7,900 tokens per row.
    ds, total = make_dataset(rng, 8000, 4300, special=False, n_chunks=8)
    ds.save_to_disk(str(work / "bench-ds"))
    ds = datasets.load_from_disk(str(work / "bench-ds"))  # memory-mapped, as the real cache is
    print(f"speed dataset: {len(ds):,} rows, {total:,} tokens ({total / len(ds):,.0f} per row), memory-mapped")
    res = {}
    for name, mod, kw in (("orig", mods["orig"], {}), ("patched-1", mods["patched"], {"workers": 1, "depth": 1}),
                          ("patched", mods["patched"], {})):
        d = work / f"bench-{name}"
        t0 = time.perf_counter()
        p, secs = run(mod, d, ds, resume=False, **kw)
        res[name] = (secs, p)
        print(f"  {name:9s} collect {secs:7.2f} s  {len(ds) / secs:9,.0f} rows/s  {total / secs / 1e6:7.1f} M tokens/s"
              f"  (whole conversion {time.perf_counter() - t0:.1f} s)")
    if not res["orig"][1] == res["patched"][1] == res["patched-1"][1]:
        FAILS.append("speed dataset partial files differ")
        print("FAIL speed dataset partial files differ")
    else:
        print(f"  partial files identical; collect speedup {res['orig'][0] / res['patched'][0]:.1f}x "
              f"({res['orig'][0] / res['patched-1'][0]:.1f}x on one thread)")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        src = build_sources(work)
        mods = {k: load_module(src[k], f"nconv_{k}") for k in ("base", "orig", "patched")}
        correctness(mods, work / "c")
        speed(mods, work / "s")
    print(f"datasets {datasets.__version__}, pyarrow {pa.__version__}, numpy {np.__version__}")
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
