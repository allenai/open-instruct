"""o3i7b eval specs: two dense Olmo 3 7B Instruct SFT checkpoints on the H051 size-peer battery (dry run).

Writes specs/<name>.json, one per (model, cell), and prints a table. Launches nothing.

Models (the two specs of a cell are byte-identical except the model path and the names):
  released  allenai/Olmo-3-7B-Instruct-SFT @ e1452fc572d51966ff4aaeb25118b891eb93e549, snapshot on WEKA
  tpu       the MaxText (TPU) reproduction, exported by ../export_hf.sh (does not exist yet)

Like experiments/h051/mkspecs_h051.py (ledger branch emo-vs-noemo-battery), each spec is copied from a working spec
and changes only what it must; the rest is asserted equal to the source. Sources (the H051 sources, except that the
"OFF" thinking-off IF cell replaces the forced-think T25 one, since this model does not think; the two differ only in
provider.tokenizer, which is replaced here anyway):
  OFF  H045 IF strip32k thinking-off cell (olmo-eval 25ae07ac): maths, GPQA, MMLU-Pro (at main 5ef9cee), IF cells
  LCB  H045 lcb-v3-forcedthink-t10 (olmo-eval 0131141, codex_python harness)
  BFCL C4 thinking-off BFCL single-turn (olmo-eval 9538453e55, vllm_server, its own venv)
  RUL  H045 46faa7ee ruler_all__65536 (the model's 65,536 window; H051 ran 131,072)

Deliberate deviations from the sources, beyond H051's own (model, task, registered sampling, names, normal priority,
jupiter only, MMLU-Pro commit + 4 instances, LCB 6h -> 24h timeout):
  D1 Olmo-3.5-only pieces removed. Stock vLLM 0.19.1 serves Olmo3ForCausalLM (models/olmo2.py), so these go:
     provider.dependencies + their GANTRY_INSTALL_CMD steps: flash-linear-attention, ai2-olmo-core ("hero-core"; the
     plugins' runtime), olmoe3-vllm-plugin, olmoe3-transformers-plugin, the flash-attn stub wheel; the kwarg
     mamba_ssm_cache_dtype; the plugin-only env vars OLMO_VLLM_TORCH_GROUPED_MOE and OLMO_VLLM_FLA_KDA. BFCL also loses
     the ladders tar (sha check, extract, test), the hero-core clone, their three `-e` installs, fla, and
     reasoning_parser=olmo3 (a <think> parser; this model emits none).
     Kept on purpose: vllm==0.19.1 and transformers 5.14.1 (5.16.1 for BFCL) as pinned; the olmo-core runtime deps
     (cached-path, ..., pandas: harmless, and dropping them would change the resolve); `rm -rf .../site-packages/flash_attn`
     (removes the image's flash_attn; vLLM 0.19.1 uses its bundled vllm_flash_attn, and with no flash_attn installed
     its rotary find_spec("flash_attn") path stays off); UV_CONSTRAINT (pins only hf-hub 1.16.1 + transformers 5.14.1);
     enforce_eager, prefix caching off, FLASH_ATTN, language_model_only (all valid EngineArgs at 0.19.1).
  D2 provider.tokenizer = the released snapshot, for both models (served with the released chat template/tokenizer).
     Replaces C4's tok-thinkoff dir in OFF/BFCL; added to LCB/RUL. The TPU training tokenizer
     (../tok-olmo3-instruct-dev-fixed) encodes identically (same vocab, merges, pre-tokenizer, added tokens; checked).
  D3 provider.kwargs.hf_overrides = flat YaRN rope_parameters (hf_overrides.json). Required: transformers 5.x nests
     Olmo 3's rope_parameters per layer type, and vLLM 0.19.1's olmo2.py reads rope_parameters["rope_theta"], so
     without it engine init dies with KeyError 'rope_theta' (validate/check_released.py reproduces this on CPU). With
     it, full-attention layers get YaRN (factor 8, attention_factor 1.2079) and sliding layers plain RoPE at theta
     500000. Same override as ~/handoff/xarch-evals/eval_kda_dense.sh, verified there on the dense 7B (gsm8k 0.7604 vs
     in-loop 0.7612).
  D4 provider.kwargs.generation_config = the released snapshot, for both models. export_hf.sh copies
     generation_config.json from the BASE model (allenai/Olmo-3-1025-7B), which has no eos_token_id list and no sampling
     defaults; served as-is the TPU model would stop only on <|endoftext|>, not <|im_end|> (100265), and the BFCL
     server would apply different request defaults. For the released model this is a no-op (vLLM loads the same file).
     validate/check_engine_args.py shows both models then resolve to eos [100265, 100257] and the same defaults.
  D5 BFCL tool_call_parser qwen3_xml -> olmo3 (vLLM 0.19.1 tool_parsers/olmo3_tool_parser.py,
     Olmo3PythonicToolParser: reads <function_calls>name(k=v)\\n...</function_calls>, which is what the released
     template asks for). olmo-eval's patch_olmo3_tool_parser is NOT used: it also swaps in olmo-eval's own chat
     template, which would break D2.
  D6 RULER provider.max_model_len stays 65536 (the source's), not H051's 131072; the RULER source's urgent/saturn
     becomes normal/jupiter as in H051; its minRuntime 0s / autoResume true (backfill) is kept as in H051's RULER cells.
  D7 Gantry-based specs keep the task name "main" (it matches GANTRY_TASK_NAME); H051 renamed it. BFCL's task gets the
     experiment name. ifeval_mt_ood (not an H051 cell; H051 reused existing cells) gets top_p=1.0 like IFEval.
  D8 Not applied: T=1.0 on BFCL and RULER. As in H051, BFCL keeps its task decoding (temperature 0.001, 4096 tokens)
     and RULER the battery's default decoding.
"""

import copy
import json
import pathlib
import sys

import yaml

HERE = pathlib.Path("/weka/oe-adapt-default/abhishekr/tpu-posttrain/evals")
OUT = HERE / "specs"
DATE = "20261007"  # America/Los_Angeles date the specs were generated; goes into the experiment group
REL = str(HERE / "models/olmo3-7b-instruct-sft-released")
MODELS = {"released": REL, "tpu": str(HERE / "models/olmo3-7b-instruct-sft-tpu-s0")}
SRC = {
    "OFF": "/root/handoff/c4/specs/h045-if-strip32k-off-s34521-ifeval_mt_ood-t10-20261005.json",
    "LCB": "/root/handoff/h045/battery-s34521/lcb-v3-forcedthink-t10.yaml",
    "BFCL": "/root/handoff/c4/specs/c4-thinkoff-s34521-bfcl-single-20261005.json",
    "RUL": "/root/handoff/h045/battery-s34521/46faa7ee-ruler_all__65536.yaml",
}
SRC_MODEL = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/hero-sft-4t/runs/h045-anchor-emo-olmo35-s34521/hf_step11768"
C4_TOK = "/weka/oe-adapt-default/abhishekr/handoff/c4/tok-thinkoff"
MAIN = "5ef9cee1cfa4eafd8bb624bd18e2817c74cd54cc"

# D3: flat YaRN, exactly config.json's rope_scaling plus rope_theta.
HF_OVERRIDES = {
    "rope_parameters": {
        "rope_type": "yarn",
        "factor": 8.0,
        "original_max_position_embeddings": 8192,
        "attention_factor": 1.2079441541679836,
        "beta_fast": 32,
        "beta_slow": 1,
        "rope_theta": 500000,
    }
}
HF_OVR_ARG = "provider.kwargs.hf_overrides=" + json.dumps(HF_OVERRIDES, separators=(",", ":"))
ADD = [f"provider.tokenizer={REL}", HF_OVR_ARG, f"provider.kwargs.generation_config={REL}"]  # D2, D3, D4

# D1
DROP_DEPS = (
    "flash-linear-attention",
    "ai2-olmo-core",
    "olmoe3-vllm-plugin",
    "olmoe3-transformers-plugin",
    "flash-attn @",
)
DROP_INSTALL = (
    "flash-linear-attention",
    "ai2-olmo-core",
    "olmoe3-vllm-plugin",
    "olmoe3-transformers-plugin",
    "fa-stub",
)
DROP_KWARGS = ("provider.kwargs.mamba_ssm_cache_dtype=",)
DROP_ENV = ("OLMO_VLLM_TORCH_GROUPED_MOE", "OLMO_VLLM_FLA_KDA")

SAMPLING = ("temperature=", "do_sample=", "max_tokens=", "top_p=", "num_samples=")
REASON = ["temperature=1.0", "do_sample=true", "max_tokens=32768", "top_p=0.95"]
IF = ["temperature=1.0", "do_sample=true", "max_tokens=32768", "top_p=1.0"]
# cell: (source, task, sampling overrides (None = keep the source's), GPUs (None = source's), olmo-eval commit or None)
CELLS = {
    "aime25": ("OFF", "aime_2025:pass_at_32", REASON + ["num_samples=16"], 1, None),
    "aime26": ("OFF", "aime_2026:pass_at_32", REASON + ["num_samples=16"], 1, None),
    "hmmt26": ("OFF", "hmmt_feb_2026:pass_at_32", REASON + ["num_samples=16"], 1, None),
    "gpqa": ("OFF", "gpqa_diamond:cot", REASON + ["num_samples=8"], 1, None),
    "mmlupro": ("OFF", "mmlu_pro:cot", REASON, 4, MAIN),
    "lcb": ("LCB", "livecodebench:lite", ["temperature=1.0", "max_tokens=32768", "num_samples=8"], None, None),
    "ifeval": ("OFF", "ifeval", IF, 1, None),
    "ifbench": ("OFF", "ifeval_ood", ["temperature=1.0", "do_sample=true", "max_tokens=32768"], 1, None),
    "ifeval_mt_ood": ("OFF", "ifeval_mt_ood_wildchat_unused_withRewrite", IF, 1, None),
    "bfcl_single": ("BFCL", None, None, None, None),
    "ruler64k": ("RUL", "ruler_all__65536", None, None, None),
}


def load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())


def env(t, name):
    return next(e for e in t["envVars"] if e["name"] == name)


def opt_index(a, prefix):
    """Index of the value of `-o <prefix>...`, or None."""
    hits = [j for j in range(1, len(a)) if a[j - 1] == "-o" and str(a[j]).startswith(prefix)]
    assert len(hits) <= 1, (prefix, hits)
    return hits[0] if hits else None


def drop_deps(dep_arg):
    head, body = dep_arg.split("=[", 1)
    assert head == "provider.dependencies" and body.endswith("]"), dep_arg
    items = body[:-1].split(",")
    kept = [x for x in items if not x.startswith(DROP_DEPS)]
    assert len(items) - len(kept) == len(DROP_DEPS), (items, kept)
    assert kept[-2:] == ["transformers==5.14.1", "huggingface-hub==1.16.1"], kept  # last-writer-wins order kept
    return "provider.dependencies=[" + ",".join(kept) + "]"


def drop_install(cmd):
    steps = cmd.split(" && ")
    kept = [s for s in steps if not any(m in s for m in DROP_INSTALL)]
    assert len(steps) - len(kept) == len(DROP_INSTALL), [s for s in steps if s not in kept]
    return " && ".join(kept)


def build_olmo_eval(cell, model, name):
    src_key, task, ovr, gpus, commit = CELLS[cell]
    src = load(SRC[src_key])
    spec = copy.deepcopy(src)
    t = spec["tasks"][0]
    a = t["arguments"]
    a[a.index("-m") + 1] = MODELS[model]
    a[a.index("-t") + 1] = task
    a[a.index("--experiment-group") + 1] = f"{name}-{DATE}"
    a[a.index("--experiment-name") + 1] = name
    if ovr is not None:  # as H051: drop the source's sampling overrides, insert the registered ones right after -t
        kept, j = [], 0
        while j < len(a):
            if a[j] == "-o" and str(a[j + 1]).startswith(SAMPLING):
                j += 2
                continue
            kept.append(a[j])
            j += 1
        k = kept.index("-t") + 2
        a[:] = kept[:k] + [x for o in ovr for x in ("-o", o)] + kept[k:]
    for p in DROP_KWARGS:  # D1
        i = opt_index(a, p)
        del a[i - 1 : i + 1]
    i = opt_index(a, "provider.dependencies=")
    a[i] = drop_deps(a[i])
    i = opt_index(a, "provider.tokenizer=")  # D2
    if i is not None:
        assert a[i] == f"provider.tokenizer={C4_TOK}", a[i]
        del a[i - 1 : i + 1]
    a += [x for o in ADD for x in ("-o", o)]  # D2-D4
    if gpus and gpus > 1:
        a[opt_index(a, "provider.num_instances=")] = f"provider.num_instances={gpus}"
        t["resources"]["gpuCount"] = gpus
    if commit:
        env(t, "GIT_REF")["value"] = commit
    t["envVars"] = [e for e in t["envVars"] if e["name"] not in DROP_ENV]
    env(t, "GANTRY_INSTALL_CMD")["value"] = drop_install(env(t, "GANTRY_INSTALL_CMD")["value"])
    if src_key == "LCB":
        t["timeout"] = "24h"  # as H051: the source's 6h was for k=1; k=8 needs longer
    t["context"]["priority"] = "normal"
    t["constraints"]["cluster"] = ["ai2/jupiter"]
    spec["description"] = (
        f"o3i7b cell {cell} on {model} ({MODELS[model]}); copied from {pathlib.Path(SRC[src_key]).name}"
        f"; changes: model, task, registered sampling, names, normal priority, jupiter"
        + (", olmo-eval commit, instances" if commit else "")
        + (", timeout 24h" if src_key == "LCB" else "")
        + "; Olmo-3.5 plugin stack removed, released tokenizer + generation_config, flat YaRN"
        " hf_overrides (see mkspecs_o3i7b.py D1-D8)."
    )
    check_olmo_eval(spec, src)
    return spec


def norm_olmo_eval(spec):
    """Drop every field build_olmo_eval may change, so the rest can be compared with the source."""
    s = copy.deepcopy(spec)
    s.pop("description", None)
    t = s["tasks"][0]
    for k in ("timeout",):
        t.pop(k, None)
    t["resources"].pop("gpuCount", None)
    t["context"].pop("priority", None)
    t["constraints"].pop("cluster", None)
    out, a, i = [], t["arguments"], 0
    may_change = (
        SAMPLING
        + DROP_KWARGS
        + (
            "provider.num_instances=",
            "provider.dependencies=",
            "provider.tokenizer=",
            "provider.kwargs.hf_overrides=",
            "provider.kwargs.generation_config=",
        )
    )
    while i < len(a):
        if a[i] in ("-m", "-t", "--experiment-group", "--experiment-name"):
            i += 2
            continue
        if a[i] == "-o" and str(a[i + 1]).startswith(may_change):
            i += 2
            continue
        out.append(a[i])
        i += 1
    t["arguments"] = out
    t["envVars"] = [e for e in t["envVars"] if e["name"] not in ("GIT_REF", "GANTRY_INSTALL_CMD") + DROP_ENV]
    return s


def check_olmo_eval(spec, src):
    assert norm_olmo_eval(spec) == norm_olmo_eval(src)
    t, a = spec["tasks"][0], spec["tasks"][0]["arguments"]
    s = src["tasks"][0]
    assert a[opt_index(a, "provider.dependencies=")] == drop_deps(
        s["arguments"][opt_index(s["arguments"], "provider.dependencies=")]
    )
    assert env(t, "GANTRY_INSTALL_CMD")["value"] == drop_install(env(s, "GANTRY_INSTALL_CMD")["value"])
    assert "provider.max_model_len=65536" in a and "provider.dtype=bfloat16" in a, a
    joined = json.dumps(spec["tasks"])
    for gone in (
        "olmoe3",
        "fa-stub",
        "flash-linear-attention",
        "ai2-olmo-core",
        "mamba_ssm",
        "tok-thinkoff",
        "reasoning_parser",
        "OLMO_VLLM_FLA_KDA",
        "OLMO_VLLM_TORCH_GROUPED_MOE",
        SRC_MODEL,
    ):
        assert gone not in joined, gone
    assert all(o in a for o in ADD)
    if t["resources"]["gpuCount"] > 1:
        assert f"provider.num_instances={t['resources']['gpuCount']}" in a


# BFCL: one bash script; edit it line by line.
BFCL_DROP_LINES = (
    'echo "12117833',
    "mkdir -p /tmp/hero-ladder",
    "test -d /tmp/hero-ladder",
    "git init --quiet /tmp/hero-core",
    "git -C /tmp/hero-core fetch",
)
BFCL_PIP_DROP = (
    " 'flash-linear-attention==0.5.2'",
    " -e /tmp/hero-core",
    " -e /tmp/hero-ladder/ladders/olmoe3/vllm_plugin",
    " -e /tmp/hero-ladder/ladders/olmoe3/transformers_plugin",
)


def replace_once(s, old, new):
    assert s.count(old) == 1, (old, s.count(old))
    return s.replace(old, new)


def build_bfcl(cell, model, name):
    src = load(SRC["BFCL"])
    spec = copy.deepcopy(src)
    t = spec["tasks"][0]
    (script,) = t["arguments"]
    lines = script.split("\n")
    kept = [ln for ln in lines if not ln.startswith(BFCL_DROP_LINES)]
    assert len(lines) - len(kept) == len(BFCL_DROP_LINES)
    out = []
    for ln in kept:
        if ln.startswith("uv pip install --python /tmp/hero-eval-env/bin/python 'vllm=="):
            for d in BFCL_PIP_DROP:
                ln = replace_once(ln, d, "")
            assert "'vllm==0.19.1' 'transformers==5.16.1'" in ln and "hero" not in ln.replace("hero-eval-env", ""), ln
        elif ln.startswith("olmo-eval run "):
            ln = replace_once(ln, f"-o provider.tokenizer={C4_TOK}", f"-o provider.tokenizer={REL}")  # D2
            ln = replace_once(
                ln, "-o provider.kwargs.tool_call_parser=qwen3_xml", "-o provider.kwargs.tool_call_parser=olmo3"
            )  # D5
            ln = replace_once(ln, " -o provider.kwargs.reasoning_parser=olmo3", "")  # D1
            ln = replace_once(ln, " -o provider.kwargs.mamba_ssm_cache_dtype=float32", "")  # D1
            ln = replace_once(
                ln,
                "-o provider.kwargs.timeout=1800",  # D3, D4
                f"-o provider.kwargs.timeout=1800 -o '{HF_OVR_ARG}' -o provider.kwargs.generation_config={REL}",
            )
            ln = replace_once(ln, f"-m {SRC_MODEL} ", f"-m {MODELS[model]} ")
            ln = replace_once(ln, "--experiment-name c4-thinkoff-s34521-bfcl-single ", f"--experiment-name {name} ")
            ln = replace_once(
                ln, "--experiment-group c4-thinkoff-s34521-bfcl-single-20261005 ", f"--experiment-group {name}-{DATE} "
            )
        out.append(ln)
    t["arguments"] = ["\n".join(out)]
    t["name"] = name  # D7
    t["context"]["priority"] = "normal"
    t["constraints"]["cluster"] = ["ai2/jupiter"]
    spec["description"] = (
        f"o3i7b cell {cell} on {model} ({MODELS[model]}); copied from {pathlib.Path(SRC['BFCL']).name}"
        "; changes: model, names, ladders tar + hero-core + olmoe3 plugins + fla removed, released"
        " tokenizer + generation_config, flat YaRN hf_overrides, tool_call_parser olmo3, no"
        " reasoning_parser (see mkspecs_o3i7b.py D1-D8)."
    )
    # every untouched line is byte-identical, in order; only the pip and run lines changed
    changed = [(x, y) for x, y in zip(kept, out) if x != y]
    assert len(changed) == 2, changed
    s2, o2 = copy.deepcopy(src), copy.deepcopy(spec)
    for d in (s2, o2):
        d.pop("description")
        d["tasks"][0].pop("arguments")
        d["tasks"][0].pop("name")
    assert s2 == o2
    joined = json.dumps(spec["tasks"])
    for gone in (
        "olmoe3",
        "hero-core",
        "hero-ladder",
        "ladders-3542d99",
        "flash-linear-attention",
        "mamba_ssm",
        "tok-thinkoff",
        "reasoning_parser",
        "qwen3_xml",
        SRC_MODEL,
    ):
        assert gone not in joined, gone
    return spec


def build(cell, model):
    name = f"o3i7b-{model}-{cell}-t10"
    spec = (build_bfcl if CELLS[cell][0] == "BFCL" else build_olmo_eval)(cell, model, name)
    return name, spec


def mask_model(spec, model, name):
    """The spec with its model path (only where it is the model: -m and the description) and names masked."""
    s = copy.deepcopy(spec)
    s["description"] = s["description"].replace(f"on {model} ({MODELS[model]})", "on <model>")
    t = s["tasks"][0]
    a = t["arguments"]
    if len(a) == 1:  # BFCL script
        a[0] = replace_once(a[0], f"-m {MODELS[model]} ", "-m <M> ")
    else:
        assert a[a.index("-m") + 1] == MODELS[model]
        a[a.index("-m") + 1] = "<M>"
    return json.loads(json.dumps(s).replace(name, "<N>"))


def same_but_model(a, b, ma, mb, na, nb):
    return mask_model(a, ma, na) == mask_model(b, mb, nb)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (HERE / "hf_overrides.json").write_text(json.dumps(HF_OVERRIDES) + "\n")
    rows = []
    for cell in CELLS:
        built = {m: build(cell, m) for m in MODELS}
        (na, sa), (nb, sb) = built["released"], built["tpu"]
        assert same_but_model(sa, sb, "released", "tpu", na, nb), cell
        for name, spec in built.values():
            (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
            t = spec["tasks"][0]
            ref = (
                env(t, "GIT_REF")["value"][:10] if any(e["name"] == "GIT_REF" for e in t["envVars"]) else "9538453e55"
            )
            rows.append(
                (
                    name,
                    t["resources"]["gpuCount"],
                    t["timeout"],
                    t["context"]["minRuntime"],
                    t["context"]["autoResume"],
                    ref,
                )
            )
    for r in rows:
        print(*r, sep="\t")
    print(len(rows), "specs written to", OUT, file=sys.stderr)
