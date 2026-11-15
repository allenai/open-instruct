"""Run open_instruct/olmo_core_finetune.py with speed-study hooks (A4B SFT throughput ladder).

  torchrun ... scripts/c12/speed/speed_wrap.py <olmo_core_finetune.py arguments>

Run from the open-instruct root. Every hook is off unless its variable is set, so with none set this is exactly
olmo_core_finetune.py.

  SPEED_SCHED_STEPS=N    build the LR schedule for N total steps (warmup = int(N * warmup_ratio)) whatever
                         --max_train_steps says, so a short run follows the first steps of the long run's schedule.
  SPEED_NO_FINAL_SAVE=1  skip the checkpointer's end-of-training save.
  SPEED_CFG_PATCH=<json> overrides applied to a copy of --config_name before anything reads it. Keys are dotted
                         paths from the config root ("model.recompute_each_block"); a key starting "block." is applied
                         to model.block and to every model.block_overrides entry that has the parent dict.
  SPEED_PROFILE=<skip>,<active>,<ranks>   torch profiler: skip <skip> steps, warm up 1, trace <active> steps on the
                         ranks (";"-separated). Traces go to /output (the Beaker result dir) and a category summary
                         (trace_summary.py) is printed by each traced rank.
  SPEED_UPDATE_STATS=N   for optimizer steps 1..N, rank 0 prints per category (dense, routers, norms, routed experts) the
                         post-clip grad norm reaching AdamW, the fp32 master update norm / max / rms, the norm of the
                         bf16 model-param change, the fraction of elements whose bf16 value changed, and the fraction of
                         grads below eps / exactly zero (lines "SPEED rank 0 UPD ...").
  SPEED_MEM=1            every rank prints its max device memory in use (total - free), max reserved and max
                         allocated over the run at the end of training.
"""

import json
import os
import runpy
import sys
import time

import torch

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from olmo_core.train.callbacks import Callback, ProfilerCallback  # noqa: E402
from olmo_core.train.callbacks import checkpointer as oc_checkpointer  # noqa: E402

from open_instruct import olmo_core_utils  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


def _log(msg):
    print(f"SPEED rank {RANK} {msg}", flush=True)


# --- schedule length -------------------------------------------------------------------------------------------
SCHED_STEPS = int(os.environ.get("SPEED_SCHED_STEPS", "0"))
if SCHED_STEPS:
    _orig_build_scheduler = olmo_core_utils.build_scheduler

    def _build_scheduler(lr_scheduler_type, warmup_ratio, num_training_steps):
        _log(f"scheduler built for {SCHED_STEPS} steps instead of {num_training_steps}")
        return _orig_build_scheduler(lr_scheduler_type, warmup_ratio, SCHED_STEPS)

    olmo_core_utils.build_scheduler = _build_scheduler

# --- no final checkpoint ---------------------------------------------------------------------------------------
if os.environ.get("SPEED_NO_FINAL_SAVE") == "1":
    oc_checkpointer.CheckpointerCallback.post_train = lambda self: None


# --- model config patch ----------------------------------------------------------------------------------------
def _set(d, path, value):
    *parents, leaf = path.split(".")
    for p in parents:
        if not isinstance(d, dict) or p not in d:
            return False
        d = d[p]
    if not isinstance(d, dict):
        return False
    d[leaf] = value
    return True


PATCH = os.environ.get("SPEED_CFG_PATCH")
if PATCH:
    patch = json.loads(PATCH)
    i = sys.argv.index("--config_name")
    with open(sys.argv[i + 1]) as f:
        cfg = json.load(f)
    for key, value in patch.items():
        if key.startswith("block."):
            sub = key[len("block.") :]
            targets = [cfg["model"]["block"], *cfg["model"].get("block_overrides", {}).values()]
            hits = sum(_set(t, sub, value) for t in targets)
            if hits == 0:
                raise KeyError(f"SPEED_CFG_PATCH {key} matched nothing")
        elif not _set(cfg, key, value):
            raise KeyError(f"SPEED_CFG_PATCH {key} matched nothing")
    out = f"/tmp/speed_cfg_rank{RANK}.json"
    with open(out, "w") as f:
        json.dump(cfg, f, indent=1)
    sys.argv[i + 1] = out
    _log(f"config patched {patch} -> {out}")


# --- extra callbacks -------------------------------------------------------------------------------------------
class _MemCallback(Callback):
    def __init__(self):
        self.max_used = 0
        self.max_reserved = 0
        self.max_alloc = 0
        self.t_last = None

    def post_train_batch(self):
        # Before the GPU monitor's post_step resets the peak counters.
        free, total = torch.cuda.mem_get_info()
        self.max_used = max(self.max_used, total - free)
        self.max_reserved = max(self.max_reserved, torch.cuda.max_memory_reserved())
        self.max_alloc = max(self.max_alloc, torch.cuda.max_memory_allocated())

    def post_train(self):
        g = 2**30
        _log(
            f"MEM max_device_used {self.max_used / g:.1f} GiB max_reserved {self.max_reserved / g:.1f} GiB "
            f"max_allocated {self.max_alloc / g:.1f} GiB"
        )


class _SpeedProfiler(ProfilerCallback):
    def _on_trace_ready(self, prof):
        path = f"/output/trace-rank{RANK}-step{prof.step_num}.json.gz"
        t = time.time()
        prof.export_chrome_trace(path)
        _log(f"trace saved to {path} in {time.time() - t:.0f}s")
        try:
            import trace_summary

            trace_summary.summarize(path, top=45, out=lambda s: print(f"SPEED-TRACE rank {RANK} {s}", flush=True))
        except Exception as e:  # never kill training over the summary
            _log(f"trace summary failed: {e!r}")


PROFILE = os.environ.get("SPEED_PROFILE")
MEM = os.environ.get("SPEED_MEM") == "1"
if PROFILE or MEM:
    _orig_callbacks = olmo_core_utils.build_base_callbacks

    def _build_base_callbacks(*args, **kwargs):
        result = _orig_callbacks(*args, **kwargs)
        if PROFILE:
            skip, active, ranks = PROFILE.split(",")
            result["speed_profiler"] = _SpeedProfiler(
                skip_first=int(skip),
                wait=0,
                warmup=1,
                active=int(active),
                repeat=1,
                with_stack=False,
                ranks=[int(r) for r in ranks.split(";")],
            )
        if MEM:
            result["speed_mem"] = _MemCallback()
        return result

    olmo_core_utils.build_base_callbacks = _build_base_callbacks

# --- optimizer update statistics -------------------------------------------------------------------------------
UPD_STEPS = int(os.environ.get("SPEED_UPDATE_STATS", "0"))
if UPD_STEPS:
    import torch.distributed as dist
    from olmo_core.optim import moe_optimizer as oc_moe_opt

    _CATS = ("dp_dense", "dp_router", "dp_norm", "ep_experts", "ep_other")
    _STATE = {"active": False, "step": 0, "acc": None, "ptr2cat": {}}

    def _category(pg, name):
        if pg == "ep_dp":
            return "ep_experts" if "routed_experts" in name else "ep_other"
        if "router" in name:
            return "dp_router"
        if "norm" in name:
            return "dp_norm"
        return "dp_dense"

    _orig_foreach = oc_moe_opt.foreach_adamw_step

    def _foreach_with_stats(params, grads, exp_avgs, exp_avg_sqs, steps, **kw):
        if not _STATE["active"]:
            return _orig_foreach(params, grads, exp_avgs, exp_avg_sqs, steps, **kw)
        old = [p.detach().clone() for p in params]
        _orig_foreach(params, grads, exp_avgs, exp_avg_sqs, steps, **kw)
        acc = _STATE["acc"]
        _STATE["eps"] = kw.get("eps")
        _STATE["lr"] = kw.get("lr")
        _STATE["step_factor"] = kw.get("step_factor")
        eps = kw.get("eps") or 1e-8
        with torch.no_grad():
            for p, o, g in zip(params, old, grads):
                cat = _STATE["ptr2cat"].get(p.data_ptr())
                if cat is None:
                    continue  # replicated copy counted on another rank, or unknown
                a = acc[cat]
                g = g.float()
                d = p - o
                db = p.to(torch.bfloat16).float() - o.to(torch.bfloat16).float()
                a[0] += g.pow(2).sum()
                a[1] += d.pow(2).sum()
                a[2] = torch.maximum(a[2], d.abs().max())
                a[3] += db.pow(2).sum()
                a[4] += (db != 0).sum().float()
                a[5] += (g.abs() < eps).sum().float()
                a[6] += (g == 0).sum().float()
                a[7] += float(g.numel())
                a[8] = torch.maximum(a[8], g.abs().max())
        del old

    oc_moe_opt.foreach_adamw_step = _foreach_with_stats
    _orig_opt_step = oc_moe_opt.OLMoDDPOptimizer.step

    def _opt_step_with_stats(self, closure=None):
        _STATE["step"] += 1
        if _STATE["step"] > UPD_STEPS:
            return _orig_opt_step(self, closure)
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        # one entry per local main-param shard that this rank owns uniquely
        ptr2cat = {}
        for group in self.param_groups:
            pg = group["pg"]
            mesh = self._get_dp_device_mesh_for_tag(pg)
            for name, p in group["named_params"].items():
                if not p.requires_grad:
                    continue
                main = self.states[f"{name}.main"]
                if main.to_local().numel() == 0:
                    continue
                if not main.placements[0].is_shard() and mesh is not None and mesh.get_local_rank() != 0:
                    continue
                ptr2cat[main.to_local().data_ptr()] = _category(pg, name)
        _STATE["ptr2cat"] = ptr2cat
        _STATE["acc"] = {c: [torch.zeros((), device=dev) for _ in range(9)] for c in _CATS}
        _STATE["active"] = True
        try:
            out = _orig_opt_step(self, closure)
        finally:
            _STATE["active"] = False
        acc = _STATE["acc"]
        sums = torch.stack([torch.stack([acc[c][i] for i in (0, 1, 3, 4, 5, 6, 7)]) for c in _CATS])
        maxs = torch.stack([torch.stack([acc[c][i] for i in (2, 8)]) for c in _CATS])
        if dist.is_initialized():
            dist.all_reduce(sums)
            dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
        if RANK == 0:
            sf = _STATE.get("step_factor")
            sf = float(sf) if torch.is_tensor(sf) else sf
            gn = self.latest_grad_norm
            gn = float(gn) if torch.is_tensor(gn) else gn
            _log(
                f"UPD step {_STATE['step']} lr {_STATE.get('lr')} eps {_STATE.get('eps')} step_factor {sf} "
                f"total_grad_norm_preclip {gn}"
            )
            for c, s, m in zip(_CATS, sums.tolist(), maxs.tolist()):
                gsq, usq, bsq, bchg, glt, gz, n = s
                if n == 0:
                    continue
                _log(
                    f"UPD step {_STATE['step']} {c:10s} numel {n:.4g} grad_norm_postclip {gsq ** 0.5:.4e} "
                    f"grad_max {m[1]:.3e} upd_norm {usq ** 0.5:.4e} upd_max {m[0]:.3e} "
                    f"upd_rms {(usq / n) ** 0.5:.3e} bf16_change_norm {bsq ** 0.5:.4e} "
                    f"bf16_frac_changed {bchg / n:.4f} frac_|g|<eps {glt / n:.4f} frac_g==0 {gz / n:.4f}"
                )
        return out

    oc_moe_opt.OLMoDDPOptimizer.step = _opt_step_with_stats

sys.argv = [os.path.join(ROOT, "open_instruct", "olmo_core_finetune.py"), *sys.argv[1:]]
runpy.run_path(sys.argv[0], run_name="__main__")
