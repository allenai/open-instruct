"""Per-update timeline for an async MILES/Core run, from its Beaker log.

Usage: python scripts/miles/analyze_async_run.py LOG [--json OUT]

LOG is the text of ``beaker experiment logs EXPERIMENT`` for the trainer replica.

Async phases overlap, so this reports per optimizer step: the trainer's own
step time, publication time, the cadence between consecutive steps, and the
implied trainer idle fraction; plus rollout-side throughput and eval scores.
"""

import ast
import json
import re
import statistics as st
import sys
from datetime import datetime

STAMP = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) ([a-z0-9_]+)\]")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def literal(text):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None


def parse(path):
    steps, contracts, pubs, perf, evals, checks = {}, {}, {}, {}, {}, []
    with open(path, errors="replace") as handle:
        lines = handle.readlines()
    for raw in lines:
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip("\n")
        m = STAMP.search(line)
        if not m:
            continue
        t, who = ts(m.group(1)), m.group(2)
        body = line[m.end() :]
        if who == "actor_cell0_rank0":
            if "Core optimizer step" in body:
                n = int(re.search(r"optimizer step (\d+):", body).group(1))
                d = literal(body[body.index("{") :])
                steps[n] = dict(t=t, **({k.split("/", 1)[1]: v for k, v in d.items()} if d else {}))
            elif "Core step contract:" in body:
                d = literal(body[body.index("{") :])
                if d:
                    contracts[d["step"]] = dict(t=t, **d)
            elif "Core weight publication:" in body:
                d = literal(body[body.index("{") :])
                if d and not d.get("repeated_version"):
                    pubs[d["version"]] = dict(t=t, **d)
            elif "Core scoring check:" in body:
                d = literal(body[body.index("{") :])
                if d:
                    checks.append(dict(t=t, **d))
        elif who == "rollout_manager":
            mm = re.search(r"metrics\.py:\d+ - (perf|eval) (\d+): ", body)
            if mm:
                d = literal(body[mm.end() :])
                if d is None:
                    continue
                (perf if mm.group(1) == "perf" else evals)[int(mm.group(2))] = dict(t=t, **d)
    return dict(steps=steps, contracts=contracts, pubs=pubs, perf=perf, evals=evals, checks=checks)


def summarize(r, warm_from=3):
    steps, contracts, pubs, perf = r["steps"], r["contracts"], r["pubs"], r["perf"]
    rows = []
    for n in sorted(steps):
        s, c, p = steps[n], contracts.get(n, {}), pubs.get(n, {})
        prev = steps.get(n - 1)
        rows.append(
            dict(
                step=n,
                cadence=(s["t"] - prev["t"]) if prev else None,
                step_seconds=s.get("step_seconds", c.get("elapsed_seconds")),
                microbatches=c.get("local_microbatches"),
                active_tokens=(c.get("normalization") or {}).get("active_tokens"),
                publication=p.get("total_seconds"),
                broadcast=p.get("broadcast_seconds"),
                engine=p.get("engine_seconds"),
                logprob_gap=s.get("train_rollout_logprob_abs_diff"),
                tis_clipfrac=s.get("tis_clipfrac"),
                pg_clipfrac=s.get("pg_clipfrac"),
                grad_norm=s.get("grad_norm"),
                scoring_pass=c.get("scoring_pass"),
            )
        )
    out = {"updates": len(rows), "rows": rows}
    warm = [x for x in rows if x["step"] >= warm_from and x["cadence"]]
    if warm:
        cad = st.mean(x["cadence"] for x in warm)
        trn = st.mean(x["step_seconds"] for x in warm if x["step_seconds"] is not None)
        pub = (
            st.mean(x["publication"] for x in warm if x["publication"] is not None)
            if any(x["publication"] for x in warm)
            else 0.0
        )
        out["warm"] = dict(
            cycles=len(warm),
            cadence_mean=cad,
            cadence_median=st.median(x["cadence"] for x in warm),
            train_step_mean=trn,
            publication_mean=pub,
            trainer_busy_fraction=(trn + pub) / cad,
            microbatches_per_rank=warm[0]["microbatches"],
            seconds_per_microbatch=trn / warm[0]["microbatches"] if warm[0]["microbatches"] else None,
            logprob_gap_mean=st.mean(x["logprob_gap"] for x in warm if x["logprob_gap"] is not None),
            tis_clipfrac_mean=st.mean(x["tis_clipfrac"] for x in warm if x["tis_clipfrac"] is not None),
        )
    if perf:
        ks = sorted(perf)
        out["rollout"] = dict(
            collections=len(ks),
            reward_mean=st.mean(perf[k].get("rollout/episode_raw_reward", 0.0) for k in ks),
            response_len_mean=st.mean(perf[k].get("rollout/response_len/mean", 0.0) for k in ks),
            truncated_ratio_mean=st.mean(perf[k].get("rollout/truncated_ratio", 0.0) for k in ks),
            all_zero_group_frac=st.mean(perf[k].get("rollout/zero_std/all_zero_percentage", 0.0) for k in ks),
            all_one_group_frac=st.mean(perf[k].get("rollout/zero_std/all_one_percentage", 0.0) for k in ks),
            staleness_max=max(perf[k].get("rollout/fully_async/max_staleness", 0) for k in ks),
            stale_groups_filtered=sum(perf[k].get("rollout/fully_async/stale_groups_filtered", 0) for k in ks),
            tokens_per_gpu_per_sec_mean=st.mean(perf[k].get("perf/tokens_per_gpu_per_sec", 0.0) for k in ks),
        )
        if len(ks) > warm_from:
            span = perf[ks[-1]]["t"] - perf[ks[warm_from]]["t"]
            samples = sum(perf[k].get("rollout/num_training_samples", 0) for k in ks[warm_from + 1 :])
            toks = sum(
                perf[k].get("rollout/num_training_samples", 0) * perf[k].get("rollout/response_len/mean", 0.0)
                for k in ks[warm_from + 1 :]
            )
            out["rollout"]["warm_response_tokens_per_sec"] = toks / span if span else None
            out["rollout"]["warm_samples_per_sec"] = samples / span if span else None
    if r["evals"]:
        out["evals"] = {
            n: {
                k: v
                for k, v in e.items()
                if k.startswith("eval/") and k.count("/") == 1 or k.endswith("truncated_ratio")
            }
            for n, e in sorted(r["evals"].items())
        }
    if r["checks"]:
        out["scoring_checks"] = [
            {k: c[k] for k in ("step", "mean_abs", "max_abs", "tokens_above_edge", "tolerance") if k in c}
            for c in r["checks"]
        ]
    return out


def main():
    path = sys.argv[1]
    out = summarize(parse(path))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as handle:
            json.dump(out, handle, indent=2)
    print(f"updates: {out['updates']}")
    for x in out["rows"]:
        print(
            "  step {step:>3}  cadence {cad}  train {tr}  mb {mb}  pub {pub}  gap {gap}  tis_clip {tc}  {sp}".format(
                step=x["step"],
                cad=f"{x['cadence']:7.1f}s" if x["cadence"] else "      - ",
                tr=f"{x['step_seconds']:6.1f}s" if x["step_seconds"] else "   -  ",
                mb=x["microbatches"],
                pub=f"{x['publication']:5.2f}s" if x["publication"] else "  -  ",
                gap=f"{x['logprob_gap']:.4f}" if x["logprob_gap"] is not None else "-",
                tc=f"{x['tis_clipfrac']:.3f}" if x["tis_clipfrac"] is not None else "-",
                sp=x["scoring_pass"] or "",
            )
        )
    for key in ("warm", "rollout", "evals", "scoring_checks"):
        if key in out:
            print(f"\n{key}:")
            print(json.dumps(out[key], indent=2, default=str))


if __name__ == "__main__":
    main()
