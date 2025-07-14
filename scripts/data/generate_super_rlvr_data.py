# --- imports ---------------------------------------------------------------
import argparse, os, random, signal, multiprocessing as mp
from tqdm import tqdm
from datasets import Dataset
from open_instruct.VerifiableProblem.verifiable.problems import problem2class
from open_instruct.VerifiableProblem.verifiable.parameter_controllers import problem2controller

# --- per-task helper -------------------------------------------------------
def _timeout_handler(signum, frame):
    raise TimeoutError

def _worker_init():                       # runs **once** in every worker
    signal.signal(signal.SIGALRM, _timeout_handler)

def _gen_one(arg):
    """Run inside the worker.  Must be top-level picklable."""
    task_name, seed, parameter = arg
    signal.alarm(1)                       # hard 1-s wall-clock limit
    try:
        inst = problem2class[task_name]()
        inst.generator(seed, parameter)
        prompt = inst.prompt_generator()
        signal.alarm(0)                   # clear alarm
        return {
            "ok": True,
            "data": {
                "dataset": "verifiable_problem_z",
                "label": {"task_name": task_name, "parameters": inst.__dict__},
                "messages": [{"role": "user", "content": prompt}],
            },
        }
    except TimeoutError:
        return {"ok": False, "err": "timeout", "task": task_name, "seed": seed}
    except Exception as e:
        return {"ok": False, "err": repr(e),      "task": task_name, "seed": seed}

# --- main ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_list", nargs="+")
    ap.add_argument("--samples_per_task", type=int)
    ap.add_argument("--difficulty_levels", type=int)
    args = ap.parse_args()

    seed = 42
    upd_every = args.samples_per_task // args.difficulty_levels
    all_rows = []

    n_workers = max(1, mp.cpu_count() - 2)            # leave a core or two free
    with mp.get_context("fork").Pool(
            processes=n_workers,
            initializer=_worker_init,
            maxtasksperchild=256) as pool:            # auto-recycle after N calls

        for task in tqdm(args.task_list, desc="tasks"):
            pc = problem2controller[task]()
            # build argument tuples once per task
            job_args = []
            for i in range(args.samples_per_task):
                param = pc.get_parameter_list()[0]
                job_args.append((task, seed, param))
                seed += 1
                if (i + 1) % upd_every == 0:
                    pc.update()

            # imap_unordered yields as soon as a worker is done; batch to amortize
            for res in pool.imap_unordered(_gen_one, job_args, chunksize=32):
                if res["ok"]:
                    all_rows.append(res["data"])
                elif res["err"] == "timeout":
                    print(f"[timeout] {res['task']} seed {res['seed']}")
                else:
                    print(f"[error]   {res['task']} seed {res['seed']}: {res['err']}")

    print(f"kept {len(all_rows)} / {seed-42} examples")
    Dataset.from_list(all_rows).push_to_hub("hamishivi/verifiable_problem_z")