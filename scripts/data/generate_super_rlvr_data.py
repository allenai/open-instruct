import argparse, json, multiprocessing as mp, os, random, signal, sys
from tqdm import tqdm
from datasets import Dataset
from open_instruct.VerifiableProblem.verifiable.problems import problem2class
from open_instruct.VerifiableProblem.verifiable.parameter_controllers import problem2controller
from open_instruct.ground_truth_utils import VerifiableProblemZVerifier, VerifierConfig
from utils.timeout import TimeoutException

# --------------------------------------------------------------------------- #
# hard timeout for every call ------------------------------------------------ #
# --------------------------------------------------------------------------- #
_HARD_LIMIT = 1        # seconds per instance; make it a CLI flag if you like

def _alarm_handler(signum, _frame):
    raise TimeoutException

def _gen_one(arg):
    task_name, seed, parameter = arg
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(_HARD_LIMIT)          # start watchdog
    try:
        inst = problem2class[task_name]()
        inst.generator(seed, parameter)
        prompt = inst.prompt_generator()

        return dict(
            ok=True,
            data=dict(
                dataset=["verifiable_problem_z"],
                ground_truth=[{"task_name": task_name,
                        "parameters": json.dumps(inst.__dict__)}],
                messages=[{"role": "user", "content": prompt}],
            ),
        )
    except TimeoutException:
        return {"ok": False, "err": "timeout", "task": task_name, "seed": seed}
    except Exception as e:
        return {"ok": False, "err": repr(e), "task": task_name, "seed": seed}
    finally:
            # 1) stop the timer, 2) put the old handler back
            signal.alarm(0)
            signal.signal(signal.SIGALRM, signal.SIG_DFL)

# --------------------------------------------------------------------------- #
# main ---------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_list", nargs="+", required=True)
    ap.add_argument("--samples_per_task", type=int, required=True)
    ap.add_argument("--difficulty_levels", type=int, default=1)
    args = ap.parse_args()

    seed = 42
    upd_every = args.samples_per_task // args.difficulty_levels
    all_rows = []

    # **spawn** avoids the “fork-after-import” dead-lock
    ctx = mp.get_context("spawn")
    n_workers = max(1, ctx.cpu_count() - 2)

    with ctx.Pool(processes=n_workers, maxtasksperchild=256) as pool:
        for task in tqdm(args.task_list, desc="tasks"):
            pc = problem2controller[task]()
            job_args = []

            for i in range(args.samples_per_task):
                # grab a fresh parameter before every call
                param = random.choice(pc.get_parameter_list())
                job_args.append((task, seed, param))
                seed += 1
                if (i + 1) % upd_every == 0:
                    pc.update()

            for res in pool.imap_unordered(_gen_one, job_args, chunksize=32):
                if res["ok"]:
                    all_rows.append(res["data"])
                elif res["err"] == "timeout":
                    print(f"[timeout] {res['task']} (seed {res['seed']})", file=sys.stderr)
                else:
                    print(f"[error]   {res['task']} (seed {res['seed']}): {res['err']}",
                          file=sys.stderr)

    print(f"kept {len(all_rows)} / {seed - 42} examples")

    # push outside the Pool context; avoids weird forking issues in HF client
    Dataset.from_list(all_rows).push_to_hub("hamishivi/verifiable_problem_z")
