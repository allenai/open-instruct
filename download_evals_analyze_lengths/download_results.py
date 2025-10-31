from beaker import *
import os

def beaker_experiment_failed(exp):
    """
    Returns if beaker experiment failed.
    """
    if exp.jobs[0].execution.spec.replicas is not None:
        num_replicas = exp.jobs[0].execution.spec.replicas
    else:
        num_replicas = 1
        
    checks = []
    for job in exp.jobs:
        if job.status.exited is None:
            return True # at least one job is still running
        checks.append(job.status.finalized is not None and job.status.exit_code == 0)

    return sum(checks) != num_replicas

def gather_experiments(author_list, workspace_name="ai2/tulu-3-results", relevant_match="ifeval_ood", limit=30):
    """
    Gather experiments from workspace_name, with relevant_match in their name.
    """
    beaker = Beaker.from_env()
    experiments = []

    # Nice bookkeeping to see how many failed per author - a good gut check, if nothing else
    num_author_exps = {}
    if author_list:
        for author in author_list:
            num_author_exps[author] = 0

    exps = beaker.workspace.experiments(workspace=workspace_name, limit=limit, match=relevant_match)
    for exp in exps:
        author = exp.author.name

        # If author_list is provided, filter by it; otherwise accept any author
        if author_list and author not in author_list:
            continue
        # only collect final experiments
        if beaker_experiment_failed(exp):
            continue
        experiments.append(exp)
        if author_list:
            num_author_exps[author] += 1

    if author_list:
        print (f"Total experiments from {author_list}: {len(experiments)}")
        for author, count in num_author_exps.items():
            print(f"Author {author} had {count} experiments")
    else:
        print (f"Total experiments (all authors): {len(experiments)}")
    return experiments

def main():
    from pprint import pprint
    beaker = Beaker.from_env()
    workspace = "ai2/saumyam"
    experiments = gather_experiments(["saumyam"], limit=7, workspace_name=workspace, relevant_match="ifeval_ood")

    for i,exp in enumerate(experiments):
        pprint(vars(exp))
        result_dataset = exp.jobs[0].result.beaker

        # api models have different naming
        if workspace == "ai2/saumyam":
            model=exp.jobs[0].execution_results['model_config']['model']
        else:
            model=exp.jobs[0].execution_results['model_config']['model_path']

        # skip fetching if model path already exists
        if os.path.exists(model):
            print(f"Skipping {model} because it already exists")
            continue
        beaker.dataset.fetch(result_dataset, target=model, quiet=True)
        
        

if __name__ == "__main__":
    main()