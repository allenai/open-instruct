import argparse
from open_instruct.VerifiableProblem.verifiable.problems import problem2class
from open_instruct.VerifiableProblem.verifiable.parameter_controllers import problem2controller
import random
import pandas as pd
from datasets import Dataset
import os

def process_fn(example, task_name):

    data = {
        "dataset": f"verifiable_problem_z",
        "label": task_name,
        "messages": [{
            "role": "user",
            "content": example["prompt"]
        }],
    }
    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_list', nargs='+')
    parser.add_argument('--samples_per_task', type=int)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--difficulty_levels', type=str)
    args = parser.parse_args()

    seed = 42

    update_difficulty_every = args.samples_per_task // args.difficulty_levels
    
    
    all_data = []
    for task_name in args.task_list:
        # initial setup
        parameter_controller = problem2controller[task_name]()
        parameter_list = controller.get_parameter_list()
        problem = problem2class[args.task]()
        # generate data
        task_data = []
        for i in range(args.samples_per_task):
            parameter = random.choice(parameter_list)
            instance = problem()
            instance.generator(seed, parameter)
            seed += 1
            task_data.append(process_fn(instance, task_name))
            if i % update_difficulty_every == 0:
                parameter_list = parameter_controller.update()
        all_data.extend(task_data)

    ds = Dataset.from_list(all_data)
    ds.push_to_hub("hamishivi/verifiable_problem_z")
