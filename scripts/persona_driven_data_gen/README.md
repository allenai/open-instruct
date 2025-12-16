## Persona-driven Data Generation


To start make sure you have your OpenAI and Anthropic API keys and have installed the libraries listed in `requirements.txt`:

```
pip install -r requirements.txt
```

This folder contains code to synthetically generate data (both prompts and responses) for target skill using a [persona-driven approach](https://arxiv.org/pdf/2406.20094):


**1- Precise Instruction Following:**

```
# Generate Instruction Following prompts
python persona_driven_generate_ifdata.py --model "gpt-4o" --start_index 0 --end_index 1000 --output_path if_prompts.jsonl --openai_key Z --org_id YYY --dataset ai2-adapt-dev/personahub_personas --template instruction_following

# Generate Responses for generated prompts
python persona_driven_generate_ifdata.py --model "gpt-4o" --start_index 0 --end_index 1000 --output_path if_solutions.jsonl --openai_key Z --org_id YYY --dataset if_prompts.jsonl --template instruction_following_solution

# Rewrite prompts to form Rejected Response (used for Presona-IF DPO data)
python persona_driven_generate_ifdata.py --model "gpt-4o" --start_index 0 --end_index 1000 --output_path if_solutions.jsonl --openai_key Z --org_id YYY --dataset if_prompts.jsonl --template rewrite_if_prompt
```


**2- Math World Problems**
```
# Generate math word problems
python persona_driven_generate_math_code.py --model "gpt-4o" --end_index 1000 --output_path <MATH_PROBLEMS> --openai_key XXX --org_id YYY --dataset ai2-adapt-dev/personahub_personas --template math

# Generate math solutions for generated math problems
python persona_driven_generate_math_code.py --model "gpt-4o" --end_index 1000 --output_path <OUTPUT_MATH> --openai_key XXX --org_id YYY --dataset <MATH_PROBLEMS> --template math_solution
```
Note that you can change `--template` to any of `['grade_math', 'math_int_algebra']` to generate other types of math data.



**3- Code (python)**
```
# Generate python problems

python persona_driven_generate_math_code.py --model "gpt-4o" --start_index 0 --end_index 1000 --output_path <PYTHON_PROBLEMS> --openai_key XXX --org_id YYY --dataset ai2-adapt-dev/personahub_personas --template code

# Generate python code
python persona_driven_generate_math_code.py --org_name anthropic --model 'claude-3-5-sonnet-20240620' --start_index 0 --end_index 1000 --output_path <OUTPUT_CODE> --openai_key XXX --org_id YYY --dataset <PYTHON_PROBLEMS> --template code_solution
```
Note that we used `claude-3-5-sonnet-20240620` to generate python codes.


All generated prompts and solutions will be saved in the `messages` format ready for supervised finetunig. An example output can be found [here](https://huggingface.co/datasets/ai2-adapt-dev/personahub_math_v5_regen_149960)
