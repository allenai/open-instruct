"""
My dumb script to create ground truth data for GTRL training.
"""

import random

from datasets import Dataset, load_dataset
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils
from open_instruct.math_utils import last_boxed_only_string, remove_boxed

# exemplars we will use to prompt the model
GSM8K_EXEMPLARS = [
    {
        "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "cot_answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is 6.",
        "short_answer": "6",
    },
    {
        "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "cot_answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is 5.",
        "short_answer": "5",
    },
    {
        "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "cot_answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is 39.",
        "short_answer": "39",
    },
    {
        "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "cot_answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is 8.",
        "short_answer": "8",
    },
    {
        "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "cot_answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is 9.",
        "short_answer": "9",
    },
    {
        "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "cot_answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is 29.",
        "short_answer": "29",
    },
    {
        "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "cot_answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is 33.",
        "short_answer": "33",
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "cot_answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is 8.",
        "short_answer": "8",
    },
]


MATH_EXAMPLARS = [
    {
        "question": "Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "cot_answer": "The expressions inside each square root must be non-negative.\nTherefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$.\nAlso, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$.\nTherefore, the domain of the expression is $\\boxed{[2,5)}$.",
        "short_answer": "[2,5)",
    },
    {
        "question": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "cot_answer": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
        "short_answer": "24",
    },
    {
        "question": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "cot_answer": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{align*}\n30n&=480\\\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
        "short_answer": "16",
    },
    {
        "question": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "cot_answer": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
        "short_answer": "-\\frac{2}{3}",
    },
]
math_messages = [
    [{"role": "user", "content": sample["question"]}, {"role": "assistant", "content": sample["cot_answer"]}]
    for sample in MATH_EXAMPLARS
]
# flatten
math_messages = [item for sublist in math_messages for item in sublist]

# now, we construct gsm8k data
gsm8k_prompt = ""
for sample in GSM8K_EXEMPLARS:
    gsm8k_prompt += f"Question: {sample['question'].strip()}\nAnswer:{sample['cot_answer'].strip()}\n\n"

gsm8k_dataset = load_dataset("gsm8k", "main", split="train", num_proc=open_instruct_utils.max_num_processes())
new_data = []
for sample in gsm8k_dataset:
    answer = sample["answer"].split("####")[-1].strip()
    new_data.append(
        {
            "messages": [{"role": "user", "content": gsm8k_prompt + f"Question: {sample['question'].strip()}"}],
            "ground_truth": answer,
            "dataset": "gsm8k",
        }
    )

# also make a test split for eval
gsm8k_dataset = load_dataset("gsm8k", "main", split="test", num_proc=open_instruct_utils.max_num_processes())
test_data = []
for sample in gsm8k_dataset:
    answer = sample["answer"].split("####")[-1].strip()
    test_data.append(
        {
            "messages": [{"role": "user", "content": gsm8k_prompt + f"Question: {sample['question'].strip()}"}],
            "ground_truth": answer,
            "dataset": "gsm8k",
        }
    )

# now, we construct math data
math_prompt = ""
for sample in MATH_EXAMPLARS:
    math_prompt += f"Question: {sample['question'].strip()}\nAnswer:{sample['cot_answer'].strip()}\n\n"
math_dataset = load_dataset("lighteval/MATH", "all", split="train", num_proc=open_instruct_utils.max_num_processes())
for sample in math_dataset:
    # same code used to extract answer for eval
    answer = remove_boxed(last_boxed_only_string(sample["solution"]))
    if answer is None:
        print("skipping")
        continue
    new_data.append(
        {
            "messages": [{"role": "user", "content": math_prompt + f"Question: {sample['problem'].strip()}"}],
            "ground_truth": answer,
            "dataset": "MATH",
        }
    )

# combine into one dataset and push
# random.shuffle(new_data)
# train_dataset = Dataset.from_list(new_data)
# test_dataset = Dataset.from_list(test_data)
# dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
# dataset.push_to_hub("ai2-adapt-dev/gsm8k_math_ground_truth")

# # alternate dataset: metamathqa!
# metamathqa_dataset = load_dataset("meta-math/MetaMathQA", "main", split="train", num_proc=open_instruct_utils.max_num_processes())
# # let's re-use the MATH prompt.
# new_data = []
# def extract_answer(text):
#     # Regular expression to match content after "The answer is:" including numbers, LaTeX fractions, or other expressions
#     pattern = r'The answer is:\s*([^\s.]+)'
#     matches = re.findall(pattern, text)
#     return matches[-1] if matches else None
# for sample in metamathqa_dataset:
#     # same code used to extract answer for eval
#     answer = extract_answer(sample["response"])
#     if answer is None:
#         print("skipping")
#         continue
#     new_data.append({
#         "messages": [{"role": "user", "content": math_prompt + f"Question: {sample['query'].strip()}"}],
#         "ground_truth": answer,
#         "dataset": "MATH"  # lets use the math eval setup
#     })

# # combine into one dataset and push
# random.shuffle(new_data)
# dataset = Dataset.from_list(new_data)
# dataset.push_to_hub("ai2-adapt-dev/metamathqa_ground_truth")

# alternate dataset: numina-tir
metamathqa_dataset = load_dataset(
    "AI-MO/NuminaMath-TIR", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# let's re-use the MATH prompt.
new_data = []


def find_last_outermost_boxed(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    if retval is not None:
        retval = retval[7:-1]  # remove \boxed{}
    return retval


for sample in tqdm(metamathqa_dataset):
    # same code used to extract answer for eval
    answer = find_last_outermost_boxed(sample["solution"])
    if answer is None:
        print("skipping")
        continue
    # lets use multi-turn cot prompt instead
    new_data.append(
        {
            "messages": [{"role": "user", "content": math_prompt + f"Question: {sample['problem'].strip()}"}],
            "ground_truth": answer,
            "dataset": "MATH",  # lets use the math eval setup
        }
    )

# combine into one dataset and push
random.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/numinamath_tir_ground_truth_one_turn")

# alternate dataset: numina-cot (much, much larger)
metamathqa_dataset = load_dataset(
    "AI-MO/NuminaMath-CoT", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# let's re-use the MATH prompt.
new_data = []
for sample in tqdm(metamathqa_dataset):
    # same code used to extract answer for eval
    answer = find_last_outermost_boxed(sample["solution"])
    if answer is None:
        print("skipping")
        continue
    # lets use multi-turn cot prompt instead
    new_data.append(
        {
            "messages": [{"role": "user", "content": math_prompt + f"Question: {sample['problem'].strip()}"}],
            "ground_truth": answer,
            "dataset": "MATH",  # lets use the math eval setup
        }
    )

# combine into one dataset and push
random.shuffle(new_data)
dataset = Dataset.from_list(new_data)
dataset.push_to_hub("ai2-adapt-dev/numinamath_cot_ground_truth_one_turn")
