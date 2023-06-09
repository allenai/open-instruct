import argparse
import os
import re
import json
import random
import tqdm
import evaluate
from eval.utils import generate_completions, load_hf_lm_and_tokenizer
from eval.mgsm.examplars import MGSM_EXEMPLARS
import pandas as pd


cot_question_encoding_templates = {
    "en": ("Question: {}", "Step-by-Step Answer:"),
    "es": ("Pregunta: {}", "Respuesta paso a paso:"),
    "de": ("Frage: {}", "Schritt-für-Schritt-Antwort:"),
    "fr": ("Question: {}", "Schritt-für-Schritt-Antwort:"),
    "ja": ("問題：{}", "ステップごとの答え："),
    "ru": ("Задача: {}", "Пошаговое решение:"),
    "zh": ("问题：{}", "逐步解答："),
    "th": ("โจทย์: {}", "คำตอบทีละขั้นตอน:"),
    "te": ("ప్రశ్న: {}", "దశలవారీగా సమాధానం:"),
    "sw": ("Swali: {}", "Jibu la Hatua kwa Hatua:"),
    "bn": ("প্রশ্ন: {}", "ধাপে ধাপে উত্তর:"),
}

exact_match = evaluate.load("exact_match")

def main(args):
    random.seed(42)

    print("Loading data...")
    data = {}
    for lang in ["en", "es", "de", "fr", "ja", "ru", "zh", "th", "te", "sw", "bn"]:
        with open(os.path.join(args.data_dir, f"mgsm_{lang}.tsv")) as fin:
            df = pd.read_csv(fin, sep="\t", header=None, names=["question", "answer"])
            data[lang] = df.to_dict(orient="records")
            if args.max_num_examples_per_lang and len(data[lang]) > args.max_num_examples_per_lang:
                data[lang] = random.sample(data[lang], args.max_num_examples_per_lang)
        
    print("Loading model and tokenizer...")
    model, tokenizer = load_hf_lm_and_tokenizer(
        model_name_or_path=args.model_name_or_path, 
        tokenizer_name_or_path=args.tokenizer_name_or_path, 
        load_in_8bit=args.load_in_8bit, 
        load_in_half=True,
        gptq_model=args.gptq
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    metrics = {}
    for lang in tqdm.tqdm(data, desc="Evaluating"):
        print(f"Language: {lang}")
        test_data = data[lang]
        examplars = list(MGSM_EXEMPLARS[lang].values())

        if args.n_shot and len(examplars) > args.n_shot:
            examplars = random.sample(examplars, args.n_shot)
        demonstration_prompt = "\n\n".join([f"{examplar['q']}\n{examplar['a']}" for examplar in examplars])

        q_template, a_template = cot_question_encoding_templates[lang]

        prompts = []
        for example in test_data:
            question = q_template.format(example["question"])
            if args.use_chat_format:
                prompt = "<|user|>\n" + demonstration_prompt + "\n\n" + question.strip() + "\n<|assistant|>\n" + a_template
            else:
                prompt = demonstration_prompt + "\n\n" + question.strip() + "\n" + a_template
            prompts.append(prompt)

        new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            stop_id_sequences=[[new_line_token]]
        )

        predictions = []
        for output in outputs:
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
            if numbers:
                predictions.append(numbers[-1])
            else:
                predictions.append(output)
            
        print("Calculating accuracy...")
        targets = [example["answer"] for example in test_data]

        em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
        metrics[lang] = em_score
        print(f"Exact match for {lang}: {em_score}")

        predictions = [{
            "question": example["question"],
            "answer": example["answer"],
            "model_output": output,
            "prediction": pred
        } for example, output, pred in zip(test_data, outputs, predictions)]

        with open(os.path.join(args.save_dir, f"{lang}_predictions.jsonl"), "w") as fout:
            for prediction in predictions:
                fout.write(json.dumps(prediction) + "\n") 
    
    metrics["average"] = sum(metrics.values()) / len(metrics)
    print(metrics)
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(metrics, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--max_num_examples_per_lang", type=int, default=None, help="maximum number of examples per language to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    parser.add_argument("--model_name_or_path", type=str, default="../hf_llama_models/7B")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=2, help="max number of examples to use for demonstration.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()
    main(args)
