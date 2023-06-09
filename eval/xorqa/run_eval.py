import argparse
import os
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.utils import generate_completions, load_hf_lm_and_tokenizer
from eval.xorqa.compute_metrics import calculate_f1_em_bleu


encoding_templates_for_langs = {
    "te": ("ప్రశ్న: {}", "సమాధానం:"),  # Telugu
    "sw": ("Swali: {}", "Jibu:"),  # Swahili
    "th": ("คำถาม: {}\nคำตอบ:"),  # Thai
    "fi": ("Kysymys: {}", "Vastaus:"),  # Finnish
    "id": ("Pertanyaan: {}", "Jawaban:"),  # Indonesian
    "ja": ("質問: {}", "回答:"),  # Japanese
    "ru": ("Вопрос: {}", "Ответ:"),  # Russian
    "ar": ("السؤال: {}", "الجواب:"),  # Arabic
    "en": ("Question: {}", "Answer:"),  # English
    "bn": ("প্রশ্ন: {}", "উত্তর:"),  # Bengali
    "ko": ("질문: {}", "답변:"),  # Korean
    "es": ("Pregunta: {}", "Respuesta:"),  # Spanish
    "he": ("שאלה: {}","תשובה:"),  # Hebrew
    "sv": ("Fråga: {}", "Svar:"),  # Swedish
    "da": ("Spørgsmål: {}", "Svar:"),  # Danish
    "de": ("Frage: {}", "Antwort:"),  # German
    "hu": ("Kérdés: {}", "Válasz:"),  # Hungarian
    "it": ("Domanda: {}", "Risposta:"),  # Italian
    "km": ("សំណួរ: {}", "ចម្លើយ:"),  # Khmer
    "ms": ("Soalan: {}", "Jawapan:"),  # Malay
    "nl": ("Vraag: {}", "Antwoord:"),  # Dutch
    "no": ("Spørsmål: {}", "Svar:"),  # Norwegian
    "pt": ("Pergunta: {}", "Resposta:"),  # Portuguese
    "tr": ("Soru: {}", "Cevap:"),  # Turkish
    "vi": ("Câu hỏi: {}", "Trả lời:"),  # Vietnamese
    "fr": ("Question: {}", "Réponse:"),  # French
    "pl": ("Pytanie: {}", "Odpowiedź:"),  # Polish
    "zh-CN": ("问题：{}", "回答："),  # Simplified Chinese
    "zh-HK": ("問題：{}", "答案："),  # Hong Kong Chinese
    "zh-TW": ("問題：{}", "答案："),  # Traditional Chinese
    "ta": ("கேள்வி: {}", "பதில்:"),  # Tamil
    "tl": ("Tanong: {}", "Sagot:")  # Tagalog
}


def main(args):
    random.seed(42)

    print("Loading data...")

    test_data_file = os.path.join(args.data_dir, "mia_2022_dev_xorqa.jsonl")    
    with open(test_data_file) as fin:
        test_data = [json.loads(line) for line in fin]
        if args.max_num_examples_per_lang:
            sampled_test_data = []
            for lang in encoding_templates_for_langs.keys():
                test_data_for_lang = [example for example in test_data if example["lang"] == lang]
                if len(test_data_for_lang) > args.max_num_examples_per_lang:
                    test_data_for_lang = random.sample(test_data_for_lang, args.max_num_examples_per_lang)
                sampled_test_data += test_data_for_lang
            test_data = sampled_test_data
        # remove languages that have no examples
        examples_for_langs = {lang: [example for example in test_data if example["lang"] == lang] for lang in encoding_templates_for_langs.keys()}
        test_data = [example for example in test_data if len(examples_for_langs[example["lang"]]) > 0]
        test_languages = set([example["lang"] for example in test_data])
        print(f"Loaded {len(test_data)} examples from {len(test_languages)} languages: {test_languages}")

    if args.n_shot > 0:
        train_data_file = os.path.join(args.data_dir, "mia_2022_train_data.jsonl")
        with open(train_data_file) as fin:
            train_data = [json.loads(line) for line in fin]
        train_data_for_lang = {
            lang: [example for example in train_data if example["lang"] == lang and not "has_eng_answer_only" in example] \
                for lang in test_languages}
        print("Sizes for each language in train data:")
        for lang, data in train_data_for_lang.items():
            print(f"{lang}: {len(data)}")
        sampled_train_data_for_lang = {}
        for lang, data in train_data_for_lang.items():
            assert len(data) >= args.n_shot, f"Language {lang} has {len(data)} examples, which is less than {args.n_shot}."
            sampled_train_data_for_lang[lang] = random.sample(data, args.n_shot)
        train_data_for_lang = sampled_train_data_for_lang

    
        
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

    prompts = []
    for example in test_data:
        lang = example["lang"]
        q_template, a_template = encoding_templates_for_langs[lang]
        prompt = ""
        if args.n_shot > 0:
            formatted_demo_examples = []
            for train_example in train_data_for_lang[lang]:
                formatted_demo_examples.append(
                    q_template.format(train_example["question"]) + "\n" + a_template + " " + train_example["answers"][0]
                )
            prompt += "\n\n".join(formatted_demo_examples) + "\n\n"
        
        prompt += q_template.format(example["question"]) + "\n"
        if args.use_chat_format:
            prompt = "<|user|>\n" + prompt.strip() + "\n<|assistant|>\n" + a_template
        else:
            prompt += a_template
        prompts.append(prompt)

    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=50,
        batch_size=args.eval_batch_size,
        stop_id_sequences=[[new_line_token]]
    )

    predictions = {
        example["id"]: output for example, output in zip(test_data, outputs)
    }

    with open(os.path.join(args.save_dir, "xorqa_predictions.jsonl"), "w") as fout:
        for prediction in predictions.values():
            fout.write(json.dumps(prediction) + "\n")

    print("Calculating F1, EM, and BLEU...")
    eval_scores = calculate_f1_em_bleu(dataset=test_data, predictions=predictions)

    eval_scores = {lang: scores for lang, scores in eval_scores.items() if lang in test_languages}
    eval_scores["average"] = {metric: sum([scores[metric] for scores in eval_scores.values()]) / len(eval_scores) for metric in eval_scores["ja"].keys()}

    print("Scores:")
    print(json.dumps(eval_scores, indent=2))
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(eval_scores, fout, indent=2)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/xorqa/")
    parser.add_argument("--max_num_examples_per_lang", type=int, default=None, help="maximum number of examples per language to evaluate.")
    parser.add_argument("--n_shot", type=int, default=0, help="number of examples to use for few-shot evaluation.")
    parser.add_argument("--save_dir", type=str, default="results/xorqa")
    parser.add_argument("--model_name_or_path", type=str, default="../hf_llama_models/7B")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()
    main(args)
