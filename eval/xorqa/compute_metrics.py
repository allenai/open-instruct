import jsonlines
import json
from nltk.translate import bleu
import MeCab
from collections import Counter
import string
import argparse

wakati = MeCab.Tagger("-Owakati")

lang_dic = {'telugu': 'te', 'swahili': 'sw', 'thai': 'th', 'finnish': 'fi', 'indonesian': 'id',
            'japanese': 'ja', 'russian': 'ru', 'arabic': 'ar', 'english': 'en', 'bengali': 'bn',
            "korean": "ko", "spanish": "es", "hebrew": "he", "swedish": "sv", "danish": "da", "german": "de",
            "hungarian": "hu", "italian": "it", "khmer": "km", "malay": "ms", "dutch": "nl",
            "norwegian": "no", "portuguese": "pt", "turkish": "tr", "vietnamese": "vi", "french": "fr", "polish": "pl",
            "chinese (simplified)": "zh_cn",  "chinese (hong kong)": 'zh_hk', "chinese (traditional)": "zh_tw", "tamil": "ta", "tagalog": "tl"}


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def load_tydi_answer(tydi_eval_open_domain_data):
    answer_dict = {}
    eval_data = read_jsonlines(tydi_eval_open_domain_data)
    for item in eval_data:
        answer_dict[item["id"]] = item["answers"]
    return answer_dict


def normalize_answer(s):
    # TODO: should we keep those counter removal?
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# 3. XOR-Full Evaluation
def calculate_f1_em_bleu(dataset, predictions):
    lang_dict = {lang: {"count": 0, "f1": 0, "bleu": 0, "em": 0}
                 for lang in lang_dic.values()}

    for qa in dataset:
        lang = qa["lang"]
        q_id = qa["id"]
        gts = qa["answers"]
        if gts[0] == "No Answer":
            continue
        lang_dict[lang]["count"] += 1
        if q_id not in predictions:
            print(q_id)
            print("no answers")
            continue
        pred = predictions[q_id]
        if isinstance(gts, str):
            gts = [gts]

        final_gts = []
        # for japanese, we need to tokenize the input as there are no white spaces.
        if lang == "ja":
            for gt in gts:
                gt = wakati.parse(gt)
                final_gts.append(gt)
            final_pred = wakati.parse(pred.replace("・", " ").replace("、", ","))
        else:
            final_gts = gts
            final_pred = pred
        lang_dict[lang]["f1"] += metric_max_over_ground_truths(
            f1_score, final_pred, final_gts)
        lang_dict[lang]["bleu"] += bleu(final_gts, pred)
        lang_dict[lang]["em"] += metric_max_over_ground_truths(
            exact_match_score, final_pred, final_gts)
    # finalize scores
    for lang, scores in lang_dict.items():
        if scores["count"] == 0:
            continue
        for score_key in scores:
            if "count" != score_key:
                lang_dict[lang][score_key] = scores[score_key]/scores["count"]
    return lang_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file",
                        default=None, type=str)
    parser.add_argument("--pred_file",
                        default=None, type=str)

    args = parser.parse_args()

    dataset = read_jsonlines(args.data_file)
    with open(args.pred_file) as prediction_file:
        predictions = json.load(prediction_file)

    results = calculate_f1_em_bleu(dataset, predictions)

    f1_total, em_total, bleu_total = 0.0, 0.0, 0.0
    total_num = 0
    lang_count = 0
    for lang in results:
        if results[lang]["count"] == 0:
            continue
        lang_count += 1
        f1_total += results[lang]["f1"]
        em_total += results[lang]["em"]
        bleu_total += results[lang]["bleu"]
        total_num += results[lang]["count"]
        print("Evaluating the performance on {0} for {1} examples".format(
            lang, results[lang]["count"]))
        print("F1: {0}, EM:{1}, BLEU:{2}".format(
            results[lang]["f1"] * 100, results[lang]["em"] * 100, results[lang]["bleu"] * 100))
    print("avg f1: {}".format(f1_total / lang_count * 100))
    print("avg em: {}".format(em_total / lang_count * 100))
    print("avg bleu: {}".format(bleu_total / lang_count * 100))


if __name__ == "__main__":
    main()