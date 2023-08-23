import argparse
import json
import logging
import re
import string
from collections import Counter

import evaluate
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_score(metric_name, prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    logger.info(f"Prediction: {prediction}")
    logger.info(f"Ground Truth: {ground_truth}")

    if metric_name == "f1":
        return f1_score(prediction, ground_truth)
    elif metric_name == "exact_match":
        metric = evaluate.load("exact_match")
    elif metric_name == "rouge":
        metric = evaluate.load("rouge")
        result = metric.compute(predictions=[prediction], references=[ground_truth])
        return result["rouge1"]
    elif metric_name == "bleu":
        metric = evaluate.load("bleu")
    elif metric_name == "bertscore":
        metric = evaluate.load("bertscore")
        result = metric.compute(
            predictions=[prediction], references=[ground_truth], lang="en"
        )
        return result["f1"]
    else:
        raise NotImplementedError

    result = metric.compute(predictions=[prediction], references=[ground_truth])
    return result[metric_name]


def mean_reciprocal_rank(predictions, ground_truths):
    """
    Each entry in predictions is top 4 results retrieved for each query
    predictions = [
        ["answer1", "answer2", "answer3", "answer4"],  # Ordered by ranking, first is best
        ["answer5", "answer6", "answer7", "answer8"],
    ]

    ground_truths = [
        ["answer2"],
        ["answer7"],
    ]
    """
    q = len(predictions)

    reciprocal = 0
    for i in range(q):
        try:
            first_result = predictions.index(ground_truths[i][0])
            reciprocal += 1 / (first_result + 1)
            logger.info(f"Reciprocal Rank for query {i}: {1 / (first_result + 1)}")
        except ValueError:  # There is no relevant texts for that query
            continue

    score = reciprocal / q
    logger.info(f"Mean Reciprocal Rank score: {score}")
    with open("eval_result_ms_marco.txt", "a") as f:
        f.write(f"Mean Reciprocal Rank score: {score}\n")


def get_precision_at_k(preds_path, gold_data_path, k=4):
    with open(gold_data_path, "r") as f:
        gt_data = [json.loads(line) for line in f]

    with open(preds_path, "r") as f:
        pred_data = [json.loads(line) for line in f]

    em = total = 0
    for hypo, reference in tqdm(zip(pred_data, gt_data)):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t"))
        total += 1
        em += len(hypo_provenance & ref_provenance) / k

    em = 100.0 * em / total
    logger.info(f"Precision@{k}: {em: .2f}")


def get_scores(metric_name, preds_path, gold_data_path):
    with open(gold_data_path, "r") as f:
        gt_data = [json.loads(line) for line in f]

    with open(preds_path, "r") as f:
        pred_data = [json.loads(line) for line in f]

    score = total = 0

    for pred, gtruth in tqdm(zip(pred_data, gt_data)):
        total += 1
        logger.info(f"Calculating score for query {total}")

        if isinstance(pred["predictions"], list):
            pred["predictions"] = pred["predictions"][0]["context"]

        tmp_score = calculate_score(
            metric_name=metric_name,
            prediction=pred["predictions"],
            ground_truth=gtruth["ground_truths"][0],
        )
        if isinstance(tmp_score, list):
            tmp_score = tmp_score[0]
        score += tmp_score

    logger.info(f"{metric_name} score: {score / total}")
    with open("eval_result_ms_marco.txt", "a") as f:
        f.write(f"{metric_name}: {score / total}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_mode",
        choices=["generator", "retrieval"],
        default="retrieval",
        type=str,
        help=(
            "Evaluation mode, generator calculates bleu-1 and rouge-1, retrieval calculates"
            " exact match, F1, and mean reciprocal rank"
        ),
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Name of the predictions file",
    )

    args = parser.parse_args()
    return args


def main(args):
    logger.info("***** Running evaluation*****")

    # if os.path.exists(args.predictions_path) and (not args.recalculate):
    #     logger.info("Calculating metrics based on an existing predictions file: {}".format(args.predictions_path))
    #     score_fn(args, args.predictions_path, args.gold_data_path)

    if args.eval_mode == "retrieval":
        get_scores(
            metric_name="f1",
            preds_path=args.predictions_path,
            gold_data_path=args.gold_data_path,
        )
        get_scores(
            metric_name="exact_match",
            preds_path=args.predictions_path,
            gold_data_path=args.gold_data_path,
        )
        mean_reciprocal_rank(args.predictions_path, args.gold_data_path)
    elif args.eval_mode == "generator":
        get_scores(
            metric_name="bleu",
            preds_path=args.predictions_path,
            gold_data_path=args.gold_data_path,
        )
        get_scores(
            metric_name="rouge",
            preds_path=args.predictions_path,
            gold_data_path=args.gold_data_path,
        )
        get_scores(
            metric_name="bertscore",
            preds_path=args.predictions_path,
            gold_data_path=args.gold_data_path,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = get_args()
    main(args)
