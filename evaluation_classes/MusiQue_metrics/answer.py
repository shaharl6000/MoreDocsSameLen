"""
Answer metric -- mostly taken directly from squad_tools of allennlp.
"""
import re
import string
import collections
from typing import Tuple, List
import ast
from evaluation_classes.MusiQue_metrics.metric import Metric
import json

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def extract_content_from_gt(input_string):
    input_dict = ast.literal_eval(input_string)

    # Extracting the value of 'answer_content'
    answer_content = input_dict['answer_content']

    return answer_content
def extract_content_pred(input_string):
    match = re.search(r"'answer_content':\s*'(.*?)'", input_string)

    if match:
        answer_content = match.group(1)
    else:
        answer_content = input_string

    return answer_content
def extract_format(input_string):
    input_string = re.sub(r"'is_answerable':\s*\w+,\s*", "", input_string)

    # Using regular expression to remove 'answer_content' and its value
    input_string = re.sub(r"'answer_content':\s*'[^']*'", "", input_string)

    # Removing any leading or trailing commas and whitespace
    cleaned_string = input_string.strip().strip(',')

    return cleaned_string


def compute_f1_with_content(a_gold, a_pred):
    # a = {'is_answerable': True, 'answer_content': 'Arna Selznick'}
    a_gold_content = a_gold
    a_pred_content = a_pred
    try:
        temp_gt = a_gold
        temp_pred = a_pred

        temp_gt = temp_gt.replace("'", '"').replace("True", "true").replace("False", "false")
        temp_pred = temp_pred.replace("'", '"').replace("True", "true").replace("False", "false")

        temp_gt = json.loads(temp_gt)
        temp_pred = json.loads(temp_pred)

        a_gold_content = str(temp_gt["answer_content"])
        a_pred_content = str(temp_pred["answer_content"])
    except:
        a_gold_content = a_gold_content.replace("{", "").replace("is_answerable","").replace('answer_content',"").replace('answer_content',"").replace('True',"").replace(',', "")
        a_gold_content = a_gold_content.replace('False',"").replace('}',"").replace('answer_content',"").replace(':',"")

        a_pred_content = a_pred_content.replace("{", "").replace("is_answerable", "").replace('answer_content', "").replace(
            'answer_content', "").replace('True', "")
        a_pred_content = a_pred_content.replace('False', "").replace('}', "").replace('answer_content', "").replace(':', "").replace(',', "")

        a_gold_content = " ".join([word for word in a_gold_content.split() if word != "''"])
        a_pred_content = " ".join([word for word in a_pred_content.split() if word != "''"])

    gold_toks = get_tokens(a_gold_content)
    pred_toks = get_tokens(a_pred_content)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1_with_format(a_gold, a_pred):
    # a = {'is_answerable': True, 'answer_content': 'Arna Selznick'}
    content_gt = a_gold.replace("{", "").replace("is_answerable", "").replace('answer_content', "").replace(
        'answer_content', "").replace('True', "")
    content_gt = content_gt.replace('False', "").replace('}', "").replace('answer_content', "").replace(':', "")

    content_pred = a_pred.replace("{", "").replace("is_answerable", "").replace('answer_content', "").replace(
        'answer_content', "").replace('True', "")
    content_pred = content_pred.replace('False', "").replace('}', "").replace('answer_content', "").replace(':', "")

    content_pred = " ".join([word for word in content_pred.split() if word != "''"])
    content_gt = " ".join([word for word in content_gt.split() if word != "''"])

    a_gold_u = " ".join([word for word in a_gold.split() if word not in content_gt.split()])
    a_pred_u = " ".join([word for word in a_pred.split() if word not in content_pred.split()])

    for www in content_gt.split(" "):
        a_gold_u = a_gold_u.replace(www, "")

    for www in content_pred.split(" "):
        a_pred_u = a_pred_u.replace(www, "")

    a_gold_u = a_gold_u.replace('True', "").replace('False', "")
    a_pred_u = a_pred_u.replace('True', "").replace('False', "")

    a_gold_u = " ".join([word for word in a_gold_u.split() if word != "''"])
    a_pred_u = " ".join([word for word in a_pred_u.split() if word != "''"])

    gold_toks = get_tokens(a_gold_u)
    pred_toks = get_tokens(a_pred_u)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1_is_answerable(a_gold, a_pred):
    # a = {'is_answerable': True, 'answer_content': 'Arna Selznick'}
    content_gt = a_gold.replace("{", "").replace("is_answerable", "").replace('answer_content', "").replace(
        'answer_content', "").replace('True', "")
    content_gt = content_gt.replace('False', "").replace('}', "").replace('answer_content', "").replace(':', "")

    content_pred = a_pred.replace("{", "").replace("is_answerable", "").replace('answer_content', "").replace(
        'answer_content', "").replace('True', "")
    content_pred = content_pred.replace('False', "").replace('}', "").replace('answer_content', "").replace(':', "")

    content_gt = " ".join([word for word in content_gt.split() if word != "''"])
    content_pred = " ".join([word for word in content_pred.split() if word != "''"])

    a_gold_u = " ".join([word for word in a_gold.split() if word not in content_gt.split()])
    a_pred_u = " ".join([word for word in a_pred.split() if word not in content_pred.split()])

    for www in content_gt.split(" "):
        a_gold_u = a_gold_u.replace(www, "")

    for www in content_pred.split(" "):
        a_pred_u = a_pred_u.replace(www, "")

    a_gold_u = a_gold_u.replace("{", "").replace("is_answerable", "").replace(
        'answer_content', "").replace('}', "").replace(':', "").replace(',', "")
    a_pred_u = a_pred_u.replace("{", "").replace("is_answerable", "").replace(
        'answer_content', "").replace('}', "").replace(':', "").replace(',', "")

    a_gold_u = " ".join([word for word in a_gold_u.split() if word != "''"])
    a_pred_u = " ".join([word for word in a_pred_u.split() if word != "''"])

    gold_toks = get_tokens(a_gold_u)
    pred_toks = get_tokens(a_pred_u)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AnswerMetric(Metric):
    def __init__(self) -> None:
        # self._total_em = 0.0
        # self._total_f1 = 0.0
        self._count = 0
        self._em = []
        self._f1 = []
        self._f1_content = []
        self._f1_format = []
        self._f1_is_answerable = []
    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):

        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        self._em.append(exact_scores)
        f1_scores = metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )
        self._f1.append(f1_scores)
        f1_scores_with_format = metric_max_over_ground_truths(
            compute_f1_with_format, predicted_answer, ground_truth_answers
        )

        self._f1_format.append(f1_scores_with_format)

        f1_scores_with_content = metric_max_over_ground_truths(
            compute_f1_with_content, predicted_answer, ground_truth_answers
        )
        self._f1_content.append(f1_scores_with_content)

        f1_scores_is_answerable = metric_max_over_ground_truths(
            compute_f1_is_answerable, predicted_answer, ground_truth_answers
        )
        self._f1_is_answerable.append(f1_scores_is_answerable)



        # self._total_em += int(exact_scores)
        # self._total_f1 += f1_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float, list, list]:
        exact_match = sum(self._em) / self._count if self._count > 0 else 0
        f1_score = sum(self._f1) / self._count if self._count > 0 else 0
        f1_score_with_content = sum(self._f1_content) / self._count if self._count > 0 else 0
        f1_score_with_format = sum(self._f1_format) / self._count if self._count > 0 else 0
        f1_score_is_answerable = sum(self._f1_is_answerable) / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score,f1_score_with_content, f1_score_with_format, f1_score_is_answerable,  self._em, self._f1,  self._f1_content, self._f1_format, self._f1_is_answerable

    def reset(self):
        # self._total_em = 0.0
        self._em = []
        # self._total_f1 = 0.0
        self._f1 = []
        self._count = 0


if __name__ == "__main__":
    def compute_f1(a_gold, a_pred):
        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    a_pred = "Tian Yunzhang"
    a_gold = "Tianjin"
    print(compute_f1(a_gold= a_gold, a_pred = a_pred))