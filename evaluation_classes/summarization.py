
from evaluation_classes.eval_base_class import Eval
from rouge_score import rouge_scorer


class Summarization(Eval):
    def __init__(self, id_key, predictions_dir, out_path):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        super().__init__(id_key, predictions_dir, out_path)

    def _evaluate(self, predictions, model_name, sample_index):
        predictions_only = [self.get_only_response(sample) for sample in predictions]
        target_only = [sample["target"] for sample in predictions]
        rouge_scores = [self.rouge.score(pred, target) for pred, target in zip(predictions_only, target_only)]
        metrics = rouge_scores[0].keys()
        results = {metric: [score[metric].fmeasure for score in rouge_scores] for metric in metrics}
        return results


