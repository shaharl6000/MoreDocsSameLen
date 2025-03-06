
# import fic_evaluation
from evaluation_classes.eval_base_class import Eval
import torch
import gc


class FuseReviews(Eval):
    def __init__(self, id_key, predictions_dir, out_path):
        super().__init__(id_key, predictions_dir, out_path)

    @staticmethod
    def get_faithfulness(predictions, only_responses):
        faithfulness_metric = fic_evaluation.HighlightsFaithfulnessEvaluator()
        faithfulness_results = faithfulness_metric.evaluate(
            concatenated_highlights=[sample["highlights_concat"] for sample in predictions],
            predictions=only_responses)
        return faithfulness_results

    @staticmethod
    def get_coverage(predictions, only_responses):
        coverage_metric = fic_evaluation.HighlightsCoverageEvaluator()
        coverage_results = coverage_metric.evaluate(
            review_side_alignments=[sample["review_side_alignments"] for sample in predictions],
            predictions=only_responses)
        return coverage_results

    def _evaluate(self, predictions, model_name, sample_index):
        only_responses = [self.get_only_response(s) for s in predictions]
        try:
            faithfulness_results = self.get_faithfulness(predictions, only_responses)
        except Exception as e:
            # print(predictions)
            raise e
        coverage_results = self.get_coverage(predictions, only_responses)
        out = faithfulness_results | coverage_results
        return out

