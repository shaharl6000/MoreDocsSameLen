import os.path
from argparse import ArgumentParser
from evaluation_classes.fuse_reviews import FuseReviews
from evaluation_classes.question_answering import QA
from evaluation_classes.coreference import Coref
from evaluation_classes.summarization import Summarization


def main(predictions_dir, output_path, ds_name):
    if ds_name == 'FuseReviews':
        eval_class = FuseReviews("guid", predictions_dir, output_path)
    elif ds_name == 'MusiQue':
        eval_class = QA("id", predictions_dir, output_path)
    elif ds_name == 'ECB':
        eval_class = Coref("topic_id", predictions_dir, output_path)
    elif ds_name == 'SciCo':
        eval_class = Coref("sample_id", predictions_dir, output_path)
    elif ds_name == 'MultiNews':
        eval_class = Summarization("id", predictions_dir, output_path)
    elif ds_name == 'OpenASP':
        eval_class = Summarization("guid", predictions_dir, output_path)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    eval_class.eval_all_in_dir()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ds_name", type=str, required=True)
    args = parser.parse_args()
    dir_out = os.path.dirname(args.output_path)
    os.makedirs(dir_out, exist_ok=True)
    main(args.predictions_dir, args.output_path, args.ds_name)
