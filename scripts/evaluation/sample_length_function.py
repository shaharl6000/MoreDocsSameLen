import os
from argparse import ArgumentParser
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hmean
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.metrics import r2_score

from scripts.evaluation.compare_models import tasks_order
TOKENS_TRUNCATED = 7500
# LABELS = {'rouge1': 'R-1', 'rouge2': 'R-2', 'rougeL': 'R-L', 'faithfulness_score': 'faithfulness',
#           'coverage_score': 'coverage', 'all_f1': 'F1', 'Roug'}

COLOR_MAP = {'FuseReviews': 'blue', 'MusiQue': 'red', 'ECB': 'green', 'SciCo': 'purple', 'MultiNews': 'orange', 'OpenASP': 'brown'}
MARKER_MAP = {'FuseReviews': ',', 'MusiQue': '*', 'MultiNews': '.', 'OpenASP': '+'}


def eval_length(res_dir, model_name):
    deg = 3
    # plt.figure(figsize=(9, 6))
    for task in tasks_order:
        task_dir = os.path.join(res_dir, task)
        with open(os.path.join(task_dir, "results.json"), 'rt') as f:
            results = json.load(f)
        scores = results["models"][model_name]["scores"]
        metrics = scores[0].keys()
        mean = []
        for metric in metrics:
            metric_scores = np.array([score[metric] for score in scores])
            mean.append(metric_scores)
        if task in ["OpenASP", "MultiNews"]:
            # geometric mean for rouge
            mean = np.prod(mean, axis=0) ** (1 / 3)
            name = 'ROUGE'
            scores = [{name: mean[i]} for i in range(len(mean))]
        elif task == 'FuseReviews':
            mean = hmean(mean, axis=0)
            name = 'F1'
            scores = [{name: mean[i]} for i in range(len(mean))]
        elif task == 'MusiQue':
            scores = [{'F1': s['all_f1']} for s in scores]
        metrics = scores[0].keys()
        lengths = results["sample_lengths"]
        sorted_by_lengths = sorted(lengths.keys(), key=lambda x: lengths[x])
        sorted_lengths = np.array(sorted(lengths.values()))
        valid_indices = np.where(np.array(sorted_lengths) <= TOKENS_TRUNCATED)[0]
        for metric in metrics:
            if metric.lower() == 'conll f1':
                continue
            mean_scores_per_sample = []
            stds = []
            metric_scores = np.array([score[metric] for score in scores])
            # normalize scores to be in [0, 100]
            metric_scores = (metric_scores - np.min(metric_scores)) / (np.max(metric_scores) - np.min(metric_scores))
            if metric_scores.ndim == 1:
                continue
            for sample_id in sorted_by_lengths:
                total_for_sample = []
                for run_index in range(len(metric_scores)):
                    if sample_id in results["models"][model_name]['ids'][run_index]:
                        index = results["models"][model_name]['ids'][run_index].index(sample_id)
                        score = metric_scores[run_index, index]
                        total_for_sample.append(score)
                mean_scores_per_sample.append(np.mean(total_for_sample))
                stds.append(np.std(total_for_sample))
            # sampled_valid_indices = np.random.choice(valid_indices, 100)
            valid_lengths = np.array(sorted_lengths)[valid_indices]
            args_sorted = np.argsort(valid_lengths)
            valid_lengths = valid_lengths[args_sorted]
            valid_scores = np.array(mean_scores_per_sample)[valid_indices]
            valid_scores = valid_scores[args_sorted]

            # remove duplicates from valid lengths, and mean over the duplicates scores in valid_scores
            # unique_lengths = np.unique(valid_lengths)
            # unique_scores = []
            # for length in unique_lengths:
            #     indices = np.where(valid_lengths == length)
            #     unique_scores.append(np.mean(valid_scores[indices]))



            # Assuming you have your data points in separate lists named 'x' and 'y'

            # # Plot the data points
            # plt.plot(unique_lengths, unique_scores, 'o', label=f'{task} {metric}')
            #
            # # Perform cubic spline interpolation
            # spline = CubicSpline(unique_lengths, unique_scores)
            #
            # # Generate smoother curve using spline interpolation
            # smooth_x = np.linspace(min(unique_lengths), max(unique_lengths), 100)  # Adjust number of points for smoothness
            # smooth_y = spline(smooth_x)
            #
            # # Plot the spline curve
            # plt.plot(smooth_x, smooth_y, label='Spline Interpolation')
            #
            # # Calculate and display R-squared
            # y_pred = spline(unique_lengths)  # Predicted values using spline
            # r2 = r2_score(unique_scores, y_pred)
            # print(f"R-squared: {r2}")
            #
            # # Add labels and title
            # plt.xlabel("Sample length (# tokens)")
            # plt.ylabel("Score")
            # plt.title("Data with Spline Interpolation")
            # plt.legend()
            # plt.grid(True)
            # plt.show()




            coefficients = np.polyfit(valid_lengths, valid_scores, deg=deg)
            polynomial = np.poly1d(coefficients)

            # Generate fitted line
            x_fit = np.linspace(min(valid_lengths), max(valid_lengths), 100)
            y_fit = polynomial(x_fit)

            label = f'{task} {metric}'

            # Plot fitted line
            # plt.plot(x_fit, y_fit, color=COLOR_MAP[task], alpha=0.5)
            plt.scatter(valid_lengths, valid_scores, label=label, alpha=0.5, marker=MARKER_MAP[task], color=COLOR_MAP[task])
            # plt.show()
    # plt.title("Data with Spline Interpolation")
    # plt.legend()
    # plt.grid(True)
    # plt.show()





    plt.xlabel("Sample length (# tokens)", fontsize=15)
    plt.ylabel("Relative score", fontsize=15)
    plt.xticks(fontsize=14)
    plt.legend(loc='upper right')
    plt.tight_layout()

    path = os.path.join(res_dir, f"polyfit_{model_name}.pdf")
    # Show plot
    plt.savefig(path)
    # plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    eval_length(args.results_dir, args.model_name)