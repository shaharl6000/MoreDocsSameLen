
import os.path
import sys
from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ASD import calc_dominance
from itertools import product
import seaborn as sns
from scipy.stats import hmean

TOKENS_TRUNCATED = 7500
COLORS = ['plum', 'mediumpurple', 'cadetblue', 'cornflowerblue', 'coral', 'palevioletred', 'peru']

def analyze_results(results_path):
    dataset_name = os.path.dirname(results_path).split("/")[-1]
    out_dir = os.path.dirname(results_path)
    figs_dir = os.path.join(out_dir, "figs")
    scores_path = os.path.join(out_dir, "scores.txt")
    scores_file = open(scores_path, "w")
    os.makedirs(figs_dir, exist_ok=True)
    with open(results_path, 'r') as f:
        results = json.load(f)
    lengths = results["sample_lengths"]
    sorted_by_lengths = sorted(lengths.keys(), key=lambda x: lengths[x])
    sorted_lengths = np.array(sorted(lengths.values()))
    # remove outlier lengths
    valid_indices = np.where(np.array(sorted_lengths) <= TOKENS_TRUNCATED)[0]
    total = {}
    vectors = {}
    for_box_plot = {}
    for model in results["models"]:
        plt.figure()
        print(f"Model: {model}", file=scores_file)
        scores = results["models"][model]["scores"]
        metrics = scores[0].keys()
        total[model] = {}
        if len(list(metrics)) > 1:
            # calc geometric mean per sample
            mean = []
            for metric in metrics:
                metric_scores = np.array([score[metric] for score in scores])
                mean.append(metric_scores)
            if 'rouge' in list(metrics)[0].lower():
                # geometric mean for rouge
                mean = np.prod(mean, axis=0) ** (1 / len(metrics)) * 100
                name = 'ROUGE'
            elif 'faithfulness' in list(metrics)[0].lower():
                mean = hmean(mean, axis=0)
                name = 'F1'
            else:
                print()
                raise ValueError("Unknown metric")
            metrics = [name]
            scores = [{name: mean[i]} for i in range(len(mean))]
            results["models"][model]["scores"] = scores
        for metric in metrics:
            if metric not in for_box_plot:
                for_box_plot[metric] = {'models': [], 'scores': []}
            metric_scores = np.array([score[metric] for score in scores])
            if metric_scores.ndim == 1:
                metric_scores = metric_scores[:, None]
            mean_per_run = np.mean(metric_scores, axis=1)
            if metric not in vectors:
                vectors[metric] = {}
            vectors[metric][model] = mean_per_run
            std_between_runs = np.std(mean_per_run)
            total_mean = np.mean(metric_scores)
            # if total_mean < 1:
            #     total_mean *= 100
            #     std_between_runs *= 100
            total[model][metric] = (total_mean, std_between_runs)
            for_box_plot[metric]['models'].append(model)
            for_box_plot[metric]['scores'].append(mean_per_run)
            print(f"\t{metric}: {total_mean} +/- {std_between_runs}", file=scores_file)
            if metric_scores.shape[1] == 1:
                continue
            mean_scores_per_sample = []
            stds = []
            for sample_id in sorted_by_lengths:
                total_for_sample = []
                for run_index in range(len(metric_scores)):
                    if sample_id in results["models"][model]['ids'][run_index]:
                        index = results["models"][model]['ids'][run_index].index(sample_id)
                        score = metric_scores[run_index, index]
                        total_for_sample.append(score)
                mean_scores_per_sample.append(np.mean(total_for_sample))
                stds.append(np.std(total_for_sample))
            # plt.errorbar(sorted_lengths, mean_scores_per_sample, yerr=stds, label=metric)
            plt.scatter(sorted_lengths[valid_indices], np.array(mean_scores_per_sample)[valid_indices], label=metric)
        plt.title(f"{model} scores")
        plt.xlabel("Sample length")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(os.path.join(figs_dir, f'{model.replace("/", "_")}_scores.pdf'))
    f.close()


    for metric in for_box_plot:
        plt.figure(figsize=(6.4, 3.8))
        argsort_by_model_name = np.argsort(for_box_plot[metric]['models'])
        for_box_plot[metric]['models'] = [for_box_plot[metric]['models'][i] for i in argsort_by_model_name]
        for_box_plot[metric]['scores'] = [for_box_plot[metric]['scores'][i] for i in argsort_by_model_name]
        name = metric
        if metric == 'conll_F1':
            name = 'CoNLL F1'
            plt.xlim(-1, 34)
        elif metric == 'ROUGE':
            plt.xlim(-1, 24)
            plt.xticks(np.arange(0, 24, 2))
        else:
            for_box_plot[metric]['scores'] = [np.array(scores) * 100 for scores in for_box_plot[metric]['scores']]
        bplot = plt.boxplot(for_box_plot[metric]['scores'], labels=for_box_plot[metric]['models'],
                            vert=False, patch_artist=True)

        # fill with colors
        for patch, color in zip(bplot['boxes'], COLORS):
            patch.set_facecolor(color)

        for line in bplot['medians']:
            line.set_color("black")

        if dataset_name == 'ECB':
            dataset_name = 'ECB+'
        xlabel = f"{dataset_name} {name}"
        plt.xlabel(xlabel, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, f"box_plot_{dataset_name}_{metric}.pdf"))

    total_df = pd.DataFrame(total).T
    total_df.to_csv(os.path.join(out_dir, "total_scores.csv"))
    asd_file = open(os.path.join(out_dir, "asd.txt"), "w")
    asd_table = os.path.join(out_dir, "ranking_asd.csv")
    sorted_model_names = sorted(total.keys())
    for metric in vectors:
        asd_file.write(f"\n\n===== {metric} =====\n\n")
        tab = {}
        visited = set()
        for a, b in product(total, total):
            if a == b or (b, a) in visited:
                continue
            dom = calc_dominance(data_A=vectors[metric][a], data_B=vectors[metric][b], alpha=0.05,
                                 name_A=a, name_B=b, out_file=asd_file)
            if a not in tab:
                tab[a] = {}
            if b not in tab:
                tab[b] = {}
            tab[a][b] = round(dom, 2)
            tab[b][a] = 1 - round(dom, 2)
            visited.add((a, b))
            asd_file.write("\n")
        plt.figure()
        sum_dom = {k: sum(tab[k].values()) for k in tab}
        ranking_model_names = sorted(sum_dom, key=lambda x: sum_dom[x])
        one_minus_alpha = [1 - tab[ranking_model_names[i]][ranking_model_names[i+1]] for i in range(len(ranking_model_names)-1)]
        one_minus_alpha.append(None)
        ranking_df = pd.DataFrame(columns=["model_name", "dominance"])
        ranking_df["model_name"] = ranking_model_names
        ranking_df["dominance"] = one_minus_alpha
        scores = {model: (round(total_df.loc[model, metric][0] * 100, 1), round(total_df.loc[model, metric][1] * 100, 1))
                  for model in ranking_model_names}
        sorted_scores = [scores[model] for model in ranking_model_names]
        ranking_df["score"] = sorted_scores
        ranking_df.to_csv(asd_table)
    asd_file.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()
    analyze_results(args.results_path)

