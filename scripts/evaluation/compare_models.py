import os
from argparse import ArgumentParser
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps
from statistical_analysis import COLORS

tasks_order = ['MultiNews', 'OpenASP', 'FuseReviews', 'MusiQue', 'ECB', 'SciCo']


def main(results_dir):
    rankings = []
    models = {}
    dict_std_by_task = {}
    for f_name in glob(f'{results_dir}/*/ranking_asd.csv'):
        rankings.append(pd.read_csv(f_name))
        r = pd.read_csv(f_name)
        task = f_name.split(os.sep)[-2]
        if task not in dict_std_by_task:
            dict_std_by_task[task] = []
        for i, row in r.iterrows():
            model_name = row['model_name']
            model_rank = row['Unnamed: 0']+1
            model_mean, model_variance = eval(row['score'])
            if model_name not in models:
                models[model_name] = {'ranks': [], 'normalized_var': []}
            models[model_name]['ranks'].append(model_rank)
            if model_mean > 0:
                models[model_name]['normalized_var'].append(model_variance/model_mean)
                dict_std_by_task[task].append(model_variance/model_mean)
        print(f_name)
    out_dict_rank = {}
    out_dict_std = {}
    for model_name in models:
        out_dict_rank[model_name] = {}
        out_dict_std[model_name] = {}
        out_dict_rank[model_name]['Rank'] = np.mean(models[model_name]['ranks'])
        out_dict_std[model_name]['Standard Deviation'] = np.mean(models[model_name]['normalized_var'])*100
    out_dict_std_by_task = {task: {} for task in dict_std_by_task}
    for task in out_dict_std_by_task:
        out_dict_std_by_task[task]['Standard Deviation'] = np.mean(dict_std_by_task[task])*100

    # COLORS.reverse()
    output = pd.DataFrame(out_dict_rank)
    output = output[sorted(output.columns)]
    plot(sorted_output=output, COLORS=COLORS, x_label="Average rank (AR ↓)\nacross models\n",
         figs_dir=results_dir, fig_name='ranking')
    output = pd.DataFrame(out_dict_std)
    output = output[sorted(output.columns)]
    plot(sorted_output=output, COLORS=COLORS, x_label="Average relative standard\ndeviation (ARSD ↓) across\nmodels",
         figs_dir=results_dir, fig_name='rsd_for_model')
    output = pd.DataFrame(out_dict_std_by_task)
    output = output[reversed(tasks_order)]
    plot(sorted_output=output, COLORS=None, x_label="Average relative standard\ndeviation (ARSD ↓) across\ndatasets",
         figs_dir=results_dir, fig_name='rsd_for_task')


def plot(sorted_output, COLORS, x_label, figs_dir, fig_name):
    plt.figure(fig_name)
    plt.barh(sorted_output.columns, sorted_output.iloc[0], color=COLORS)
    if fig_name != 'ranking':
        plt.xticks(np.arange(5, 25, 5), fontsize=19)
    else:
        plt.xticks(np.arange(1, 8, 1), fontsize=19)
    plt.xlabel(x_label, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{figs_dir}/{fig_name}.pdf')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.results_dir)