
import subprocess
import numpy as np
from evaluation_classes.eval_base_class import Eval
import re


class Coref(Eval):

    def __init__(self, id_key, predictions_dir, out_path):
        self.correlations = []
        super().__init__(id_key, predictions_dir, out_path)

    def _postprocess(self, predictions):
        all_processed = {}
        correct_format = []
        for sample in predictions:
            sample_id = sample[self.id_key]
            only_response = self.get_only_response(sample)
            all_lists = []
            suffix_truncate = 0
            for it in re.finditer(r'\[[\d, ]+\]', only_response):
                reg = it.span()
                all_lists.append(eval(only_response[reg[0]:reg[1]]))
                suffix_truncate = max(reg[1], suffix_truncate)
            only_response = only_response[suffix_truncate:]
            last_open_list = re.search(r'\[[\d, ]+', only_response)
            if last_open_list is not None:
                reg = last_open_list.regs[0]
                all_lists.append(eval(only_response[reg[0]:reg[1]]+"]"))
            correct_format.append(len(all_lists)>0)
            mentions_to_clusters = {}
            without_singletons = [cluster for cluster in all_lists if len(cluster) > 1]
            for i, cluster in enumerate(without_singletons):
                for mention in cluster:
                    mentions_to_clusters[mention] = i
            all_processed[sample_id] = mentions_to_clusters
        self.correlations.append([np.mean(correct_format)])
        return all_processed

    def process_conll_out(self, out):
        all_scores = {}
        metrics = out.split("METRIC")[1:]
        splitted_metrics = [met.split('\n') for met in metrics]
        names = [re.sub('\W', '', met[0]) for met in splitted_metrics]
        coref_lines = [[line for line in met if 'Coreference' in line] for met in splitted_metrics]
        for i, line in enumerate(coref_lines):
            if len(line) > 1:
                index = splitted_metrics[i].index(line[0])
                for row in splitted_metrics[i][index+1:-1]:
                    if row.startswith("---"):
                        continue
                    metric_name = row[:row.find(':')]
                    scores = row[row.find(':')+2:].split("\t")
                    for score in scores:
                        name = re.search(r'[a-zA-Z]+1?: ', score).group(0)[:-2]
                        res = eval(re.search(r'\d+(\.\d+)?%', score).group(0)[:-1])
                        all_scores[f"{names[i]}_{metric_name}_{name}"] = res
            else:
                scores = re.sub('Coreference: ', '', line[0]).split("\t")
                for score in scores:
                    name = re.search(r'[a-zA-Z]+1?: ', score).group(0)[:-2]
                    res = eval(re.search(r'\d+(\.\d+)?%', score).group(0)[:-1])
                    all_scores[f"{names[i]}_{name}"] = res
        conll_f1 = np.average([all_scores[f"{name}_F1"] for name in ["muc", "bcub", "ceafe"]])
        self.correlations[-1].append(conll_f1)
        output = {"conll_F1": conll_f1}
        return output

    def _evaluate(self, predictions, model_name, sample_index):

        predictions_processed = self._postprocess(predictions)
        conll_pred = "/tmp/predictions.conll"
        out_file = open(conll_pred, "w")
        for sample_id in predictions_processed:
            pred = predictions_processed[sample_id]
            out_file.write(f"#begin document id={sample_id}\n")
            for mention_id in sorted(pred):
                out_file.write(f"{mention_id}\t({pred[mention_id]})\n")
            out_file.write("#end document\n")
        out_file.close()
        conll_gold = "/tmp/gold.conll"
        target_file = open(conll_gold, "w")
        for i, sample in enumerate(predictions, start=1):
            target = sample["targets"]
            sample_id = sample[self.id_key]
            mention_to_cluster = {}
            for cluster_id, cluster in enumerate(target):
                for mention_id in cluster:
                    mention_to_cluster[mention_id] = cluster_id
            target_file.write(f"#begin document id={sample_id}\n")
            for mention_id in sorted(mention_to_cluster):
                target_file.write(f"{mention_id}\t({mention_to_cluster[mention_id]})\n")
            target_file.write("#end document\n")
        r = subprocess.run(["perl", "/Users/gililior/research/py_repos/reference-coreference-scorers/scorer.pl", "all", conll_gold, conll_pred], capture_output=True)
        processed = self.process_conll_out(r.stdout.decode())
        return processed


