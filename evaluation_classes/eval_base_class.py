
import tiktoken
from glob import glob
import json
import os
MODELS_NAME_MAPPING = {
    "Meta-Llama-3-8B-Instruct": "Llama3-8B",
    "Meta-Llama-3-70B-Instruct": "Llama3-70B",
    "gemma-1.1-7b-it": "Gemma1.1-7B",
    "gemma-1.1-2b-it": "Gemma1.1-2B",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
    'Qwen2-72B-Instruct':"Qwen/Qwen2-72B-Instruct",
    "Qwen2.5-72B-Instruct-Turbo": "Qwen2.5-72B-Instruct-Turbo",
    "gemma-2-27b-it": "gemma-2-27b-it",
    "gemma-2b-it": "gemma-2b-it",
    "Qwen/Qwen2-7B": "Qwen/Qwen2-7B",
    "Qwen2-7B": "Qwen2-7B",
    "google/gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
    "Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

}

import numpy as np

class Eval:
    def __init__(self, id_key, predictions_dir, out_path):
        self.id_key = id_key
        self.predictions_dir = predictions_dir
        self.out_path = out_path

    def _evaluate(self, data, model_name, sample_index):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def eval_all_in_dir(self):
        print("results will be saved in:", self.out_path)
        all_results = {}
        sample_lengths = {}
        if os.path.exists(self.out_path):
            with open(self.out_path, 'rt') as f:
                existing = json.load(f)
            all_results = existing["models"]
            sample_lengths = existing["sample_lengths"]
        encoding = tiktoken.encoding_for_model("gpt-4")
        for f_name in glob(f'{self.predictions_dir}/*.json'):
            sample_name = f_name.replace(self.predictions_dir, "").replace(".json", "")
            model = sample_name[:-2].split(os.sep)[-1]
            model_name = MODELS_NAME_MAPPING[model]
            sample_index = int(sample_name[-1])
            print("Evaluating model:", model_name, "sample:", sample_index)
            if model_name in all_results and sample_index in all_results[model_name]["run_index"]:
                print("skipping", model_name, sample_index)
                continue
            if model_name not in all_results:
                all_results[model_name] = {"scores": [], "run_index": [], "ids": []}
            all_results[model_name]["run_index"].append(sample_index)

            with open(f_name, 'rt') as f:
                predictions = json.load(f)

            current_ids = []
            for pred in predictions:
                id_sample = str(pred[self.id_key])
                current_ids.append(id_sample)
                if id_sample not in sample_lengths:
                    length = len(encoding.encode(pred["final_msgs"][0]['content']))
                    sample_lengths[id_sample] = length
            all_results[model_name]["ids"].append(current_ids)
            results = self._evaluate(predictions, model_name, sample_index)
            all_results[model_name]["scores"].append(results)
            out_dict = {"sample_lengths": sample_lengths, "models": all_results}
            with open(self.out_path, 'wt') as f:
                json.dump(out_dict, f)
        # print(np.corrcoef(np.array(self.correlations).T)[0,1])

    @staticmethod
    def get_only_response(prediction):
        pred = prediction["prediction"]
        if isinstance(pred, list):
            pred = str(pred[1])

        if pred is None:
            return pred
        if '[/INST]' in pred:
            pred = pred[pred.find("[/INST]")+len("[/INST]"):].strip()
        elif "So, the answer is:" in pred:
            pred = pred[pred.rfind("So, the answer is:") + len("So, the answer is:"):].strip()
            if pred == "":
                # print(prediction["prediction"])
                pred = "No answer"
        elif "Aspect-based summary:" in pred:
            pred = pred[pred.rfind("Aspect-based summary:") + len("Aspect-based summary:"):].strip()
        elif pred.rfind("*Answer*:") != -1:
            pred = pred[pred.rfind("*Answer*:") + len("*Answer*:"):].strip()

        return pred
