import json
import os
import re
from glob import glob

import numpy as np

MODELS_NAME_MAPPING = {
    "Meta-Llama-3-8B-Instruct": "Llama3-8B",
    "Meta-Llama-3-70B-Instruct": "Llama3-70B",
    "gemma-1.1-7b-it": "Gemma1.1-7B",
    "gemma-1.1-2b-it": "Gemma1.1-2B",
    "Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "Mixtral-8x7B-Instruct-v0.1": "Mixtral-8x7B",
    "Mixtral-8x22B-Instruct-v0.1": "Mixtral-8x22B",
}

class Eval:
    def __init__(self, id_key, predictions_dir, out_path):
        self.id_key = id_key
        self.predictions_dir = predictions_dir
        self.out_path = out_path

    def _evaluate(self, data, model_name, sample_index):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _evaluate_new(self, data, model_name, sample_index):
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def eval_all_in_dir(self):
        print("Results will be saved in:", self.out_path)
        all_results = {}
        sample_lengths = {}
        if os.path.exists(self.out_path):
            with open(self.out_path, 'rt') as f:
                existing = json.load(f)
            all_results = existing["models"]
            sample_lengths = existing["sample_lengths"]
        for f_name in glob(f'{self.predictions_dir}/*/*.json'):
            sample_name = f_name.replace(self.predictions_dir, "").replace(".json", "")
            model = sample_name[:-2].split(os.sep)[-1]
            model_name = MODELS_NAME_MAPPING[model]
            sample_index = int(sample_name[-1])
            print("Evaluating model:", model_name, "sample:", sample_index)
            if model_name in all_results and sample_index in all_results[model_name]["run_index"]:
                print("Skipping", model_name, sample_index)
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
                    length = len(pred["final_msgs"][0]['content'])
                    sample_lengths[id_sample] = length
            all_results[model_name]["ids"].append(current_ids)
            results = self._evaluate_new(predictions, model_name, sample_index)
            all_results[model_name]["scores"].append(results)
            out_dict = {"sample_lengths": sample_lengths, "models": all_results}
            with open(self.out_path, 'wt') as f:
                json.dump(out_dict, f)

    @staticmethod
    def get_only_response(prediction):
        pred = prediction["prediction"]
        if '[/INST]' in pred:
            pred = pred[pred.find("[/INST]")+len("[/INST]"):].strip()
        elif "So, the answer is:" in pred:
            pred = pred[pred.rfind("So, the answer is:") + len("So, the answer is:"):].strip()
            if pred == "":
                # print(prediction["prediction"])
                pred = "No answer"
        elif "Aspect-based summary:" in pred:
            pred = pred[pred.rfind("Aspect-based summary:") + len("Aspect-based summary:"):].strip()
        else:
            pred = pred[pred.rfind("*Answer*:") + len("*Answer*:"):].strip()
        return pred

class QA(Eval):

    def __init__(self, id_key, predictions_dir, out_path):
        self.correlations = []
        super().__init__(id_key, predictions_dir, out_path)

    def _evaluate(self, predictions, model_name, sample_index):
        follow_format = []
        for sample in predictions:
            gt = sample["target"]
            pred = self.postprocess(sample)
            follow_format.append(type(pred) is dict)
        self.correlations.append([np.mean(follow_format), np.mean([])])
        metrics = {"all_f1": []}
        return metrics
    
    def _evaluate_new(self, predictions, model_name, sample_index):
        id_to_answers = {}
        for sample in predictions:
            gt = sample["target"]
            id_sample = sample["id"]
            pred = self.postprocess(sample)
            # gt, pred = self.parse(gt, pred)
            pred = extract_answer_content(pred)
            
            if id_sample not in id_to_answers:
                id_to_answers[id_sample] = {}
            id_to_answers[id_sample] = pred
        return id_to_answers

    def parse(self, ground_truth_answer, predicted_answer):
        gt = self._extract_answer_from_dict(ground_truth_answer)
        if type(predicted_answer) is dict:
            pred = self._extract_answer_from_dict(predicted_answer)
        else:
            pred = predicted_answer
        return gt, pred
    
 
    @staticmethod
    def _extract_answer_from_dict(answer_dict):
        if answer_dict != "**":
            if "is_answerable" in answer_dict:
                if answer_dict["is_answerable"]:
                    answer = answer_dict["answer_content"]
                else:
                    answer = answer_dict["is_answerable"]
                return answer
        else:
            return "Not answerable"

    def postprocess(self, pred):
        only_response = self.get_only_response(pred)
        answer_dict = re.search(r'\{.*\}', only_response)
        if answer_dict is None:
            return only_response

        str_dict = answer_dict.group(0)
        str_dict = str_dict.replace("'s", "\\'s").replace("'t", "\\'t").replace("s' ", "s\\' ")
        str_dict = str_dict.replace("\\\\_", "_").replace("\\_", "_")
        try:
            answer = eval(str_dict)
        except Exception as e:
            try:
                str_dict = str_dict.replace("}", "'}")
                answer = eval(str_dict)
            except Exception as e:
                answer = only_response
        return answer

def find_json_files(directory):
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "results" not in file and file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def extract_answer_content(result):
    pattern = r"['\"]?answer_content['\"]?\s*:\s*['\"]?([^'\"]*)['\"]?"
    if type(result) is not str:
        result = str(result)

    match = re.search(pattern, result)
    if match:
        answer_content = match.group(1)
        return answer_content
    else:
        return result 
    

if __name__ == "__main__":
    input_directory = r"C:\Users\LIHI\Downloads\base-20240721T055544Z-001\base\MusiQue"
    id_to_answers = {}
    qa_evaluator = QA("id", input_directory, "results_k.json")
    
    json_files = find_json_files(input_directory)
    print("Found JSON files:")
    # print(json_files)
    
    for file_path in json_files:
        model_name = os.path.splitext(os.path.basename(file_path))[0]
        print(file_path)
        with open(file_path, 'rt') as f:
            predictions = json.load(f)
            model_answers = qa_evaluator._evaluate_new(predictions, model_name, "sample_idx")
            for key,value in model_answers.items():
                if key not in id_to_answers:
                    id_to_answers[key] = {}
                id_to_answers[key][model_name] = value
            # id_to_answers.update(model_answers)
    
    # Print or save the results as needed
    print("\nFinal id_to_answers:")
    print(id_to_answers)
    output_file = os.path.join(input_directory, "results_k.json")
    with open(output_file, 'wt') as f:
        json.dump(id_to_answers, f)
