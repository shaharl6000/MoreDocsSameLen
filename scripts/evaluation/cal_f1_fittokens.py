
import json
import numpy as np







if __name__ == '__main__':
    name_model = "meta-llama"
    if name_model == "google":
        name_model_eval = "Gemma1.1-2B"
        name_model_second_name = "gemma-1.1-2b-it"
        name_pad_dataset = "rephrasedQuestionsPad_token_gemma"
    elif name_model == "meta-llama":
        name_model_eval = "Llama3-8B"
        name_model_second_name = "Meta-Llama-3-8B-Instruct"
        name_pad_dataset = "rephrasedQuestionsPad_token_llama"
    elif name_model == "mistralai":
        name_model_eval = "Mistral-7B"
        name_model_second_name = "Mistral-7B-Instruct-v0.2"
        name_pad_dataset = "rephrasedQuestionsPad_token_mistral"

    eval_name = "eval_wo_shuffle_fit_token"

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{eval_name}/{name_model}/MusiQue/datasets_extendedFitTokens/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    # indexs = np.where(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]) < 0.1)[0]
    print("datasets_extendedFitTokens mean", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("datasets_extendedFitTokens content", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    print("datasets_extendedFitTokens format", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_format"]))
    ###############################################################################################


    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{eval_name}/{name_model}/MusiQue/datasets_rephrasedQuestionsExtendedFitTokens/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("datasets_rephrasedQuestionsExtendedFitTokens mean", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("datasets_rephrasedQuestionsExtendedFitTokens content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    print("datasets_rephrasedQuestionsExtendedFitTokens format",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_format"]))

###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{eval_name}/{name_model}/MusiQue/fullSet/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("fullSet mean",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("fullSet content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    print("fullSet format",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_format"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{eval_name}/{name_model}/MusiQue/raplacedSet/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("raplacedSet mean",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("raplacedSet content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    print("raplacedSet format",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_format"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{eval_name}/{name_model}/MusiQue/rephrasedQuestionsReplaced/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedQuestionsReplaced mean",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedQuestionsReplaced content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    print("rephrasedQuestionsReplaced format",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_format"]))
