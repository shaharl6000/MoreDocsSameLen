
import json
import numpy as np







if __name__ == '__main__':
    name_model = "google"
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
    name_eval = "eval_bug"
    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsNoDocs/results_{name_model_second_name}.json"
    # with open(path_baseline, 'rt') as f:
    #     results = json.load(f)
    #
    # indexs = np.where(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]) < 0.1)[0]
    # print("nodoc r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))

    ###############################################################################################
    indexs = [1]

    # path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsNoDocs/results_{name_model_second_name}.json"
    # with open(path_baseline, 'rt') as f:
    #     results = json.load(f)
    #
    # print("rephrasedNodoc r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    # print("rephrasedNodoc r without internal knowledge",
    #       np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    # print("datasets_no doc content",
    #       np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsExtended/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedQuestionsExtended r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedQuestionsExtended r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    print("datasets_etended content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsBaseline/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedQuestionsBaseline r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedQuestionsBaseline r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    print("datasets_baseline content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsReplaced/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedQuestionsReplaced r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedQuestionsReplaced r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    print("datasets_raplaced content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/{name_pad_dataset}/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print(f"{name_pad_dataset} r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print(f"{name_pad_dataset} r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    # print("datasets_no doc content",
    #       np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))

    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestions/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedfull r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedfull r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    print("datasets_full content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    ###############################################################################################

    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/{name_eval}/{name_model}/MusiQue/rephrasedQuestionsGibberish/results_{name_model_second_name}.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("rephrasedQuestionsGibberish r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("rephrasedQuestionsGibberish r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    print("datasets_gibr content",
          np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1_content"]))
    ###############################################################################################