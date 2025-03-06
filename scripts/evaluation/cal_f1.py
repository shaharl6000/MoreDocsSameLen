
import json
import numpy as np







if __name__ == '__main__':
    name_model = "meta-llama"
    if name_model == "google":
        name_model_eval = "Gemma1.1-2B"
        name_model_second_name = "gemma-1.1-2b-it"
    elif name_model == "meta-llama":
        name_model_eval = "Llama3-8B"
        name_model_second_name = "Meta-Llama-3-8B-Instruct"
    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/nodoc/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    indexs = np.where(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]) < 0.1)[0]
    print("nodoc r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/baseline/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("baseline r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]) )
    print("baseline r without internal knowledge", np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]) )

###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/extended/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("extended r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("extended r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/full/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("full r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("full r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/raplaced/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("raplaced r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("raplaced r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    ###############################################################################################

    path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/pad/results.json"
    with open(path_baseline, 'rt') as f:
        results = json.load(f)

    print("[pad] r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    print("[pad] r without internal knowledge",
          np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    ###############################################################################################


    # path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/rephrasedQuestionsNoDocs/results_{name_model_second_name}.json"
    # with open(path_baseline, 'rt') as f:
    #     results = json.load(f)
    #
    # print("rephrasedNodoc r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    # print("rephrasedNodoc r without internal knowledge",
    #       np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    # ###############################################################################################
    #
    # path_baseline = f"/cs/labs/tomhope/nirm/MusiQue/eval/{name_model}/MusiQue/rephrasedQuestions/results_{name_model_second_name}.json"
    # with open(path_baseline, 'rt') as f:
    #     results = json.load(f)
    #
    # print("rephrasedfull r", np.mean(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"]))
    # print("rephrasedfull r without internal knowledge",
    #       np.mean(np.array(results["models"][f"{name_model_eval}"]["scores"][0]["all_f1"])[indexs]))
    # ###############################################################################################