<div align="center">
  <h1>More Documents, Same Length:<br>Isolating the Challenge of Multiple Documents in RAG</h1>
  <h3>Link to arxiv</h3>
</div>

MoreDocs is a toolset for studying how the number of documents affects Retrieval-Augmented Generation (RAG) performance when the total amount of tokens stays the same.

This repository contains code and datasets for our paper on the effects of document multiplicity while the context size is fixed in Retrieval-Augmented Generation (RAG) systems.
For detailed methodology, experiments, and analysis, please refer to the full paper 📰

## :bulb: High-Level Conclusions
Our results show that adding more retrieved documents can hurt performance—up to a 10% drop in fixed-context setups—making document rich retrieval tasks harder. 
Llama-3.1 and Gemma-2 declined, Qwen-2 stayed steady, and smaller LLMs (7–9B) followed the trend less strongly. This suggests systems need to balance relevance and variety to cut conflicts, and future models might improve by filtering out contradictory details while using the range of documents.

## 🔬 Our Methodology:
<div style="max-width: 400px; margin: 0 auto;">
Starting with a Wikipedia-derived dataset, we created different sets with the same amount of tokens but fewer documents by adjusting the length of the key documents for each question.
Our sets use the same multi-hop questions and supporting documents with key info <b>(pink)</b> , while varying distractor documents <b>(blue)</b>.
We began with 20 documents, then omitted redundant ones while lengthening the remaining ones to match the original size.
</div>
 <br>
<br>






<div align="center">
  <img src="/Main_Fig_Horizontal.png" alt="Alt text" width="800">
</div>


## :desktop_computer:  Reproduction Instructions:

### Download the different benchmark datasets
Our custom benchmark datasets include a control set, the original dataset, and variants with replaced distractors for varying document multiplicity. 
You can Download them from  [here](https://drive.google.com/file/d/1z6L0Xl0zhRoOOpwD5WuQI9ukSaEgCraM/view?usp=drive_link)
Alternatively, regenerate them using  [`scripts/create_various_sets.py`](scripts/create_various_sets.py).

### Prepare the environment

To set up the running environment, run the following command:
```bash
git clone git@github.cs.huji.ac.il:lihish400-1/CATBM.git
cd CATBM
export PYTHONPATH=./
python3.11 -m venv <PATH_TO_VENV>
source <PATH_TO_VENV>/bin/activate
pip install -r requirements.txt
```

### Run predictions
For running in inference on the chosen benchmark dataset you need to define for each benchmark data set a config file under configuration folder [`files/configuration/predict.json`](files/configuration/predict.json).

The `predict.json` file contains the path to the generated benchmark from previous step, the batch size, and the decoding temperature for the LLMs.

We supply two option for running the code with small models (the code run locally), with large model (the code run with Together platform)

To run prediction with the small models, run the following command:
```bash
python scripts/run_model_predictions.py --config <PATH_TO_CONFIG> --model_name <MODEL_NAME>
```

For the large model add ['together_api_key.py'](together_api_key.py) under the root path and define: API_KEY = XXXXX

then run the following command:

```bash
python scripts/run_model_predictions.py --config <PATH_TO_CONFIG> --model_name <MODEL_NAME> --run_together
```

### Evaluate the predictions

To evaluate the predictions, you can use [`scripts/evaluate_dataset.py`](scripts/evaluate_dataset.py) by providing 
the path to the predictions from previous step, and output path where all results will be saved.

```bash
python scripts/evaluate_dataset.py --predictions_dir <OUTPUT_PATH_FROM_PREV_STEP> --output_path <RESULT_OUTPUT> --ds_name MusiQue
```

## :newspaper: Citation

If you use this code or the datasets in your research, please cite:

```
@inproceedings{author2024more,
  title={More Documents, Same Length: Isolating the Challenge of Multiple Documents in RAG},
  author={Author, A. and Author, B.},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```


