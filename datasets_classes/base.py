import os
from glob import glob
import pandas as pd
from datasets import load_dataset
import json
from transformers import AutoTokenizer
import numpy as np
import random
from together import Together
from tqdm import tqdm
import time
from datasets_classes.dataset_loader import MyDataset
from torch.utils.data import Dataset as TorchDataset, DataLoader


def collate_fn( batch):
    """
    Custom collate function to handle batches of dictionaries.

    Args:
        batch: List of dictionaries. Each element in the batch is a dict.

    Returns:
        A list of dictionaries, with no concatenation of the values.
    """
    return batch

class Dataset:
    common_data = {"num_demos": 3, "max_num_samples": -1, "random": random.Random(42)}

    @staticmethod
    def update_common_data(key, value):
        """ A static method to update the common data dictionary. """
        Dataset.common_data[key] = value

    def __init__(self, name, dir_path, split_name):
        self.dir_path = dir_path
        self.name = name
        self.all_prompts = self._get_prompts_from_json()
        self.cur_prompt = None
        self.all_data = self.load()
        self.max_num_docs = self.get_max_num_docs()
        self.shuffled_doc_ids = self.common_data["random"].sample(range(self.max_num_docs), k=self.max_num_docs)
        self.all_samples = self.pre_process()
        self._remove_long_samples(lim=-1)
        self.split_name = split_name
        if self.common_data["max_num_samples"] > -1:
            self._random_sampling()

    def get_max_num_docs(self):
        return max([self.all_data[k]["documents"].apply(len).max() for k in self.all_data])

    def _random_sampling(self):
        max_num_samples = self.common_data["max_num_samples"]
        if max_num_samples == -1:
            return
        for k in self.all_samples:
            if len(self.all_samples[k]) <= max_num_samples:
                continue
            random_generator = self.common_data["random"]
            self.all_samples[k] = random_generator.sample(self.all_samples[k], k=max_num_samples)

    def _remove_long_samples(self, lim):
        if lim == -1:
            return
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        for k in self.all_samples:
            encoded_batch = [tokenizer.encode(sample['final_msgs'][1]['content']) for sample in self.all_samples[k]]
            lengths = [len(encoded) for encoded in encoded_batch]
            print(f"Max length in {k}: {max(lengths)}")
            print(f"Min length in {k}: {min(lengths)}")
            print(f"Mean length in {k}: {np.mean(lengths)}")
            longer_than_limit = [i for i, l in enumerate(lengths) if l > lim]
            if len(longer_than_limit) > 0:
                print(f"Removing {len(longer_than_limit)} out of {len(self.all_samples[k])} samples that are longer than {lim} tokens")
                self.all_samples[k] = [sample for i, sample in enumerate(self.all_samples[k]) if i not in longer_than_limit]
            print(f"Number of samples in {k}: {len(self.all_samples[k])}")

    def load(self):
        """ read the dataset files and return the data in a dictionary """
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def pre_process(self):
        """ pre-process the data and return a dictionary of samples that expects 'final_msgs' and 'target' keys """
        raise NotImplementedError("This method needs to be implemented by subclasses")

    def _get_prompts_from_json(self):
        json_path = os.path.join("../files/prompts", f"{self.name}.json")
        with open(json_path, 'r', encoding='utf-8') as file:
            prompts = json.load(file)
        return prompts

    def get_prompt(self):
        random_generator = self.common_data['random']
        if self.cur_prompt is None:
            random_prompt = random_generator.choice(self.all_prompts['prompts'])
            selected_demonstrations = random_generator.sample(self.all_prompts['demonstrations'], k=self.common_data['num_demos']) # no replacements sampling
            processed_demos = []
            for i, sample in enumerate(selected_demonstrations, start=1):
                documents = sample.pop("documents")
                demonstration = self.get_sample(random_prompt["instructions"], documents, **sample)
                processed_demos.append(demonstration)
            random_prompt["few_shots"] = "\n\n".join(processed_demos)
            self.cur_prompt = random_prompt
        return self.cur_prompt

    def get_shuffled_documents(self, documents):
        tmp_documents = [None] * self.max_num_docs
        tmp_documents[:len(documents)] = documents
        shuffled = np.array(tmp_documents)[self.shuffled_doc_ids]
        new_documents = shuffled[~pd.isnull(shuffled)].tolist()
        return new_documents

    def get_sample(self, instructions, documents, **kwargs):
        documents = self.get_shuffled_documents(documents)
        doc_strings = "\n".join([f"Document {j}: ```{doc}```" for j, doc in enumerate(documents, start=1)])
        target = kwargs.get("target", "")
        sample = (f"*Instructions*: {instructions}\n"
                  f"*The documents*:\n{doc_strings}\n"
                  f"*Answer*: {target}")
        return sample


    def predict(self, model, out_path, num_truncation_tokens):
        split_name = self.split_name
        all_msgs = [sample['final_msgs'] for sample in self.all_samples[split_name]]
        all_responses = model.batch(all_msgs, num_truncation_tokens)

        # all_responses = []
        # print(len(all_msgs))
        # if len(all_msgs) == 200:
        #     jumps = 1
        # else:
        #     jumps = 1
        # for i in tqdm(range(0, len(all_msgs), jumps)):
        #     end = min((i + jumps), len(all_msgs))
        #     responses = model.batch(all_msgs[i:end], num_truncation_tokens)
        #     all_responses = all_responses + responses
    # def predict(self, model, out_path, num_truncation_tokens):
    #     split_name = self.split_name
    #     all_msgs = [sample['final_msgs'] for sample in self.all_samples[split_name]]
    #     print("len_msg", len(all_msgs))
    #     print("len_msg", type(all_msgs))
    #     print("len_msg", all_msgs[0][0].keys())
    #     dataset = MyDataset(all_msgs)
    #     if len(all_msgs) == 200:
    #         batch_size = 1
    #     else:
    #         batch_size = 2
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn = collate_fn)
    #
    #
    #     all_responses = []
    #     print(len(all_msgs))
    #
    #     for samples in tqdm(dataloader):
    #         # end = min((i + jumps), len(all_msgs))
    #         # print("len_sampels",len(samples))
    #         # print("len_sampels",type(samples))
    #         # print("len_sampels",samples[0][0].keys())
    #         # print(samples[0])
    #         responses = model.batch(samples, num_truncation_tokens)
    #         all_responses = all_responses + responses


        # responses = model.batch(all_msgs[2368:], num_truncation_tokens)
        # all_responses = all_responses + responses



        # all_responses = []
        # for i in tqdm(range(0, 5, 1)):
        #     responses = model.batch(all_msgs[i:i + 1], num_truncation_tokens)
        #     all_responses = all_responses + responses
        # responses = model.batch(all_msgs[5:10], num_truncation_tokens)
        # all_responses = all_responses + responses
        for i, response in enumerate(all_responses):
            self.all_samples[split_name][i]['prediction'] = response

        out_df = pd.DataFrame(self.all_samples[split_name])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_json(out_path, orient='records', indent=2)
        return out_df


    def predict_togehter(self, model_name, out_path, num_truncation_tokens):
        split_name = self.split_name
        all_msgs = [sample['final_msgs'] for sample in self.all_samples[split_name]]
        client = Together()

        # all_responses = model.batch(all_msgs, num_truncation_tokens)
        all_responses = []
        print(len(all_msgs))
        end = 0
        counter = 0
        flattened_all_msgs = [d for sublist in all_msgs for d in sublist]
        if len(all_msgs) == 200:
            jumps = 1
        else:
            jumps = 1
        start_time = time.time()
        counter = 0
        for i in tqdm(range(0, len(all_msgs), jumps)):
            # if counter > 5:
            #     break
            # else:
            #     counter += 1
            end = min((i + jumps), len(all_msgs))
            stream = client.chat.completions.create(
                model=model_name,
                temperature=0.8,
                max_tokens=512,
                # model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=flattened_all_msgs[i:end],
                stream=True,
            )
            answer = ''
            for chunk in stream:
                answer = answer + chunk.choices[0].delta.content

            # print(answer)
            all_responses = all_responses + [answer]

        for i, response in enumerate(all_responses):
            self.all_samples[split_name][i]['prediction'] = response

        out_df = pd.DataFrame(self.all_samples[split_name])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_json(out_path, orient='records', indent=2)
        return out_df


    def get_sample2msg(self, src_docs, **kwargs):
        prompt = self.get_prompt()
        sample_content = self.get_sample(prompt["instructions"], src_docs, **kwargs)
        demonstrations = prompt["few_shots"]
        user_message = f"{demonstrations}\n\n{sample_content}"
        messages = [{"role": "user", "content": user_message}]
        return messages

    def _read_files(self, suffix):
        f_names = os.path.join(self.dir_path, f'*.{suffix}')
        data_file_paths = {
            os.path.basename(fname).split('.')[0]: fname
            for fname in glob(f_names)
        }
        data_files_text = {}
        for file_name in data_file_paths:
            with open(data_file_paths[file_name], 'r') as f:
                data_files_text[file_name] = f.read()
        return data_files_text, data_file_paths

    @staticmethod
    def _load_from_hf(dataset_name, **kwargs):
        data_from_hf = load_dataset(dataset_name, **kwargs)
        data_dfs = {}
        for category, data in data_from_hf.items():
            data_dfs[category] = pd.DataFrame(data)
        return data_dfs
