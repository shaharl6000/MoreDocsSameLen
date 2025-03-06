import json

import pandas as pd
from datasets_classes.base import Dataset


class MusiQue(Dataset):

    def get_max_num_docs(self):
        return 20

    def get_sample(self, instructions, documents, **kwargs):
        documents = self.get_shuffled_documents(documents)
        question = kwargs.get("question", "")
        doc_strings = "\n".join([f"Document {j}: ```{doc}```" for j, doc in enumerate(documents, start=1)])
        output_format = kwargs.get("output_format", "")
        target = kwargs.get("answer", "")
        sample = (f"*Instructions*:{instructions} {output_format}\n"
                  f"*Question*: {question}? "
                  f"*The documents*:\n{doc_strings}\n"
                  f"*Answer*: {target}")
        return sample

    def load(self):
        data_files_text, data_file_paths = self._read_files('jsonl')
        df_musique = {}
        for split_name, split_data in data_files_text.items():
            lines = split_data.strip().split('\n')
            df_musique[split_name] = pd.DataFrame([json.loads(line) for line in lines])
        return df_musique

    def pre_process(self):
        all_samples = {}
        for split_name, split_data in self.all_data.items():
            results_data = []
            for sample_id, row in split_data.iterrows():
                src_docs = [p['paragraph_text'] for p in row['paragraphs']]
                msgs = self.get_sample2msg(src_docs, question=row['question'])
                answer = row.get('answer', '')
                answerable = row.get('answerable', '')
                target = {"is_answerable": answerable, "answer_content": answer}
                results_data.append({
                    'id': row['id'],
                    'final_msgs': msgs,
                    'target': target
                })
            all_samples[split_name] = results_data
        return all_samples
