import os.path
from argparse import ArgumentParser
import json
from datasets_classes.base import Dataset
from datasets_classes.qa.MuSiQue import MusiQue
import random
import pickle as pkl
from datasets import load_dataset

def read_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    fill_default_arguments(meta)
    return meta


def fill_default_arguments(metadata):
    metadata['num_demonstrations'] = metadata.get('num_demonstrations', 3)
    metadata['max_num_samples'] = metadata.get('max_num_samples', -1)


def load_all_datasets(num_demos, l_datasets, num_runs, max_num_samples, random_seed):
    Dataset.update_common_data("num_demos", num_demos)
    Dataset.update_common_data("max_num_samples", max_num_samples)
    Dataset.update_common_data("random", random.Random(random_seed))
    all_names = set()
    for ds_instance in l_datasets:
        name = ds_instance['name']
        split = ds_instance['split_name']
        all_names.add(name)
        path = ds_instance.get('path')
        print(f"Loading dataset {name}")
        ds_instance['instances'] = []
        for i in range(num_runs):
            ds = MusiQue(name, path, split)
            ds_instance["instances"].append(ds)
    print(f"Loaded all datasets_classes:\n{all_names} * {num_runs} runs.")
    return l_datasets


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    metadata_dict = read_metadata(args.config)
    out_dir = metadata_dict['out_dir']
    pickle_out = os.path.join(out_dir, f"datasets_{metadata_dict['run_name']}.pkl")
    print(f"Saving datasets_classes to {pickle_out}")
    datasets_list = load_all_datasets(metadata_dict['num_demonstrations'],
                                      metadata_dict['datasets_classes'],
                                      num_runs=metadata_dict['num_different_runs'],
                                      max_num_samples=metadata_dict['max_num_samples'],
                                      random_seed=metadata_dict['random_seed'])
    pickle_out = os.path.join(out_dir, f"datasets_{metadata_dict['run_name']}.pkl")
    print(f"Saving datasets_classes to {pickle_out}")
    with open(pickle_out, 'wb') as f:
        pkl.dump(datasets_list, f)
