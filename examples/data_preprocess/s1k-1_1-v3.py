"""
Preprocess the simplerl dataset to parquet format
"""

import os
import datasets
import argparse

data_source = 'skytliang/s1K-1.1_v3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/s1k-1_1-v3')
    args = parser.parse_args()

    train_dataset = datasets.load_dataset(data_source, split='train')

    def process_fn_train(example, idx):
        data = {
            "data_source": data_source,
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example['answer'],
                "question": example['question'],
            }
        }
        return data

    def process_fn_test(example, idx):
        data = {
            "data_source": data_source,
            "prompt": example['messages'],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer']
            },
            "extra_info": {
                'split': 'test',
                'index': idx,
                'answer': example['answer'],
                "question": example['question'],
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn_train, with_indices=True)
    test_dataset = train_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))