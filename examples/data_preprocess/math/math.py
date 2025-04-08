import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from typing import Dict, List, Optional, Any
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/cmu/deepscaler/data')
    parser.add_argument('--remote_dir', default='d1shs0ap/DeepScaleR-Preview-Dataset')
    

    args = parser.parse_args()

    train_dataset = load_dataset(args.remote_dir, split='train').filter(lambda x: 12 / 16 <= x['reward'])

    def make_map_fn(split: str):
        """Create a mapping function to process dataset examples.

        Args:
            split: Dataset split name ('train' or 'test')

        Returns:
            Function that processes individual dataset examples
        """
        def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            question = example.pop('problem')
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            question = f"{question} {instruction}"
            answer = example.pop('answer')

            data = {
                "data_source": "",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
