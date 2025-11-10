# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/aime25_with_llm_answers')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source = 'MathArena/aime_2025_outputs'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)

    test_dataset = dataset['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            problem_index = example.pop('problem_idx')
            answer_index = example.pop('idx_answer')
            question = example.pop('problem')
            answer = example.pop('answer')
            model_name = example.pop('model_name')
            instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
            is_correct = example.pop('correct')
            
            data = {
                "data_source": data_source,
                "question" : question + " " + instruction_following,
                "answer" : answer,
                'problem_index' : problem_index,
                "answer_index" : answer_index,
                "model_name" : model_name,
                "correct" : is_correct
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('train'),
                                    with_indices=True,
                                    remove_columns=test_dataset.column_names)
    
    # Step 1: Keep only rows with correct answers
    test_dataset = test_dataset.filter(lambda x: x["correct"])

    # Step 2: Remove duplicate questions (based on the user prompt content)
    # Each prompt has [{'role': 'user', 'content': question + instruction}], so we extract question content
    def get_question(example):
        return example["question"]

    # Keep only the first occurrence of each unique question
    unique_questions = {}
    unique_data = []
    for item in test_dataset:
        question_text = get_question(item)
        if question_text not in unique_questions:
            unique_questions[question_text] = True
            unique_data.append(item)

    from datasets import Dataset
    test_dataset = Dataset.from_list(unique_data)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
