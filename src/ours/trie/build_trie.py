from argparse import ArgumentParser
from transformers import AutoTokenizer
import pickle
import pandas as pd

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List
import marisa_trie

class MarisaTrie(object):
    def __init__(
        self,
        sequences: List[List[int]] = [],
        cache_fist_branch=True,
        max_token_id=32144,
    ):

        self.int2char = [chr(i) for i in range(min(max_token_id, 55000))] + (
            [chr(i) for i in range(65000, max_token_id + 10000)]
            if max_token_id >= 55000
            else []
        )
        self.char2int = {self.int2char[i]: i for i in range(max_token_id)}

        self.cache_fist_branch = cache_fist_branch
        if self.cache_fist_branch:
            self.zero_iter = list({sequence[0] for sequence in sequences})
            assert len(self.zero_iter) == 1
            self.first_iter = list({sequence[1] for sequence in sequences})

        self.trie = marisa_trie.Trie(
            "".join([self.int2char[i] for i in sequence]) for sequence in sequences
        )

    def get(self, prefix_sequence: List[int]):
        if self.cache_fist_branch and len(prefix_sequence) == 0:
            return self.zero_iter
        elif (
            self.cache_fist_branch
            and len(prefix_sequence) == 1
            and self.zero_iter == prefix_sequence
        ):
            return self.first_iter
        else:
            key = "".join([self.int2char[i] for i in prefix_sequence])
            return list(
                {
                    self.char2int[e[len(key)]]
                    for e in self.trie.keys(key)
                    if len(e) > len(key)
                }
            )

    def __iter__(self):
        for sequence in self.trie.iterkeys():
            yield [self.char2int[e] for e in sequence]

    def __len__(self):
        return len(self.trie)

    def __getitem__(self, value):
        return self.get(value)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default="t5-base", required=True)
    config = parser.parse_args()
    
    return config

def main(config):
    df = pd.read_csv(config.input, sep="\t")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    # with open("/work/Ex-GR-CoT/src/pilot_study/trie/kmeans_token.pkl", "rb") as f:
    #     new_token_list = pickle.load(f)
    
    with open("/work/Ex-GR-CoT/src/pilot_study/new_token_list.pkl", "rb") as f:
        new_token_list = pickle.load(f)
    
    tokenizer.add_tokens(new_token_list)
    # tokenizer.add_special_tokens({"additional_special_tokens": new_token_list})
    encoded_docid_list = tokenizer(df["docid"].values.tolist(), padding=False)
    trie = MarisaTrie([[32100] + docid for docid in encoded_docid_list["input_ids"]])
    
    return trie


if __name__ == "__main__":
    config = parse_args()
    trie = main(config)
    
    with open(config.output, "wb") as f:
        pickle.dump(trie, f)