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
        max_token_id=256001,
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
    

def recall(pred_docid, gt_docid):
    recall = []
    recall_num = [1, 10, 100]
    
    for i in recall_num:
        result = []
        for pred, gt in zip(pred_docid, gt_docid):
            is_hit = 0
            if gt in pred[:i]:
                is_hit = 1
            result.append(is_hit)
            
        recall.append(result)
        
    return recall[0], recall[1], recall[2]

def MRR100(pred_docid, gt_docid):
    mrr = []
    for pred, gt in zip(pred_docid, gt_docid):
        if gt in pred:
            rank = pred.index(gt) + 1
            mrr.append(1/rank)
        else:
            mrr.append(0)
    return mrr
        
        
def MRR10(pred_docid, gt_docid):
    mrr = []
    for pred, gt in zip(pred_docid, gt_docid):
        if gt in pred[:10]:
            rank = pred.index(gt) + 1
            mrr.append(1/rank)
        else:
            mrr.append(0)
    return mrr


def NDCG10(pred_docid, gt_docid, gt_score):
    mrr = []
    for pred, gt in zip(pred_docid, gt_docid):
        if gt in pred:
            rank = pred.index(gt) + 1
            mrr.append(1/rank)
        else:
            mrr.append(0)
    return mrr


