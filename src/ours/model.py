from typing import List, Tuple, Dict
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from argparse import ArgumentParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StoppingCriteria, LogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule, get_cosine_with_hard_restarts_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalRecall, RetrievalMRR, RetrievalHitRate
import pandas as pd
import numpy as np
import gc

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [32100]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

class StopAfterConditionMet(LogitsProcessor):
    def __init__(self, eos_token_id, stop_condition):
        self.eos_token_id = eos_token_id
        self.stop_condition = stop_condition
        
    def __call__(self, input_ids, scores):
        for i, seq in enumerate(input_ids):
            if self.stop_condition(seq):
                forced_scores = torch.full_like(scores[i], -float('inf'))
                forced_scores[self.eos_token_id] = 0
                scores[i] = forced_scores
        return scores
    

logits_processor = LogitsProcessorList([
    StopAfterConditionMet(
        eos_token_id=1,
        stop_condition=lambda seq: 32100 in seq
    )
])

# generative Retriever Model
class GenerativeRetriever(pl.LightningModule):
    def __init__(self, trie, config):
        """_summary_
            initialize generative retriever
        Args:
            tokenizer (PreTrainedTokenizer): tokenizer for pre-trained model
            config (ArgumentParser): training configuration
        """
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        special_tokens_dict = {'additional_special_tokens': ['[DOC]', '[PATH]']}

        tokenizer.add_special_tokens(special_tokens_dict)
        self.doc_token_id = tokenizer.convert_tokens_to_ids("[DOC]")
        
        if config.id_type == "atomic" or config.id_type == "path+atomic":
            if config.atomic_id_path is not None:
                with open(config.atomic_id_path, "rb") as f:
                    atomic_id_list = pickle.load(f)

                
                tokenizer.add_tokens(atomic_id_list)
                
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.stopping_criteria = EosListStoppingCriteria()

        logits_processor = LogitsProcessorList([
            StopAfterConditionMet(
                eos_token_id=1, 
                stop_condition=lambda seq: 32100 in seq)
                ])
        self.logit_processor = logits_processor
        
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        metrics = MetricCollection(
            {
                "epoch_R1": RetrievalHitRate(top_k=1, aggregation="mean"),
                "epoch_R10": RetrievalHitRate(top_k=10, aggregation="mean"),
                "epoch_R100": RetrievalHitRate(top_k=100, aggregation="mean"),
                "epoch_MRR10": RetrievalMRR(top_k=10, aggregation="mean"),
                "epoch_MRR100": RetrievalMRR(top_k=100, aggregation="mean")
            }
        )
        
        self.valid_metrics = metrics.clone(prefix='valid_')
        self.test_metrics = metrics.clone(prefix='test_')
        
        self.r1_list  = []
        self.r10_list = []
        self.r100_list = []
        self.mrr10_list = []
        self.mrr100_list = []

        self.trie = trie
        self.model_type = config.model_name.split("/")[-1].split("-")[0]
        self.save_hyperparameters()

    def forward(self, inputs):
        """_summary_
            define forward propagation
        Args:
            inputs (Dict[str, torch.Tensor]): batch data(input_ids, decoder_input_ids, labels)

        Returns:
            Seq2SeqLMOutput: result of forward propagation
        """
        
        result = self.model(input_ids=inputs['input_ids'],
                          attention_mask=inputs['attention_mask'],
                          labels = inputs['labels'],
                          return_dict=True
                          )

        return result

    def training_step(self, batch, batch_idx):
        """define training step

        Args:
            batch (Dict[str, torch.Tensor]): batch data(input_ids, decoder_input_ids, labels)
            batch_idx (int): index of batch

        Returns:
            torch.Tensor: Cross-Entropy Loss
        """
        # forward propagation of this class
        self.model.train()
        outs = self(batch)
        loss = outs["loss"]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return loss
        

    def _multi_path_beam_search(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor,
                      max_length: int,
                      num_return_sequences: int,
                      length_penalty: float,
                      ):
        """_summary_
            generate path with beam search
        Args:
            input_ids (torch.Tensor): input_ids
            attention_mask (torch.Tensor): attention_mask
            max_length (int): maximum length of output
            num_beams (int): number of beams
            
        Returns:
            Dict[str, torch.Tensor]: output of model
        """

        path_result = self.model.generate(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            max_length=max_length//2,
                                            num_beams=num_return_sequences,
                                            num_return_sequences=num_return_sequences,
                                            return_dict_in_generate=True,
                                            logits_processor = self.logit_processor,
                                            length_penalty=length_penalty
                                            )
        

        # path_result: (num_beams, max_length)
        # path_score: (num_beams)
        path_result = path_result["sequences"]
        
        # remove pad token in path_result
        path_result = self.tokenizer.batch_decode(path_result)
        path_result =  [x.split("</s>")[0] for x in path_result]
        path_result = [ sequence + "[DOC]" if not sequence.endswith("[DOC]") else sequence for sequence in path_result]
        
        path_result = [self.tokenizer(path, return_tensors="pt")["input_ids"][:, :-1] for path in path_result]

        # print(self.tokenizer.decode(path_result[0]))
        
        return path_result
        
    
    def _docid_generate(self,
                        encoder_input_ids: torch.Tensor,
                        encoder_attention_mask: torch.Tensor,
                        decoder_input_ids: torch.Tensor,
                        max_length: int,
                        num_beams: int,
                        length_penalty: int,
                        num_return_sequences: int
    ):
        """_summary_
            generate docid with beam search
        Args:
            encoder_input_ids (torch.Tensor): input_ids
            encoder_attention_mask (torch.Tensor): attention_mask
            decoder_input_ids (torch.Tensor): decoder_input_ids
            max_length (int): maximum length of output
            num_beams (int): number of beams
            num_return_sequences (int): number of return sequences
            prefix_allowed_tokens_fn (callable): function for prefix allowed tokens
            
        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        stop_token_position = decoder_input_ids.shape[-1] -1
        docid_result = self.model.generate(input_ids=encoder_input_ids,
                                            attention_mask=encoder_attention_mask,
                                            decoder_input_ids=decoder_input_ids,
                                            max_length=max_length,
                                            num_beams=num_beams,
                                            num_return_sequences=num_return_sequences,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            early_stopping=False,
                                            do_sample=False,
                                            length_penalty = length_penalty,
                                            # constrained decoding
                                            prefix_allowed_tokens_fn=lambda batch_id, sent: self.trie.get(sent[stop_token_position:].tolist())
                                            )
        return docid_result
                        
        
    def validation_step(self, batch, batch_idx):
        """define validation step

        Args:
            batch (Dict[str, torch.Tensor]): batch data(input_ids, decoder_input_ids, labels)
            batch_idx (int): index of batch
            
            validate with R@1, R@10, R@100, MRR@100

        Returns:
            _type_: _description_
        """
        if self.trainer.sanity_checking:
            return None

        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # generate path with greedy decoding without constrained decoding
            path_result = self.model._multi_path_beam_search(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=self.model.config.max_target_length,
                                num_return_sequences=self.model.config.num_path,
                                length_penalty=self.model.config.length_penalty,
                                )
            
            raw_pred_docid_list = []
            pred_docid_list = []
            sequence_score_list = []

            for i in range(len(path_result)):
                path = path_result[i].to(self.device)
                output = self._docid_generate(encoder_input_ids=input_ids,
                                            encoder_attention_mask=attention_mask,
                                            decoder_input_ids=path,
                                            max_length=self.model.config.max_target_length,
                                            length_penalty=self.model.config.length_penalty,
                                            num_beams=100,
                                            num_return_sequences=100)
                # sequence_score: (100)
                sequence_score = output["sequences_scores"]
                sequence_score_list.extend(sequence_score)

                raw_pred_docid = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=False)
                pred_docid = [docid.split("[DOC]")[-1].split("</s>")[0].strip() for docid in raw_pred_docid]

                raw_pred_docid_list.extend(raw_pred_docid)
                pred_docid_list.extend(pred_docid)


            # sort by sequence score
            sequence_score_list = torch.stack(sequence_score_list, dim=0)
            _, sorted_idx = torch.sort(sequence_score_list, descending=True)
            raw_pred_docid_list = [raw_pred_docid_list[i] for i in sorted_idx]
            pred_docid_list = [pred_docid_list[i] for i in sorted_idx]

            # get sorted sequence score
            sequence_score_list = [sequence_score_list[i] for i in sorted_idx]

            # drop duplicates pred_docid while maintaining order
            pred_docid_series = pd.Series(pred_docid_list)
            drop_duplicated_index = list(pred_docid_series.drop_duplicates(keep='first')[:100].index)

            raw_pred_docid = [raw_pred_docid_list[i] for i in drop_duplicated_index]
            pred_docid = [pred_docid_list[i] for i in drop_duplicated_index]
            sequence_score_list = [sequence_score_list[i] for i in drop_duplicated_index]

            # convert to (batch_size, 100)
            raw_pred_docid = np.array(raw_pred_docid).reshape(-1, 100)
            # convert to list in list
            raw_pred_docid = [raw_pred_docid[i].tolist() for i in range(raw_pred_docid.shape[0])]
            
            
            pred_docid = np.array(pred_docid).reshape(-1, 100)
            # convert to list in list
            pred_docid = [pred_docid[i].tolist() for i in range(pred_docid.shape[0])]


            # get sorted sequence score
            sequence_score_list = np.array(sequence_score_list).reshape(-1, 100)
            sequence_score_list = [sequence_score_list[i].tolist() for i in range(sequence_score_list.shape[0])]

            # evaluate with retrieval metrics
            # evaluate with R@1, R@10, R@100, MRR@100

            # indexes : which query a pred refers to (according to num_return_sequences)
            # ex : when num_return_sequence is 2, then [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] -> 0, 1, 2, 3, 4 are query indexes in batch
            # indexes.shape : (batch_size * num_beams)
            indexes = torch.tensor([i//100 for i in range(len(pred_docid))]).to(self.device)

            # preds : estimated probabilities of each document to be relevant
            # preds.shape : (batch_size * num_beams)
            # = output["sequences_scores"]
            preds = torch.tensor(sequence_score_list).to(self.device)

            preds = preds.view(-1, 100)
            preds = F.softmax(preds, dim=-1)
            preds = preds.view(-1)

            # gt_docid : ground truth docid
            # gt_docid.shape : (batch_size, max_length)
            gt_docid = batch["labels"]

            # convert docid to string for evaluation
            # gt_docid.shape = (batch_size)
            gt_docid = self.tokenizer.batch_decode(gt_docid, skip_special_tokens=True)

            # target : boolean list about each document id is ground truth or not
            # target.shape : (batch_size, num_beams)
            target = torch.tensor([pred == gt_docid[i//100] for i, pred in enumerate(pred_docid)]).to(self.device)

            step_metric = self.valid_metrics(preds=preds, target=target, indexes=indexes)

            self.r1_list.append(step_metric['valid_epoch_R1'])
            self.r10_list.append(step_metric['valid_epoch_R10'])
            self.r100_list.append(step_metric['valid_epoch_R100'])
            self.mrr10_list.append(step_metric['valid_epoch_MRR10'])
            self.mrr100_list.append(step_metric['valid_epoch_MRR100'])

        
    def on_validation_epoch_end(self) -> None:
        """_summary_
            logging retrieval metrics
        """
        # logging retrieval metrics


        r1 = torch.tensor(self.r1_list).to(self.device).mean()
        r10 = torch.tensor(self.r10_list).to(self.device).mean()
        r100 = torch.tensor(self.r100_list).to(self.device).mean()
        mrr10 = torch.tensor(self.mrr10_list).to(self.device).mean()
        mrr100 = torch.tensor(self.mrr100_list).to(self.device).mean()

        if not self.trainer.sanity_checking:
            self.log('valid_epoch_R1', r1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('valid_epoch_R10', r10, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('valid_epoch_R100', r100, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('valid_epoch_MRR10', mrr10, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
            self.log('valid_epoch_MRR100', mrr100, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        self.r1_list = []
        self.r10_list = []
        self.r100_list = []
        self.mrr10_list = []
        self.mrr100_list = []
        
        self.valid_metrics.reset()
        
    
    def test_step(self, batch, batch_idx):
        """define test step

        Args:
            batch (Dict[str, torch.Tensor]): batch data(input_ids, decoder_input_ids, labels)
            batch_idx (int): index of batch
            
            validate with R@1, R@10, R@100, MRR@100

        Returns:
            _type_: _description_
        """
        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # generate path with greedy decoding without constrained decoding
            path_result = self.model._multi_path_beam_search(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=self.model.config.max_target_length,
                                num_return_sequences=self.model.config.num_path,
                                length_penalty=self.model.config.length_penalty,
                                )
            
            raw_pred_docid_list = []
            pred_docid_list = []
            sequence_score_list = []

            for i in range(len(path_result)):
                path = path_result[i].to(self.device)
                output = self._docid_generate(encoder_input_ids=input_ids,
                                            encoder_attention_mask=attention_mask,
                                            decoder_input_ids=path,
                                            max_length=self.model.config.max_target_length,
                                            length_penalty=self.model.config.length_penalty,
                                            num_beams=100,
                                            num_return_sequences=100)
                # sequence_score: (100)
                sequence_score = output["sequences_scores"]
                sequence_score_list.extend(sequence_score)

                raw_pred_docid = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=False)
                pred_docid = [docid.split("[DOC]")[-1].split("</s>")[0].strip() for docid in raw_pred_docid]

                raw_pred_docid_list.extend(raw_pred_docid)
                pred_docid_list.extend(pred_docid)


            # sort by sequence score
            sequence_score_list = torch.stack(sequence_score_list, dim=0)
            _, sorted_idx = torch.sort(sequence_score_list, descending=True)
            raw_pred_docid_list = [raw_pred_docid_list[i] for i in sorted_idx]
            pred_docid_list = [pred_docid_list[i] for i in sorted_idx]

            # get sorted sequence score
            sequence_score_list = [sequence_score_list[i] for i in sorted_idx]

            # drop duplicates pred_docid while maintaining order
            pred_docid_series = pd.Series(pred_docid_list)
            drop_duplicated_index = list(pred_docid_series.drop_duplicates(keep='first')[:100].index)

            raw_pred_docid = [raw_pred_docid_list[i] for i in drop_duplicated_index]
            pred_docid = [pred_docid_list[i] for i in drop_duplicated_index]
            sequence_score_list = [sequence_score_list[i] for i in drop_duplicated_index]

            # convert to (batch_size, 100)
            raw_pred_docid = np.array(raw_pred_docid).reshape(-1, 100)
            # convert to list in list
            raw_pred_docid = [raw_pred_docid[i].tolist() for i in range(raw_pred_docid.shape[0])]
            
            
            pred_docid = np.array(pred_docid).reshape(-1, 100)
            # convert to list in list
            pred_docid = [pred_docid[i].tolist() for i in range(pred_docid.shape[0])]


            # get sorted sequence score
            sequence_score_list = np.array(sequence_score_list).reshape(-1, 100)
            sequence_score_list = [sequence_score_list[i].tolist() for i in range(sequence_score_list.shape[0])]

            # evaluate with retrieval metrics
            # evaluate with R@1, R@10, R@100, MRR@100

            # indexes : which query a pred refers to (according to num_return_sequences)
            # ex : when num_return_sequence is 2, then [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] -> 0, 1, 2, 3, 4 are query indexes in batch
            # indexes.shape : (batch_size * num_beams)
            indexes = torch.tensor([i//100 for i in range(len(pred_docid))]).to(self.device)

            # preds : estimated probabilities of each document to be relevant
            # preds.shape : (batch_size * num_beams)
            # = output["sequences_scores"]
            preds = torch.tensor(sequence_score_list).to(self.device)

            preds = preds.view(-1, 100)
            preds = F.softmax(preds, dim=-1)
            preds = preds.view(-1)

            # gt_docid : ground truth docid
            # gt_docid.shape : (batch_size, max_length)
            gt_docid = batch["labels"]

            # convert docid to string for evaluation
            # gt_docid.shape = (batch_size)
            gt_docid = self.tokenizer.batch_decode(gt_docid, skip_special_tokens=True)

            # target : boolean list about each document id is ground truth or not
            # target.shape : (batch_size, num_beams)
            target = torch.tensor([pred == gt_docid[i//100] for i, pred in enumerate(pred_docid)]).to(self.device)

            step_metric = self.test_metrics(preds=preds, target=target, indexes=indexes)

            self.r1_list.append(step_metric['test_epoch_R1'])
            self.r10_list.append(step_metric['test_epoch_R10'])
            self.r100_list.append(step_metric['test_epoch_R100'])
            self.mrr10_list.append(step_metric['test_epoch_MRR10'])
            self.mrr100_list.append(step_metric['test_epoch_MRR100'])
    
    def on_test_epoch_end(self) -> None:
        """_summary_
            logging retrieval metrics
        """

        r1 = torch.tensor(self.r1_list).to(self.device).mean()
        r10 = torch.tensor(self.r10_list).to(self.device).mean()
        r100 = torch.tensor(self.r100_list).to(self.device).mean()
        mrr10 = torch.tensor(self.mrr10_list).to(self.device).mean()
        mrr100 = torch.tensor(self.mrr100_list).to(self.device).mean()
        
        self.log('test_epoch_R1', r1, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_epoch_R10', r10, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_epoch_R100', r100, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_epoch_MRR10', mrr10, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_epoch_MRR100', mrr100, prog_bar=True, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        self.r1_list = []
        self.r10_list = []
        self.r100_list = []
        self.mrr10_list = []
        self.mrr100_list = []
        
        self.valid_metrics.reset()
        
    

    def configure_optimizers(self):
        """define optimizer and scheduler (AdamW, linear warmup)

        Returns:
            Tuple(List[torch.Optimizer], List[transformers.optimization]): optimizer and scheduler
        """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(optimizer_grouped_parameters, lr=self.config.lr, eps=1e-8)
        # optim = FusedAdam(self.parameters(), lr=2e-5, eps=1e-8)
        if self.config.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.config.num_warmup_steps_ratio,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.config.lr_scheduler == "constant":
            scheduler = get_constant_schedule(optim)

        elif self.config.lr_scheduler == "cosine":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optim,
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.config.num_warmup_steps_ratio,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        
        return [optim], [scheduler]
    
    