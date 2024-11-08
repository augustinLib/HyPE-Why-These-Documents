import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
import logging

def data_pipeline(train_data_path,
                  valid_data_path,
                  tokenizer,
                  config
                  ):
    if config.task == "indexing":
        input_col_name = "query"
        output_col_name = "docid"

    elif config.task == "retrieval":
        input_col_name = "query"
        output_col_name = "docid"
        
    elif config.task == "multi-task":
        input_col_name = "query"
        output_col_name = "docid"
    
    else:
        raise ValueError("task must be either 'indexing', 'retrieval' or 'multi-task'")
    

    data_module = GenerativeRetrievalDataModule(config=config,
                                                train_data_path=train_data_path,
                                                valid_data_path=valid_data_path,
                                                tokenizer=tokenizer,
                                                input_col_name=input_col_name,
                                                output_col_name=output_col_name,
                                                input_max_length=config.max_source_length,
                                                output_max_length=config.max_target_length,
                                                batch_size=config.batch_size)

    return data_module

class GenerativeRetrievalDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 input_col_name,
                 input_max_length,
                 output_col_name,
                 output_max_length,
                 ):
        self.input_data = data[input_col_name].tolist()
        self.output_data = data[output_col_name].tolist()
        self.tokenizer = tokenizer
        
        self.input_col_name = input_col_name
        self.input_max_length = input_max_length
        self.output_col_name = output_col_name
        self.output_max_length = output_max_length
        
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        input_text = self.input_data[idx]
        output_text = self.output_data[idx]
        
        input_encodings = self.tokenizer(input_text,
                                            truncation=True,
                                            max_length=self.input_max_length,
                                            padding='max_length',
                                            return_tensors='pt'
                                            )
        
        output_encodings = self.tokenizer(output_text,
                                            truncation=True,
                                            max_length=self.output_max_length,
                                            padding='max_length',
                                            return_tensors='pt'
                                            )
        
        result = {
            'input_ids': input_encodings['input_ids'].squeeze(),
            'attention_mask': input_encodings['attention_mask'].squeeze(),
            'labels': output_encodings['input_ids'].squeeze(),
        }
        
        
        return result

class GenerativeRetrievalDataModule(LightningDataModule):
    def __init__(self,
                 config,
                 train_data_path: str,
                 valid_data_path: str,
                 tokenizer,
                 input_col_name: str,
                 output_col_name:str = None,
                 test_data_path = None,
                 input_max_length=64,
                 output_max_length=128,
                 batch_size=64):
          
        super().__init__()
        self.config = config
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.test_data_path = test_data_path
        self.input_col_name = input_col_name
        self.output_col_name = output_col_name            
        
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.batch_size = batch_size
        self.valid_batch_size = config.valid_batch_size
        self.train_sampler = None
        self.valid_sampler = None
        
    def setup(self, stage: str) -> None:
        train_data = pd.read_csv(self.train_data_path, sep="\t")
        valid_data = pd.read_csv(self.valid_data_path, sep="\t")
        test_data = None
        
        if self.test_data_path is not None:
            test_data = pd.read_csv(self.test_data_path, sep="\t")
            
        if stage == 'fit':
            self.train_dataset = GenerativeRetrievalDataset(train_data,
                                                            self.tokenizer,
                                                            self.input_col_name,
                                                            self.input_max_length,
                                                            self.output_col_name,
                                                            self.output_max_length)
            
            self.valid_dataset = GenerativeRetrievalDataset(valid_data,
                                                            self.tokenizer,
                                                            self.input_col_name,
                                                            self.input_max_length,
                                                            self.output_col_name,
                                                            self.output_max_length)
            
            if self.config.strategy == "ddp" or self.config.strategy == "deepspeed_stage_3" or self.config.strategy == "deepspeed_stage_2":
                logging.info("-"*30 + "Sampler Initialized!!" + "-"*30)
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.valid_sampler = DistributedSampler(self.valid_dataset)
            
        
        if stage == 'test':
            if self.test_data_path is not None:
                test_input_encodings = self.tokenizer(test_data[self.input_col_name].tolist(),
                                                      truncation=True,
                                                      max_length=self.input_max_length,
                                                      padding='max_length',
                                                      return_tensors='pt'
                                                      )
                
        
        if stage == 'predict':
            if self.test_data_path is not None:
                self.test_dataset = GenerativeRetrievalDataset(test_data,
                                                            self.tokenizer,
                                                            self.input_col_name,
                                                            self.input_max_length,
                                                            self.output_col_name,
                                                            self.output_max_length)

                
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=self.train_sampler,
                          num_workers=8)
        
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.valid_batch_size,
                          shuffle=False,
                          sampler=self.valid_sampler,
                          num_workers=8)
                          
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.valid_batch_size,
                          shuffle=False,
                          num_workers=8)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset,
                            batch_size=self.valid_batch_size,
                            shuffle=False,
                            num_workers=8)
