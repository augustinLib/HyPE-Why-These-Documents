import logging
import pickle
import wandb
import numpy as np
import random
import torch
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from model import GenerativeRetriever
from data import data_pipeline


def train(config, trie):
    devices = None
    accelerator = None
    if config.device == -1:
        accelerator = "cpu"
    else:
        accelerator = "gpu"
        
        temp = config.device.split(",")
        devices = [int(x) for x in temp]

    model_name = config.model_name.split("/")[-1]
    gpu_count = len(devices) if devices is not None else 1

    wandb_logger = WandbLogger(project=config.wandb_project, name=f"{config.dataset_name}-{config.id_type}-batch_size {config.batch_size* gpu_count * config.accumulate_grad_batches}-{config.lr}-{config.lr_scheduler}-{config.memo}")
    logging.info("-"*30 + "Wandb Setting Complete!" + "-"*30)
    
    seed = config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info("-"*30 + f"Seed {seed} Setting Complete!" + "-"*30)
    
    model = GenerativeRetriever(config=config, trie = trie)
    logging.info("-"*30 + "Model initialized!" + "-"*30)
    
    tokenizer = model.tokenizer

    data_module = data_pipeline(config.train_data_path, config.valid_data_path, tokenizer, config)

    # test_dataloader = data_pipeline(f"{data_path}/test.csv", tokenizer, config)

    logging.info("-"*30 + "Data Loaded!" + "-"*30)


    checkpoint_callback = ModelCheckpoint(monitor='valid_epoch_R1',
                                          dirpath=f'{config.checkpoint_path}',
                                          filename= f"{config.dataset_name}-{config.id_type}-batch_size_{config.batch_size* gpu_count * config.accumulate_grad_batches}-seed_{config.seed}-{config.lr}-{config.lr_scheduler}-{config.memo}"+"-{valid_epoch_R1:.2f}-{epoch}epoch",
                                          save_top_k=3,
                                          save_last=False,
                                          verbose=True,
                                          mode="max")
        
    
    early_stopping = EarlyStopping(
        monitor='valid_epoch_R1', 
        mode='max',
        patience=config.early_stop,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if config.early_stop == 0:
        callback_list = [lr_monitor, checkpoint_callback]
    else:
        callback_list = [checkpoint_callback, early_stopping, lr_monitor]
        

    val_check_interval = config.val_check_interval * config.accumulate_grad_batches
    max_steps = config.max_steps * config.accumulate_grad_batches

    trainer = pl.Trainer(
                         accelerator=accelerator,
                         devices=devices,
                         precision=config.precision,
                         strategy=config.strategy,
                         enable_progress_bar=True,
                         callbacks=callback_list,
                        max_steps=max_steps,
                        #  max_epochs=config.max_epochs,
                         val_check_interval=val_check_interval,
                         check_val_every_n_epoch=None,
                        #  check_val_every_n_epoch=config.check_val_every_n_epoch,
                         num_sanity_val_steps=config.num_sanity_val_steps,
                         logger=wandb_logger,
                         accumulate_grad_batches=config.accumulate_grad_batches,
                         )
    
    
    
    logging.info("-"*30 + "Train Start!" + "-"*30)
    trainer.fit(model=model,
                datamodule=data_module,
                ckpt_path=config.ckpt_path)

    logging.info("-"*30 + "Train Finished!" + "-"*30)

    # logging.info("-"*30 + "Test Start!" + "-"*30)
    # trainer.test(model, test_dataloader, ckpt_path="best")
    # logging.info("-"*30 + "Test Finished!" + "-"*30)

    wandb.finish()