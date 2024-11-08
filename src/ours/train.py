from argparse import ArgumentParser
from trainer import train
from trie.build_trie import MarisaTrie
import pickle
import logging
import os

def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="None", type=str, help="model name for huggingface model hub")
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--id_type", default="title", type=str)
    parser.add_argument("--task", default="indexing", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--device", default= -1, type=str)
    parser.add_argument("--precision", default= "bf16-mixed") 
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--memo", type=str, default="None")
    parser.add_argument("--max_source_length", default= 64, type=int)
    parser.add_argument("--max_target_length", default= 128, type=int)
    parser.add_argument("--batch_size", default= 64, type=int)
    parser.add_argument("--valid_batch_size", default= 1, type=int)
    parser.add_argument("--max_steps", default=5000000, type=int)
    parser.add_argument("--val_check_interval", default=200000, type=int)
    parser.add_argument("--num_sanity_val_steps", default= -1, type=int)
    parser.add_argument("--num_warmup_steps_ratio", default= 0.1, type=float)
    parser.add_argument("--lr", default= 2e-5, type=float)
    parser.add_argument("--lr_scheduler", default= "constant", type=str)
    parser.add_argument("--early_stop", default=10, type=int)
    parser.add_argument("--checkpoint_path", default="./checkpoint", type=str)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--length_penalty", default=1.0, type=float)
    parser.add_argument("--trie_path", type=str)
    parser.add_argument("--num_beams", default=100, type=int)
    parser.add_argument("--seed", default=31933, type=int)
    parser.add_argument("--atomic_id_path", default=None, type=str)
    parser.add_argument("--ckpt_path", default=None)

    
    args = parser.parse_args()
    
    return args


def main(config, trie):
    train(config, trie)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)

    config = parse_argument()
    
    with open(config.trie_path, "rb") as f:
        trie = pickle.load(f)
    logging.info("-"*30 + "main function called!" + "-"*30)
    main(config, trie)