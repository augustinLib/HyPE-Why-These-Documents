# !/bin/bash

python ../train.py --model_name "t5-base" \
                    --train_data_path "/path/of/training/dataset" \
                    --valid_data_path "/path/of/valid/dataset" \
                    --dataset_name "nq or msmarco" \
                    --id_type "" \
                    --memo "" \
                    --device "0,1,2,3" \
                    --strategy "ddp" \
                    --max_source_length 512 \
                    --max_target_length 128 \
                    --batch_size 32 \
                    --val_check_interval 50000 \
                    --num_beams 100 \
                    --max_steps 2000000 \
                    --lr 5e-4 \
                    --lr_scheduler "linear" \
                    --num_warmup_steps_ratio 0.1 \
                    --trie_path "trie_path" \
                    --early_stop 10 \
                    --checkpoint_path "../checkpoint/dataset_name" \
                    --accumulate_grad_batches 1


