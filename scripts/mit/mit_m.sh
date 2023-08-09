#!/bin/bash -x

python train.py --do_test --log_output_path ./pred/mrc_10_shot/log.txt --test_path ./prompt_data/MIT_M/ --test_file mrc_10_shot --gen_batch_size=50 --num_finetune_epochs 2 --dataset mit --pred_path ./pred/ --model_selection_path ./model_selection/
