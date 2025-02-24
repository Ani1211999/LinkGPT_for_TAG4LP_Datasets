#!/bin/bash

DATASET_NAME=amazon_clothing_20k
LINKGPT_DATA_PATH=$PWD/data # you can change this to any other path you like to store the data
PROJECT_PATH=$PWD
WANDB_KEY=None # you can set this to your own wandb key
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp

python ${PROJECT_PATH}/linkgpt/dataset/generate_eval_dataset.py \
	--model_name_or_path meta-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME} \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--device_setting cuda:0 \
	--max_hop 0 \
	
