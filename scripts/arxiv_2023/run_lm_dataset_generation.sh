#!/bin/bash

DATASET_NAME=arxiv_2023
LINKGPT_DATA_PATH=$PWD/data # you can change this to any other path you like to store the data
PROJECT_PATH=$PWD
WANDB_KEY=None # you can set this to your own wandb key
TEXT_FIELD_ATTRIBUTE=title
LONGER_TEXT_FIELD_ATTRIBUTE=abstract
NODE_ID=node_id

python ${PROJECT_PATH}/linkgpt/dataset/generate_lm_dataset.py \
	--output_dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --text_field_dataset ${TEXT_FIELD_ATTRIBUTE} \
	--node_id_field ${NODE_ID} \
    --longer_text_field_dataset ${LONGER_TEXT_FIELD_ATTRIBUTE} \
	--dataset_name ${DATASET_NAME} \
	--text_data_file_path rag_data/${DATASET_NAME}_orig/paper_info.csv \
	--graph_data_file_path rag_data/${DATASET_NAME}/graph.pt \
