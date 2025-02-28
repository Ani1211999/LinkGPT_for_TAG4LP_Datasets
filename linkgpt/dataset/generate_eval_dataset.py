from typing import List, Optional, Tuple, Union
import pickle
import json
import os
import sys
import argparse
from unittest.mock import patch
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import dgl
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import DataCollatorForLanguageModeling, Trainer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
import wandb

import llmtuner
from llmtuner.model.patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from llmtuner.hparams.parser import get_train_args
import llmtuner.hparams.parser as llm_tuner_parser
from llmtuner.extras.misc import count_parameters
from llmtuner.model.loader import init_adapter

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)
from linkgpt.dataset.tag_dataset_for_lm import TAGDatasetForLM, tag_dataset_for_lm_to_dgl_graph
from linkgpt.pairwise_encoding.lpformer_dataset import get_lpformer_dataset
from linkgpt.pairwise_encoding.models.link_transformer import LinkTransformer
from linkgpt.pairwise_encoding.lpformer_model_api import get_lpformer_model
from linkgpt.model.linkgpt_model import LinkGPTForCausalLM, LinkGPTConfig, \
    unfreeze_graph_related_modules, unfreeze_lora_adapter, freeze_all_parameters, \
        save_lora_model, get_model_and_tokenizer, get_tokenizer, load_model_and_tokenizer
from linkgpt.dataset.linkgpt_dataset import LinkGPTDataset, LinkGPTDataCollator
from linkgpt.dataset.yn_dataset import YNTargetData, YNDataset, YNDatasetConfig, YNData, YNDatasetForEval, YNDatasetForEvalConfig
from linkgpt.dataset.np_dataset import NPData, NPDataset, NPDatasetConfig
from linkgpt.dataset.utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, \
    PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS, sample_neg_tgt_new
from linkgpt.utils import basics, prompts
import dgl
import torch
import random

def main():
    basics.set_seeds(42)

    parser = argparse.ArgumentParser()

    # data path and project description
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--text_embedding_method', required=True)
    parser.add_argument('--text_embedding_folder_path', required=True)
    parser.add_argument('--max_hop', default=0, type=int)
    parser.add_argument('--dataset_for_lm_path', required=True)
    parser.add_argument('--ppr_data_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--text_field_attr', required=True)
    parser.add_argument('--device_setting', default=None, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

   
    args = parser.parse_args()    
    if args.device_setting is None:
        device = basics.get_device()
        print(f"No device setting is provided. Using {device}", flush=True)
    else: 
        device = args.device_setting

    # Load dataset
    text_emb_list = []
    for i in range(args.max_hop + 1):
        if i == 0:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}.pt')
        else:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}_{i}hop.pt')
        text_emb = torch.load(text_emb_path, map_location=device)
        text_emb_list.append(text_emb)
    
    #Load the LM Dataset
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    
    ppr_data = torch.load(args.ppr_data_path).to(device)
    text_field_attr = args.text_field_attr
    #Create dictionary with the node indices and their respective text field values
    '''For example: gnid2text ={0:'Cloud Computing', 1:'AI',...}''' 
    gnid2text = {
        gnid: dataset_for_lm.data_list[nid].get(text_field_attr, "")  # Use "title" or another text field
        for nid, gnid in dataset_for_lm.nid2gnid.items()
    }

    #Create a dgl graph based on the LM Dataset
    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, include_valid=True).to(device)
    dgl_graph.ndata['feat'] = text_emb_list[0]
    print("Number of node in the graph:", dgl_graph.num_nodes())
    print("Number of edges in the graph:", dgl_graph.num_edges())
    test_src_nodes = dataset_for_lm.edge_split['test']['source_node']
    test_tgt_nodes = dataset_for_lm.edge_split['test']['target_node']
    test_neg_nodes = dataset_for_lm.edge_split['test'][ 'target_node_neg']
    print(f"Number of source nodes: {len(test_src_nodes)},len(test_tgt_nodes),len(test_neg_nodes)") 
    print(f"Number of target nodes: {len(test_tgt_nodes)}")
    print(f"Number of negative target nodes: {len(test_neg_nodes)}") #All three values must be the same.
    question_data = {
        "source_node": test_src_nodes,
        "target_node": test_tgt_nodes,
        "target_node_neg": test_neg_nodes
    }

    print("Sample question_data:", {k: v[:5] for k, v in question_data.items()})  
    
    #Load the LM tokenizer
    tokenizer = get_tokenizer()
    
    #Creating the test set for evaluation
    yn_eval_config =YNDatasetForEvalConfig()
    #Update the prompts for LLM as per the dataset
    yn_prompts = prompts.get_prompts(dataset_name=args.dataset_name,task_name='yn',allow_general_prompts=False)
    yn_eval_config.task_desc= yn_prompts['task_desc']
    yn_eval_config.source_node_intro= yn_prompts['source_node_intro']
    yn_eval_config.candidate_target_node_intro= yn_prompts['candidate_target_node_intro']
    yn_eval_config.connection_question= yn_prompts['connection_question']
    
    eval_dataset = YNDatasetForEval(dgl_graph=dgl_graph,question_data=question_data, gnid2text=gnid2text, config=yn_eval_config, tokenizer=tokenizer)
    file_name_with_path = f'data/datasets/{args.dataset_name}/eval_yn_dataset_4_examples.pkl'
    basics.save_pickle(eval_dataset,file_name_with_path) 
    
if __name__ == '__main__':
    main()