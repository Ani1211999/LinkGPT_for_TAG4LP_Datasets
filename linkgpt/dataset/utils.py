from typing import List, Tuple, Dict, Callable, Set
from dataclasses import dataclass
import random
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import torch_sparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100

NODE_START_TOKEN = '<node_start>'
NODE_TOKEN = '<node>'
PAIRWISE_START_TOKEN = '<pairwise_start>'
PAIRWISE_TOKEN = '<pairwise>'

LINKGPT_SPECIAL_TOKENS = [
    NODE_START_TOKEN,
    NODE_TOKEN,
    PAIRWISE_START_TOKEN,
    PAIRWISE_TOKEN,
]

def sample_neg_tgt(num_neg_tgt: int, pos_tgt_set: Set[int], total_node_num: int):
    """
    Randomly sample `total_node_num` elements from `pos_tgt_set`, and ensure that no sampled elements are in `pos_tgt_set`
    """
    while True:
        neg_tgt_list = random.sample(range(total_node_num), num_neg_tgt)
        if not any(neg_tgt in pos_tgt_set for neg_tgt in neg_tgt_list):
            return neg_tgt_list

def sample_neg_tgt_new(num_neg_tgt: int, pos_tgt_set: Set[int], total_node_num: int):
    """
    Efficiently sample `num_neg_tgt` negative targets that are NOT in `pos_tgt_set`.
    """
    # Get all possible negative targets
    valid_negatives = list(set(range(total_node_num)) - pos_tgt_set)

    # Ensure we don’t sample more than available negatives
    num_neg_tgt = min(num_neg_tgt, len(valid_negatives))

    return random.sample(valid_negatives, num_neg_tgt) 
def get_text_with_encoding_token(center, config, gnid2text, do_pairwise=True):
    """
    Get the text with the encoding tokens for the prompt
    """
    
    ablate_node_encoding = getattr(config, 'ablate_node_encoding', True)
    ablate_pairwise_encoding = getattr(config, 'ablate_pairwise_encoding', True)
    
    node_prefix = "" if ablate_node_encoding else NODE_START_TOKEN + NODE_TOKEN * (config.node_encoding_max_hop + 1)
    pairwise_prefix = "" if (ablate_pairwise_encoding or not do_pairwise) else PAIRWISE_START_TOKEN + PAIRWISE_TOKEN
    if node_prefix + pairwise_prefix == "":
        return 'text: ' + gnid2text[center] + '\n'
    else:
        return node_prefix + pairwise_prefix + '\n' + 'text: ' + gnid2text[center] + '\n'