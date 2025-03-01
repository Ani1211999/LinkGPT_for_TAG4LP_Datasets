import random
import sys
from typing import Dict, List, Tuple

import tqdm
import torch
import dgl
import datasets

class TAGDatasetForLM():
    '''
    1. Initiate a dataset by `dataset_for_lm = TAGDatasetForLM(nid2data, nid_pair_list)`
    2. Generate edge split by `dataset_for_lm.generate_edge_split()`
    '''
    def __init__(self, nid2data: Dict[str, Dict], nid_pair_list: List[Tuple[str]], text_field: str=None, longer_text_field: str=None):
        """
        Here "nid" means "node ID", and its type is str.
        "nid" is the the ID used on the original dataset, like ASIN (e.g., "B00AYP7KPO") for Amazon Co-purchasing dataset.
        """
        self.text_field = text_field
        self.longer_text_field = longer_text_field
        self.nid_list = list(nid2data.keys())
        self.nid2gnid = {nid: i for i, nid in enumerate(self.nid_list)}
        
        self.data_list = [nid2data[nid] for nid in self.nid_list]
        self.pair_list = [(self.nid2gnid[src], self.nid2gnid[tgt]) for src, tgt in nid_pair_list]
        
        gnid2neighbors = {gnid: [] for gnid, _ in enumerate(self.data_list)}
        for src, dst in self.pair_list:
            gnid2neighbors[src].append(dst)
        self.gnid2neighbors = gnid2neighbors
        
        # to be generated by manually calling generate_edge_split
        self.edge_split = None
        self.gnid2neighbors_train = None
        
    def get_neighbors(self, gnid: int):
        return self.gnid2neighbors[gnid]
    
    def __getitem__(self, gnid: int):
        return self.data_list[gnid]
    
    def get_pairitem(self, index: int):
        return self.pair_list[index]
    
    def len_pairitems(self):
        return len(self.pair_list)
    
    def get_textitem(self):
        return self.longer_text_field
    def __len__(self):
        return len(self.data_list)
    
    def generate_edge_split_with_splitted_pair_list(
        self,
        num_neg_dst: int,
        train_pair_list: List[Tuple[int]],
        valid_pair_list: List[Tuple[int]],
        test_pair_list: List[Tuple[int]],
        verbose: bool=True,
    ):
        pair_list = train_pair_list + valid_pair_list + test_pair_list
        dst_set = set([p[1] for p in pair_list])
        
        def sample_neg_dst(src: int):
            pos_dst_set = set(self.get_neighbors(src))
            while True:
                result = random.sample(range(len(self.data_list)), num_neg_dst)
                if all([i not in pos_dst_set for i in result]):
                    return result

        edge_split = {}
        edge_split['train'] = {
            'source_node': [pair[0] for pair in train_pair_list],
            'target_node': [pair[1] for pair in train_pair_list]
        }
        edge_split['valid'] = {
            'source_node': [pair[0] for pair in valid_pair_list],
            'target_node': [pair[1] for pair in valid_pair_list],
            'target_node_neg': [sample_neg_dst(pair[0]) for pair in tqdm.tqdm(valid_pair_list, disable=not verbose)]
        }
        edge_split['test'] = {
            'source_node': [pair[0] for pair in test_pair_list],
            'target_node': [pair[1] for pair in test_pair_list],
            'target_node_neg': [sample_neg_dst(pair[0]) for pair in tqdm.tqdm(test_pair_list, disable=not verbose)]
        }
        
        self.edge_split = edge_split
    
    def generate_edge_split(self, num_neg_dst: int, verbose: bool=True, train_ratio: float=0.9, valid_ratio: bool=0.05):
        """
        Generate edge split for training, validation, and testing.
        num_neg_dst: the number of negative target nodes to sample for each src node. 
        In this paper, we set it to 150 for LinkGPT w/o retrieval and 1800 for LinkGPT w/ retrieval.
        """
        if verbose:
            print('Generating edge split...')
            print(f'Num of negative dst per src: {num_neg_dst}')
        pair_list = self.pair_list
        random.shuffle(pair_list)
        num_train = int(len(pair_list) * train_ratio)
        num_valid = int(len(pair_list) * valid_ratio)
        num_test = len(pair_list) - num_valid - num_train
        
        train_pair_list = pair_list[:num_train]
        valid_pair_list = pair_list[num_train:num_train + num_valid]
        test_pair_list = pair_list[num_train + num_valid:]
        
        self.generate_edge_split_with_splitted_pair_list(num_neg_dst, train_pair_list, valid_pair_list, test_pair_list, verbose)
        self.generate_gnid2neighbors_train()
        
    def generate_gnid2neighbors_train(self):
        """
        Generate a dictionary that maps a gnid to its neighbors in the training set.
        """
        gnid2neighbors_train = {gnid: [] for gnid, _ in enumerate(self.data_list)}
        # since LLM does not need validation, we combine the valid and train set
        train_src_list = self.edge_split['train']['source_node'] + self.edge_split['valid']['source_node']
        train_dst_list = self.edge_split['train']['target_node'] + self.edge_split['valid']['target_node']
        for src, dst in zip(train_src_list, train_dst_list):
            gnid2neighbors_train[src].append(dst)
        self.gnid2neighbors_train = gnid2neighbors_train
        
    def get_neighbors_in_training_set(self, gnid: int):
        return self.gnid2neighbors_train[gnid]
    

def tag_dataset_for_lm_to_dgl_graph(dataset_for_lm: TAGDatasetForLM, device: str='cpu', include_valid: bool=True):
    """
    Convert a TAGDatasetForLM object to a DGL graph object.
    """
    print("Inside LM to Graph object conversion!!")
    num_nodes = len(dataset_for_lm)
    edge_split = dataset_for_lm.edge_split
    
    if include_valid:
        src_ls = torch.tensor(edge_split['train']['source_node'] + edge_split['valid']['source_node'])
        dst_ls = torch.tensor(edge_split['train']['target_node'] + edge_split['valid']['target_node'])
    else:
        src_ls = torch.tensor(edge_split['train']['source_node'])
        dst_ls = torch.tensor(edge_split['train']['target_node'])
    g = dgl.graph((src_ls, dst_ls), num_nodes=num_nodes).to(device)
    return g