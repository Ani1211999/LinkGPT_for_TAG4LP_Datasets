import pandas as pd

import torch
import pandas as pd
import numpy as np
import torch
import random
import json
import pickle
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from typing import Tuple, List, Dict, Set, Any
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
from torch_geometric.utils import from_dgl

import os,sys

project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)  
from linkgpt.utils import basics
from linkgpt.dataset import tag_dataset_for_lm
FILE_PATH = "rag_data/"
DATASET_NAME = 'arxiv_2023'

#Load the data from CSV file for retrieving text attributes of graph
def load_text_data(text_data_file_path):
    # Add your implementation here
    df = pd.read_csv(FILE_PATH + text_data_file_path)
    return df

#Load the graph data in PyG format    
def load_graph_data(graph_data_file_path) -> Data:
    data = torch.load(FILE_PATH + graph_data_file_path)
    return data

def load_tag_data(text_data_file_path,graph_data_file_path) -> Tuple[Data, List[str]]:
    graph = load_graph_data(graph_data_file_path)
    text = load_text_data(text_data_file_path)
    return graph, text



#Convert CSV to dictionary format matching with LM Dataset format with node id as key and title
def create_node_with_data_list(df,text_field_attr,longer_text_field_attr):
    nid2data={}
    for idx, row in df.iterrows():
        nid2data[row["node_id"]] = {
            text_field_attr: row[text_field_attr],
            longer_text_field_attr: row[longer_text_field_attr] if pd.notna(row.get(longer_text_field_attr)) else "", # If present
        }
    
    return nid2data

# Convert edge_index (PyG) into a list of (source, target) pairs
def create_edge_pairs(graph_data):
    nid_pair_list = []
    for src, dst in graph_data.edge_index.t().tolist():
        nid_pair_list.append((src, dst))
    return nid_pair_list

def main():
    basics.set_seeds(42)
    text_data_file_path = 'arxiv_2023_orig/paper_info.csv'
    graph_data_file_path = 'arxiv_2023/graph.pt'
    graph_data, df_text = load_tag_data(text_data_file_path,graph_data_file_path)
    print(df_text.head(5))
    
    #Preprocessing the graph data as done in TAG4LP
    if graph_data.is_directed() is True:
        graph_data.edge_index = to_undirected(graph_data.edge_index)
        undirected = True
        
    graph_data.edge_index, _ = coalesce(graph_data.edge_index, None, num_nodes=graph_data.num_nodes)
    graph_data.edge_index, _ = remove_self_loops(graph_data.edge_index)
    print(f"original num of nodes: {graph_data.num_nodes}")
    print(f"Shape of data: {graph_data.x.shape}")
    print(f"num of edges: {graph_data.edge_index.shape[1]}")

    #For arxiv_2023_dataset, change according to the respective dataset
    text_field_attr ='title'
    longer_text_field_attr = 'abstract'
    
    nid2data = create_node_with_data_list(df_text,text_field_attr,longer_text_field_attr)
    nid_pair_list = create_edge_pairs(graph_data=graph_data)
    print(f"Total number of nodes: {len(nid2data)}")
    print(f"Total number of edges: {len(nid_pair_list)}") #Verifying with the original graph number of edges and number of nodes above

    #Create a LM dataset object suitable for LinkGPT
    dataset_for_lm = tag_dataset_for_lm.TAGDatasetForLM(nid2data, nid_pair_list,'title','abstract')
    
    #Split the edges into training, validation and test sets
    dataset_for_lm.generate_edge_split(num_neg_dst=150, train_ratio=0.8, valid_ratio=0.15) 
    
    #Dump the LM dataset into suitable location
    file_path = f'data/datasets/{DATASET_NAME}/dataset_for_lm.pkl'
    basics.save_pickle(data=dataset_for_lm,file_name=file_path)
    
if __name__ == '__main__':
    main()
