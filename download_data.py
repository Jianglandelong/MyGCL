import warnings
import torch
import scipy.sparse as sp
import numpy as np
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Actor, WebKB, Amazon, Coauthor, WikiCS
from torch_geometric.utils import remove_self_loops

warnings.simplefilter("ignore")

def load_data(dataset_name):

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset_name)
    elif dataset_name in ['chameleon']:
        dataset = WikipediaNetwork(path, dataset_name)
    elif dataset_name in ['squirrel']:
        dataset = WikipediaNetwork(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['actor']:
        dataset = Actor(path)
    elif dataset_name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset_name)
    elif dataset_name in ['computers', 'photo']:
        dataset = Amazon(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['cs', 'physics']:
        dataset = Coauthor(path, dataset_name, transform=T.NormalizeFeatures())
    elif dataset_name in ['wikics']:
        dataset = WikiCS(path)

    data = dataset[0]

    edges = remove_self_loops(data.edge_index)[0]

    features = data.x
    [nnodes, nfeats] = features.shape
    nclasses = torch.max(data.y).item() + 1

    # if dataset_name in ['computers', 'photo', 'cs', 'physics', 'wikics']:
    #     train_mask, val_mask, test_mask = get_split(nnodes)
    # else:
    #     train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    # if len(train_mask.shape) < 2:
    #     train_mask = train_mask.unsqueeze(1)
    #     val_mask = val_mask.unsqueeze(1)
    #     test_mask = test_mask.unsqueeze(1)

    labels = data.y
    return data

if __name__ == '__main__':
    # choices=['cora', 'citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 'cornell',
    #                              'texas', 'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics']
    choices=['citeseer', 'pubmed', 'chameleon', 'squirrel', 'actor', 
                                  'wisconsin', 'computers', 'photo', 'cs', 'physics', 'wikics']
    for dataset in choices:
        load_data(dataset)