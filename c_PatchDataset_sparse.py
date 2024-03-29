import numpy as np
import os
import torch
from torch_geometric.data import Dataset
from f_helper_functions import load_object
from torch_geometric.data import Data



class PatchDataset(Dataset):

    '''Custom dataset for generation of datasets of graphs extracted from protein surfaces.
    The input data directory should contain instances of the GraphPatch class stored as pkl files with the characters
    0-4 of the filename indicating the complex name. The function get_item returns an instance of the
    torch_geometric.Data class which contains a graph with the following information:  
    - node features x
    - edge index (sparse)
    - edge weights (sparse)
    - label of the graph
    - fitness value of the mutant the patch has been extracted from
    '''

    def __init__(self, data_dir): 
        self.data_dir = data_dir
        self.mutants = [mutant[0:4] for mutant in os.listdir(data_dir)]

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.mutants[idx]+'_GraphPatch.pkl')
        patch = load_object(path)
        
        x_pos = torch.cat((patch.pos, patch.x), dim = -1) # added the 3D coords of the nodes to the features

        if patch.y >= 0.6:
            y = torch.tensor(1)
        else:
            y = torch.tensor(0)

        return  Data(x=x_pos.float(),
                     edge_index=patch.edge_index.long(), 
                     edge_attr = patch.edge_attr.float(), 
                     y=y.long(),
                     fitness = patch.y.float())