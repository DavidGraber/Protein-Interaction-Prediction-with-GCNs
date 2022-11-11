import numpy as np
import os
import torch
from torch.utils.data import Dataset
from f_helper_functions import load_object
from torch_geometric.data import Data

class PatchDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.mutants = [mutant[0:4] for mutant in os.listdir(data_dir)]

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.mutants[idx]+'_GraphPatch.pkl')
        patch = load_object(path)
        
        x = torch.from_numpy(patch.features)
        edge_index=torch.from_numpy(patch.edge_index)
        edge_weight=torch.from_numpy(patch.edge_weight)
        y=torch.from_numpy(patch.fitness.astype(np.int64))
        pos=torch.from_numpy(patch.coords)

        return  Data(x, edge_index, edge_weight, y, pos)