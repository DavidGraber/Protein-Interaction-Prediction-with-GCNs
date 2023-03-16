import numpy as np
import torch

class GraphPatch:

    '''Class storing the data of an extracted surface patch graph, including coordinates of the graph nodes (pos), 
    the features (x), the adjacency information (edge_index), the edge weights (edge_attr), the fitness (y) and the name of the mutant'''

    def __init__(self, x, edge_index, edge_attr, y, pos, mutant_name):
        self.pos        = torch.from_numpy(pos).float()
        self.edge_index = edge_index
        self.edge_attr  = edge_attr
        self.x          = torch.from_numpy(x).float()
        self.y          = torch.from_numpy(np.asarray(y)).float()  
        self.mutant     = mutant_name

    def num_nodes(self):
        return len(self.pos)

    def num_features(self):
        return self.x.shape[1]

    def __str__(self):
        string = '\
            Number of Nodes: {n}\n\
            Features: {f}\n\
            Edge_Index: {i}\n\
            Edge Weights (Softmin Geodesic Distances): {we}\n\
            Label: {fit}\n\
            Coordinates of Points: {c}\n\
            Mutant Name: {name}'\
            .format(n = self.pos.shape[0], f = self.x.shape, we = self.edge_attr.shape, \
            i = self.edge_index.shape, fit=self.y, c = self.pos.shape, name = self.mutant)

        return string
