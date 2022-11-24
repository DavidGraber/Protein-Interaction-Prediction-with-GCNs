import numpy as np

class GraphPatch:

    '''Class storing the data of an extracted surface patch graph, including coordinates of the graph nodes (pos), 
    the features (x), the adjacency information (adj), the edge weights (w) and the fitness (y) and the name of the mutant'''

    def __init__(self, x, adj, w, y, pos, mutant_name):
        self.pos = pos
        self.adj = adj
        self.w = w
        self.x = x
        self.y = np.asarray(y, dtype=np.float64)
        self.mutant = mutant_name

    def num_nodes(self):
        return len(self.pos)

    def num_features(self):
        return self.x.shape[1]

    def __str__(self):
        string = '\
            Number of Nodes: {n}\n\
            Features: {f}\n\
            Adjacency Matrix: {a}\n\
            Edge Weights (Softmin Geodesic Distances): {we}\n\
            Label: {fit}\n\
            Coordinates of Points: {c}\n\
            Mutant Name: {name}'\
            .format(n = self.pos.shape[0], f = self.x.shape, a = self.adj.shape, we = self.w.shape, \
            i = self.edge_index.shape, fit=self.y, c = self.pos.shape, name = self.mutant)
        return string