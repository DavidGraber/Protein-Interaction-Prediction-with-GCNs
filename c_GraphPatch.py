import numpy as np

class GraphPatch:

    '''Class storing the data of an extracted surface patch graph, including the '''

    def __init__(self, feature_matrix, A, edge_index, edge_weight, geo_dist_matrix, fitness_value, coords, mutant_name):
        self.coords = coords
        self.distance_matrix = geo_dist_matrix
        self.A = A
        self.edge_index = edge_index
        self.edge_weight = np.reshape(edge_weight, (edge_weight.shape[0], 1))
        self.features = feature_matrix
        self.fitness = np.asarray(fitness_value)
        self.mutant = mutant_name

    def num_nodes(self):
        return len(self.coords)

    def num_edges(self):
        return self.edge_index.shape[1]

    def num_features(self):
        return self.features.shape[1]

    def __str__(self):
        string = '\
            Number of Nodes: {n}\n\
            Features: {f}\n\
            Adjacency Matrix: {a}\n\
            Geodesic Distance Matrix: {d}\n\
            Edge Weights (Geodesic Distances): {w}\n\
            Edge Index: {i}\n\
            Fitness: {fit}\n\
            Coordinates of Points: {c}\n\
            Mutant Name: {name}'\
            .format(n = self.coords.shape[0], f = self.features.shape, a = self.A.shape, w = self.edge_weight.shape, \
            d = self.distance_matrix.shape, i = self.edge_index.shape, fit=self.fitness, c = self.coords.shape, name = self.mutant)
        return string