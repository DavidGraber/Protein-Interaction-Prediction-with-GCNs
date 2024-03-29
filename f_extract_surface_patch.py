import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
from torch.nn import Softmin
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse


def generate_simple_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its 10 nearest neighbors and saves that information in a double dictionary representing a graph.'''
        
    graph = {p:{} for p in indeces}

    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(coords_sel)

    #loop through each point that is within the radius and find its nearest neighbors and their euclidean distance
    for idx, point in enumerate(coords_sel):
        dist, neighbors = knn.kneighbors([point], return_distance=True)
              
        # loop through the nearest neighbors, calculate their geodesic distance to the point chosen above
        # Add the geodesic distance to a graph-dictionary
        
        for index, neighbor in enumerate(neighbors[0]):
            
            geo_dist = dist[0][index]*(2-np.dot(normals[indeces[idx]], normals[indeces[neighbor]]))        
            
            if geo_dist !=0:
                graph[indeces[idx]][indeces[neighbor]]=geo_dist
                graph[indeces[neighbor]][indeces[idx]]=geo_dist

    return graph



def distances_from_center(graph, center):
    
    '''Function that takes a graph and the starting node and returns a dictionary of distances 
    from the starting node to every other node'''
    
    dist_from_center = {key:100 for key in graph}
    dist_from_center[center] = 0
    unseen_nodes = list(dist_from_center.keys())
    
    for _ in graph:

        # IDENTIFICATION OF THE NEXT POINT TO LOOK AT (SHORTES DISTANCE FROM START)
        dist = 101
        for node in unseen_nodes:
            if dist_from_center[node]<dist:
                dist = dist_from_center[node]
                loc = node

        # LOOP THROUGH ALL THE NEIGHBORS OF THE NODE AND ADJUST THE VALUES OF THOSE, IF NEEDED
        for neighbor, weight in graph[loc].items():               
            if dist + weight < dist_from_center[neighbor]:
                dist_from_center[neighbor] = dist + weight 
        unseen_nodes.remove(loc)
        
    return dist_from_center



def generate_GNN_graph(patch_coords, patch_normals, patch_features):

    '''Function that takes a set of points, with their features, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its 10 nearest neighbors using an approximatin based on the surface normals. An adjacency matrix is 
    generated to save the edges connecting each point to its 10 nearest neighbors. The corresponding geodesic distances are 
    saved in an edge weight matrix. Both matrices are converted to sparse representations using dense_to_sparse()'''
    
    
    #Initialize Adjacency Matrix and Weight Matrix
    n = len(patch_coords)
    adj = np.zeros((n,n), dtype=np.float64)
    W = np.zeros((n, n))

    #Initialize K-Nearest-Neighbor Search
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(patch_coords)

    #Initialize softmin and the distance dictionary
    softmin = Softmin(dim = 0)
    geodesic_distances = {p:{} for p in range(n)}

    #loop through each point of the patch and find its nearest neighbors and their euclidean distance
    for idx, point in enumerate(patch_coords):
        dist, neighbors = knn.kneighbors([point], return_distance=True)
                
        # loop through the nearest neighbors and fill their neighborhood into the adjacency matrix
        # and the approximated geodesic distance into the lookup dictionary
        
        for index, neighbor in enumerate(neighbors[0]):
            
            # Fill the adjacency matrix
            adj[idx][neighbor] = 1
            adj[neighbor][idx] = 1

            #Approximation of the geodesic distance, taking the angle between the normals
            geo_dist = dist[0][index]*(2-np.dot(patch_normals[idx], patch_normals[neighbor]))
            geodesic_distances[idx][neighbor] = geo_dist
            geodesic_distances[neighbor][idx] = geo_dist

    
    # For each point, check in the adjacency matrix which neighbors it has and collect the geodesic
    # distances to those neighbors, perform a softmin over those distances and fill into the weight matrix
    for pt in range(n):
        neigh = [idx for idx, point in enumerate(adj[pt]) if point!=0]
        geo_dists_neighbors = [geodesic_distances[pt][i] for i in neigh]
        weights = softmin(torch.tensor(geo_dists_neighbors))

        for i, nb in enumerate(neigh): 
            W[pt][nb] = weights[i]


    # Generate edge_index and edge_attr from the weighted adjacency matrix
    edge_index, edge_attr = dense_to_sparse(torch.tensor(adj*W))

    return patch_coords, edge_index, edge_attr, patch_features





def extract_surface_patch_GCN(coords, center_index, radius, features):

    '''This function takes the surface point cloud with its features and coordinates, the center for 
    patch extraction and the desired geodesic radius and returns the a circular patch of the given 
    geodesic radius drawn around the given patch center and modelled as a graph with edge index, 
    adjacency matrix, edge_weights and features.'''

    
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(coords)
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn = 5))
    pointcloud.orient_normals_consistent_tangent_plane(k=5)
    normals = np.asarray(pointcloud.normals)

    first_sel = [] # to save all the points that are within the non-geodesic radius

    #loop through all the points and calculate their euclidean distance to the selected center
    for index, point in enumerate(coords):
        dist = np.linalg.norm(coords[center_index]-point)

        # first selection with only those points that are close to the center point
        if dist < radius:
            first_sel.append(index)
            
    coords_sel = coords[first_sel]


    # generate a graph with the selected points
    graph = generate_simple_graph(first_sel, coords_sel, normals)


    # check for each point the GEODESIC distance to the center with djikstra
    dist_from_center = distances_from_center(graph, center_index)


    # Collect the indeces of the points that within the geodesic radius from the center point
    patch_indeces = []
    for key in dist_from_center:
        if dist_from_center[key]<=radius:
            patch_indeces.append(key)

    #Collect Patch Data            
    patch_coords = coords[patch_indeces]
    patch_normals = normals[patch_indeces]
    patch_features = features[patch_indeces]
    

    # Generate a new graph including only the patch members
    patch_data = generate_GNN_graph(patch_coords, patch_normals, patch_features)
   
    return patch_data

