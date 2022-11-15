import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d as o3d


def generate_simple_graph(indeces, coords_sel, normals):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    #from sklearn.neighbors import NearestNeighbors
    #import numpy as np
    
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



def generate_GNN_graph(patch_coords, patch_normals, patch_features):

    '''Function that takes a set of points, with their label, coordinates and surface normals. Calculates for each point the 
    geodesic distance to its n nearest neighbors and saves that information in a dictionary representing a graph. '''
    
    #Initialize geodesic distances matrix
    geo_dist_matrix = np.full((len(patch_coords), len(patch_coords)), np.inf)

    edges = []
    
    #Initialize Adjacency Matrix
    A = np.identity(n = len(patch_coords), dtype=np.float64)

    #Initialize K-Nearest-Neighbor Search
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(patch_coords)

    #loop through each point that is within the radius and find its nearest neighbors and their euclidean distance
    for idx, point in enumerate(patch_coords):
        dist, neighbors = knn.kneighbors([point], return_distance=True)
                
        # loop through the nearest neighbors, calculate their geodesic distance to the point chosen above
        # Add the geodesic distance to the distance matrix

        for index, neighbor in enumerate(neighbors[0]):
            
            geo_dist = dist[0][index]*(2-np.dot(patch_normals[idx], patch_normals[neighbor]))
            

            if geo_dist !=0:

                geo_dist_matrix[idx][neighbor]=geo_dist
                geo_dist_matrix[neighbor][idx]=geo_dist

                A[idx][neighbor] = 1
                A[neighbor][idx] = 1

                if (neighbor, idx) not in edges:
                    edges.append((idx, neighbor))


    #Initialize Edge Weights and Edge_Index
    edge_weight = []
    edge_index = [[],[]]
    
    for node1, node2 in edges: 
        edge_index[0].append(node1)
        edge_index[1].append(node2)
        edge_weight.append(geo_dist_matrix[node1][node2])

        
    edge_index = np.asarray(edge_index)
    edge_weight = np.asarray(edge_weight)
    feature_matrix = patch_features

    return patch_coords, geo_dist_matrix, A, edge_index, edge_weight, feature_matrix



def distances_from_center(graph, center):
    
    '''Function that takes a graph and the starting node and returns a list of distances 
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





def extract_surface_patch_GNN(coords, center_index, radius, features):
    
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
    #start = time.time()
    graph = generate_simple_graph(first_sel, coords_sel, normals)
    #end = time.time()
    #print("Graph Generation: "+ str(end - start) + 's')


    # check for each point the GEODESIC distance to the center with djikstra
    #start = time.time()
    dist_from_center = distances_from_center(graph, center_index)
    #end = time.time()
    #print("Distances from center: "+ str(end - start)+ 's')


    # Collect the indeces of the points that within the geodesic radius from the center point
    #start = time.time()

    patch_indeces = []
    for key in dist_from_center:
        if dist_from_center[key]<=radius:
            patch_indeces.append(key)

    #Collect Patch Data            
    patch_coords = coords[patch_indeces]
    patch_normals = normals[patch_indeces]
    patch_features = features[patch_indeces]
    
    #end = time.time()
    #print("Extraction of patch members: "+ str(end - start)+ 's')


    # Generate a new graph including only the patch members
    #start = time.time()
    patch_graph = generate_GNN_graph(patch_coords, patch_normals, patch_features)
    #end = time.time()
    #print("Generation of Patch Graph: " + str(end - start)+ 's')

   
    return patch_graph

