from Bio.PDB.PDBParser import PDBParser
import numpy as np
from f_parse_pdb import parse_pdb
import os
from f_extract_surface_patch_GNN import *
from f_helper_functions import *
from c_GraphPatch import GraphPatch

#Import the fitness scores of all 23241 mutants
os.chdir('c:\\Users\\david\\MT_data')
fitness = np.load('fitness_dict_short.npy', allow_pickle="TRUE").item()

predfeatures_dir = [file for file in os.listdir('c:\\Users\\david\\MT_data\\masif_site_outputs\\predfeatures')]
predcoords_dir = [file for file in os.listdir('c:\\Users\\david\\MT_data\\masif_site_outputs\\predcoords')]
pdbs_dir = [file for file in os.listdir('c:\\Users\\david\\MT_data\\alphafold_outputs\\pdbs')]

source_dir_feat = 'C:/Users/david/MT_data/masif_site_outputs/predfeatures'
source_dir_cord = 'C:/Users/david/MT_data/masif_site_outputs/predcoords'
source_dir_pdbs = 'C:/Users/david/MT_data/alphafold_outputs/pdbs'


### Select a mutant to extract a patch from
min_fitness = min(fitness.values())
max_fitness = max(fitness.values())
scores = list(fitness.values())
mutants = list(fitness.keys())

for i in range(1500):
    
    # Choose to draw from binders or non-binders
    binder = np.random.choice([True, False])
    print(binder)
    
    if binder: 
        # draw from scores > 0.8
        draw = np.random.uniform(4.0, max_fitness)
        score = min(scores, key=lambda x:abs(x-draw))
        mutant_name = mutants[scores.index(score)]
        mutants.remove(mutant_name)
        scores.remove(score)
        print(mutant_name, fitness[mutant_name])
    else:  
        # draw from scores close to zero
        draw = np.random.uniform(min_fitness, 0.02)
        score = min(scores, key=lambda x:abs(x-draw))
        mutant_name = mutants[scores.index(score)]
        mutants.remove(mutant_name)
        scores.remove(score)
        print(mutant_name, fitness[mutant_name])

    ### Import the data corresponding to that mutant
    import fnmatch
    features_filename = fnmatch.filter(predfeatures_dir, mutant_name+'*')[0]
    predcoords_filename = fnmatch.filter(predcoords_dir, mutant_name+'*')[0]
    pdb_filename = fnmatch.filter(pdbs_dir, mutant_name+'*')[0]
    
    # Load the features of the mutants
    features = np.load(os.path.join(source_dir_feat, features_filename))
    features = features[:, 16:32]
    features = normalize_featurewise(features)

    # Load the predcoords of the mutant
    predcoords = np.load(os.path.join(source_dir_cord, predcoords_filename))

    # Parse the pdb of the mutant
    parser = PDBParser(PERMISSIVE=1)
    with open(os.path.join(source_dir_pdbs, pdb_filename)) as pdbfile: 
        gb1, atomcoords = parse_pdb(parser, mutant_name, pdbfile)


    ### Determine the center for patch extraction 
    atms27 = np.asarray(gb1[27]["atoms"])
    atms28 = np.asarray(gb1[28]["atoms"])
    atms31 = np.asarray(gb1[31]["atoms"])
    C_GLU27 = np.asarray(gb1[27]["coords"])[np.where(atms27 == 'C')]
    CA_LYS28 = np.asarray(gb1[28]["coords"])[np.where(atms28 == 'CA')]
    CA_LYS31 = np.asarray(gb1[31]["coords"])[np.where(atms31 == 'CA')]

    tolerance = 0.0
    center_coords = []
    while len(center_coords) < 1:
        for i, point in enumerate(predcoords):
            if 4.864-tolerance < np.linalg.norm(point-CA_LYS31) < 4.864+tolerance: # CA of LYS31 dist 4.864 (225)
                if 4.973-tolerance < np.linalg.norm(point-C_GLU27) < 4.973+tolerance: # C of GLU27 dist 4.973 (190)
                    if 5.072-tolerance < np.linalg.norm(point-CA_LYS28) < 5.072+tolerance: # CA of LYS28 dist 5.072 (198)
                        center_coords.append(list(predcoords[i])) 
        #print(center_coords)
        tolerance += 0.1

    center_index = np.where(predcoords == center_coords[0])[0][0]
    #print(center_index)

    # Extract the patch as graph
    coords, edge_data, A, edge_index, edge_weight, feature_matrix = extract_surface_patch_GNN(predcoords, center_index, 12, features)

    ##### For classification task #####
    if binder:
        fitness_value = 1
    else:
        fitness_value = 0
    ###################################
    
    print(fitness_value)
    print()
    patch = GraphPatch(feature_matrix, A, edge_index, edge_weight, edge_data, fitness_value, coords, mutant_name)

    os.chdir("C:\\Users\\david\\MT_data\\extracted_patches\\mutant_graphs_classification")
    filename = '{m}_GraphPatch.pkl'.format(m=mutant_name)
    save_object(patch, filename)