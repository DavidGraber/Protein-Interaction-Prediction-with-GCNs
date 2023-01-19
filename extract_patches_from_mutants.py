from Bio.PDB.PDBParser import PDBParser
import numpy as np
from f_parse_pdb import parse_pdb
import os
from f_extract_surface_patch import *
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
print(len(mutants))

for mutant_name in mutants:
    
    ### Import the data corresponding to that mutant
    import fnmatch
    features_filename = fnmatch.filter(predfeatures_dir, mutant_name+'*')[0]
    predcoords_filename = fnmatch.filter(predcoords_dir, mutant_name+'*')[0]
    pdb_filename = fnmatch.filter(pdbs_dir, mutant_name+'*')[0]
    
    # Load the features of the mutants
    features = np.load(os.path.join(source_dir_feat, features_filename))
    features = features[:, 16:32]
    os.chdir('c:\\Users\\david\\pyproj\\pyg\\mt') 
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
        tolerance += 0.1

    center_index = np.where(predcoords == center_coords[0])[0][0]
    center_x, center_y, center_z = center_coords[0]


    # Extract the patch as graph
    pos, edge_index, edge_attr, x = extract_surface_patch_GCN(predcoords, center_index, 12, features)
    
    # Normalize pos by the radius of the geodesic patch and write coordinates wrt center of patch
    pos[:,0]=(pos[:,0]-center_x)/12
    pos[:,1]=(pos[:,1]-center_y)/12
    pos[:,2]=(pos[:,2]-center_z)/12
    
    y = fitness[mutant_name]

    patch = GraphPatch(x, edge_index, edge_attr, y, pos, mutant_name)

    os.chdir("C:\\Users\\david\\MT_data\\extracted_patches\\mutant_graphs_all")
    filename = '{m}_GraphPatch.pkl'.format(m=mutant_name)
    save_object(patch, filename)