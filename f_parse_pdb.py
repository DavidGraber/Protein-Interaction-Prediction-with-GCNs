from Bio.PDB.PDBParser import PDBParser
import numpy as np

def parse_pdb(parser, protein_id, file, chain_number=0, numbering_offset=0):
    
    protein_id = parser.get_structure(protein_id, file)
    models = list(protein_id.get_models())
    chains = list(models[0].get_chains())
    residues = list(chains[chain_number].get_residues())
    
    protein = {p+1+numbering_offset:{} for p in range(len(residues))}
    
    prot_coords = np.zeros((1,3))
    atom_index = 0
    for i, residue in enumerate(residues):
        protein[i+1+numbering_offset]["resname"] = residue.resname
        atoms = list(residue.get_atoms())
        protein[i+1+numbering_offset]["atoms"]=[at.get_name() for at in atoms]
        protein[i+1+numbering_offset]["atom_indeces"] = list(range(atom_index, atom_index + len(residue)))
        atom_index += len(residue)
        coords = []
        for atom in atoms: 
            coord = list(atom.get_vector())
            coords.append(coord)
        protein[i+1+numbering_offset]["coords"] = np.asarray(coords)
        prot_coords = np.concatenate((prot_coords, np.asarray(coords)))

    prot_coords = prot_coords[1:]
            
    return protein, prot_coords