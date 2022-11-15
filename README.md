## Protein Interaction Prediction with Geometric Deep Learning

Applies Graph Neural Networks to potentially binding protein surfaces to compare the complementarity of their surface features. 

- Input Data: Sequence of 23'400 mutants of GB1 protein (protein G subunit B1), whose structures were predicted with AlphaFold. Around each mutant, a discretized 
molecular surface with geometric and chemical features has been computed (see https://github.com/FreyrS/dMaSIF). The WT GB1 protein binds to the constant region of human IgG with high affinity. 
The GB1 variants are randomly mutated at four selected sites. Data of the mutant's binding affinity to IgG is taken from laboratory measurements (see https://elifesciences.org/articles/16965) 

- Aim: Extract a surface patch from the IgG-binding region of all mutants. This patch should contain the main interacting amino acid residues that are
responsible for the binding interaction in the WT GB1 protein. The same patch is extracted from all mutants. Patches are modelled as graphs and are used to train 
a Convolutional Graph Neural Network to decrease the feature dimensionality and predict the IgG-binding capacity of each mutant. 
