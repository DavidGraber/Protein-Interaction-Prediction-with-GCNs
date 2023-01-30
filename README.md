![Picture1](https://user-images.githubusercontent.com/112872366/213405532-a3c62343-a841-4241-a93c-60a1c307a8a6.png)

# Protein Interaction Prediction with Graph Convolutional Networks

This repository contains the code of three graph neural network models that apply graph convolution and various graph pooling methods to generate low-dimensional embeddings of complex protein surfaces modelled as graphs and classify them into:

- binding to IgG
- not binding to IgG

Using a dataset of protein G subunit B1 (GB1) mutants with experimentally determined binding affinities, we extract a surface patch from the IgG-binding region of all mutants. This patch contains the main interacting amino acid residues that are responsible for the binding interaction in the WT GB1 protein. To retain the full scale of the geometric properties of the surface patches, the patches are modelled as graphs and are used to train a Convolutional Graph Neural Network to create low-dimensional embeddings that are useful to predict the IgG-binding capacity of each mutant. 

***

## Input Data Generation
- Genomic Sequences of the GB1 protein is randomly mutated at four selected sites (39, 40, 41, 54) to generate 23'241 mutants. The WT GB1 protein binds to the constant region of human IgG with high affinity. Experimentally measured binding affinities of the mutants have been generated by Wu et al. (https://elifesciences.org/articles/16965)
- The 3D-structure of the mutants was predicted with ColabFold, an adapted version of AlphaFold2 (https://github.com/sokrypton/ColabFold). 
- Using the resulting PDB files, a molecular surface with geometric and chemical features was computed using dMaSIF (https://github.com/FreyrS/dMaSIF). The output is a point cloud describing a protein surface, where each point has been assigned a vector of 16 learned features.

<p align="center">
  <img src="https://user-images.githubusercontent.com/112872366/213655092-bc23b95f-d2e7-4a8b-b5e6-e479c74c3927.PNG">
</p>

## Patch Extraction 
The following code is used to extract circular graph patches from all mutants and save them in a custom _GraphPatch_ python Class 

#### **extract_patches_from_mutants.py:**
> This file contains code that uses the followin data that was generated for all the mutants
>- The PDB-files generated with ColabFold (contains the 3D coordinates of all atoms of the protein)
>- The fitness/affinity values (saved in a dictionary _fitness_dict_short.npy_)
>- The 3D-coordinates of the points in the computed surface point cloud (_predcoords_)
>- The features assigned to each point in the surface point cloud (_predfeatures_)
>
>This data is used to extract a surface patch from the binding region of GB1 and modell this patch as a graph. For each mutant, the following steps are performed:
>1. Import the above described data of the mutant
>2. Parse the PDB file of the mutant with BioPython, locate the IgG-binding region (see f_parse_pdb.py)
>3. Extract a circular surface patch with fixed geodesic radius from this region and model it as a graph compatible with the PyTorch geometric library, i.e. with a feature matrix x, an edge index and edge weights correponding to the geodesic distance between the points the edge is connecting (see the function _extract_surface_patch_GCN_ in f_extract_surface_patch.py)
>4. Save it as an instance of the custom _GraphPatch_ class together with the corresponding fitness/affinity value (see c_GraphPatch.py)

#### f_extract_surface_patch.py
> This file contains a set of functions that take the surface point cloud with its features and coordinates, the center for patch extraction and the desired          geodesic   radius and returns the a circular patch of the given geodesic radius drawn around the given patch center and modelled as a graph with edge index,        adjacency matrix,     edge_weights and features. The following steps are performed:
> 1. All points that have a Euclidean distance greater than the given radius are removed from the pointcloud
> 2. From the remaining points, a simplified graph is generated by connecting each point with its 10 nearest neighbors and calculating the geodesic distance between    the connected points. All connections are saved in a dictionary together with their geodesic distances. 
> 3. For each point, Djikstra algorithm is used to compute the geodesic distance to the center. Points that are more than the desired radius away from the center are   removed. 
> 4. The connections between the remaining points are saved in an adjacency matrix, the softmin(geodesic distances) in a weight matrix, the features in a feature       matrix and the coordinates in a pos matrix. 

#### f_parse_pdb.py
>This function uses the PDBParser from BioPython to import a PDB file and extract the residues, the atom types and atom coordinates of the protein. The residues with their atoms are saved in a double dictionary, the atom coords are saved in a numpy array.'''

#### f_extract_surface_patch_padded.py & extract_patches_from_mutants_padded.py
> These files are identical to the unpadded version described above, except that they contain an additional code for padding graphs. All graphs were padded until they reached the number of nodes of the largest graph in the dataset. If a graph was smaller than the largest graph in the dataset, additional points were introduced between two randomly selected connected points of the graph until the graph reached the desired size. The coordinates and the surface normal of the new point were computed as a weighted average from its nearest neighbors. The features of the new point were computed using the knn interpolate function of the PyTorch Geometric library. 

#### c_GraphPatch.py
>Definition of a custom python Class designed to store all relevant data to describe an extracted surface patch graph, including coordinates of the graph nodes (pos), the features (x), the adjacency information (edge_index), the edge weights (edge_attr) and the fitness (y) and the name of the mutant'''

<p align="center">
  <img src="https://user-images.githubusercontent.com/112872366/213656085-c5e943ad-4ecd-4ea8-9a50-05bb12e075f4.png">
</p>

***

## Dataset Generation
To import the generated _GraphPatch_ instances for training a graph neural network, a custom _PatchDataset_ class was generated that subclasses the _torch_geometric.data.Dataset_ class. The following code generates a this _PatchDataset_ class:

#### c_PatchDataset_sparse.py 
>- In the _init_ function, the class is initialized with the path to the _GraphPatch_ objects saved as an instance variable.
>- The _getitem_ function gets a _GraphPatch_ with a given index from this directory and 
>- checks the fitness value of the imported patch and adds a classification label y to the patch according to the value of the fitness. 
>- returns the patch packaged in a _torch.geometric.data.Data_ object holding a feature matrix x, an edge index, edge weights and a label y. 

#### c_PatchDataset_dense.py
> This is the _PatchDataset_ class used for importing graphs for models that operate on dense representations of graphs. It works identically to the above-described sparse version of the _PatchDatase_ class, except that the _getitem_ function additionally converts the sparse edge index and edge weights into the corresponding dense adjacency and weight matrices using the function to dense adj function of the PyTorch Geometric library.

***

## Models

#### model_baseline.ipynb
> The baseline graph convolutional network (GCN). The model consists of three consecutive _GCNConv_ convolutional layers and a final fully connected layer, which are initialized in the init method of the model. As all four layers apply linear transformations to the input data, a layer-specific number of input and output feature maps is defined as input to the model initialization. In the forward function:
>- The feature matrix x is passed through and modified by the convolutional layers. The sparse representations of the adjacency matrix (edge index) and the edge weight matrix (edge weight) serve as inputs to the layers, but are not modified. 
>- Between the convolutional layers, the data is subjected to the nonlinear rectified linear activation (ReLU) function, which performs ReLU(v) = max(0, v). 
>- The feature matrix is reduced to one dimension by global max pooling. 
>- The resulting vector is subjected to a dropout layer 
>- The fully connected classifier layer then reduces the dimensionality to the number of classes. 
>- The two-channel output is subjected to a log-softmax function to compute the log-probabilities of class-membership.

#### model_diffpool.ipynb
> The DiffPool model, which employs a hierarchical graph coarsening strategy called Diffpool. The model comprises four hierarchical coarsening steps. Each step performs the following operations: 
>- Firstly, a pooling GCN computes from the feature matrix x a cluster assignment matrix s with layer-specific number of output clusters. 
>- Secondly, an embedding x' GCN generates a new feature embedding with layer-specific feature dimensionality. 
>- Thirdly, the two generated matrices are combined to compute a pooled node feature matrix x′ and a coarsened adjacency matrix adj′ in the _dense_diff_pool_ function of the PyTorch Geometric library. 

>In summary, each step learns a new feature embedding and a cluster assignment matrix and processes these to generate a coarsened adjacency and feature matrix as input for the next layer. The last layer assigns all nodes to a single cluster, which corresponds to a single node with a feature vector representing an embedding of the entire graph.

>The model operates with dense representations of adjacency matrices and thus the dataset of padded graphs was used for the training of this model. As the standard _GCNConv_ layer operates on sparse matrices, the dense version of this convolutional layer was used (_DenseGCNConv_). 

>The architecture includes a JumpingKnowledge layer-aggregation mechanism proposed by Xu et al. [29]. This introduces jump connections to the model and selectively combines different aggregations at the last layer. For this, representations are generated at each layer of the model by global max and mean pooling and appended to a readout list, which is then aggregated and passed to the last fully connected classifier layer. Therefore, the classifier not only exploits the final 1D-representation of the graph generated in the last pooling layer, but also intermediate representations that have ”jumped” to the last layer. This final classifier employs a dropout layer, a batch normalization layer and a final log-softmax function to compute log-probabilities of class membership.


#### model_asapool.ipynb
> The ASAPool model, which employs a more complex clustering with an attention mechanism. Similar to the DiffPool model described in section 3.4.6, four hierarchical coarsening steps are implemented before a prediction is derived with a fully connected classifier. Each coarsening step involves:
>- A convolutional _GCNConv_ layer with a layer-specific number of output channels
>- An ASAPool layer, which reduces the number nodes to N<sup>L+1</sup> = ratio · N output nodes and applies a linear transformation to increase the feature dimensionality to the number of output channels.

>The last coarsening step reduces the graph to a single node with a feature vector representing an embedding of the entire graph. As described for the DiffPool model in section 3.4.6, a JumpingKnowledge layer aggregation mechanism is implemented, which introduces jump connections to include intermediate representations in the input to the final classification network. This final classifier employs a dropout, batch normalization and a final log-softmax function layer to compute log-probabilities of class membership.


## Additional Code

#### f_helper_functions.py
>Contains small functions that are used throughout the code, e.g. 
>- _normalize_featurewise_ is used to normalize feature matrices between -1 and 1 in a column-wise manner, so that each feature is normalized independently from the other features in the dataset
>- _save_object_ and _load_object_ are used to save and load instances of the _GraphPatch_ class as pickle files. 


---------------------------------------------------------------------------------------------------


## Instructions for creating virtual environment with all required packages: 

- Download python version 3.9.13, custom installation, save it somewhere at a defined place
- Create the venv in the folder where the project files are with path to the downloaded python 3.9.13
- Activate environment 

Install all the required packages: 

- pip install numpy
- pip install -U matplotlib
- pip install -U scikit-learn
- pip install pandas
- pip install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cpu
- pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
- pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
- pip install torch-geometric
- pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cpu.html

In the console, type pip freeze to show all the packages that were installed and their versions (requirements.txt)

This combination of pytorch 1.12.0 with pytorch geometric works fine for me, many other combinations have not worked. This is the cpu installation, see the following section for GPU.

### When training models on GPU conda version 11.6 (to get cuda version: nvcc --version)

- pip install numpy
- pip install -U matplotlib
- pip install -U scikit-learn
- pip install pandas
- pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
- pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
- pip install torch-geometric
- pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0+cu116.html


------------------------------------------------------------------------------------------------------






