import numpy as np
import pickle


def normalize_featurewise(array):

    '''Used to normalize feature matrices between -1 and 1 in a column-wise manner, so that each feature 
    is normalized independently from the other features in the dataset. In this min-max scaling, the min and
    max values represent the lowest and highest values found for this feature in the complete dataset'''

    if array.shape[1] != 16: 
        raise Exception("Array must be of shape points x 16")

    #Import a dictionary where the upper and lower limits of all 16 features are saved
    limits = np.load('data/feature_limits_dict.npy', allow_pickle=True).item()

    for feature in range(16): 
        to_be_normed = array[:,feature]
        column_norm = 2*(to_be_normed - limits[feature]['min'])/(limits[feature]['max'] - limits[feature]['min']) - 1
        #column_norm = (to_be_normed - limits[feature]['min'])/(limits[feature]['max'] - limits[feature]['min'])
        array[:,feature] = column_norm
    return array




def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
