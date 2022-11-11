import numpy as np
import pickle

def normalize_01(array):
    array_norm = (array - np.min(array))/(np.max(array)-np.min(array))
    return array_norm

def normalize_m11(array):
    array_norm = 2*(array - np.min(array))/(np.max(array)-np.min(array))-1
    return array_norm

def normalize_columns_01(array):
    array_norm = np.zeros_like(array)
    for column in range(array.shape[1]):
        to_be_normed = array[:,column]
        column_norm = (to_be_normed - np.min(to_be_normed))/(np.max(to_be_normed)-np.min(to_be_normed))
        array_norm[:,column] = column_norm
    return array_norm

def normalize_columns_m11(array):
    array_norm = np.zeros_like(array)
    for column in range(array.shape[1]):
        to_be_normed = array[:,column]
        column_norm = 2*(to_be_normed - np.min(to_be_normed))/(np.max(to_be_normed)-np.min(to_be_normed))-1
        array_norm[:,column] = column_norm
    return array_norm


def cart_to_polar(cartesian_coords):
    if cartesian_coords.ndim == 2:
        x= cartesian_coords[:, 0]
        y = cartesian_coords[:,1]
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        polar_coords = np.column_stack((rho, theta))
    elif cartesian_coords.ndim ==1:
        x= cartesian_coords[0]
        y = cartesian_coords[1]
        rho = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        polar_coords = np.array([rho, theta])
    return polar_coords


def polar_to_cart(polar_coords):
    if polar_coords.ndim == 2:
        rho = polar_coords[:, 0]
        theta = polar_coords[:,1]
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        cart_coords = np.column_stack((x, y))
    elif polar_coords.ndim == 1:
        rho = polar_coords[0]
        theta = polar_coords[1]
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        cart_coords = np.array([x, y])
    return cart_coords


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
