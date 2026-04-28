"""Utils for sampling size of a molecule of a given protein pocket."""

import numpy as np
from scipy import spatial as sc_spatial

from utils.atom_num_config import CONFIG

def get_interaction_size(pocket_3d_pos, ligand_3d_pos):
    dists = sc_spatial.distance.cdist(pocket_3d_pos, ligand_3d_pos, metric='euclidean')
    closest_indices = np.argsort(dists, axis=0)[:ligand_3d_pos.shape[0]*3]
    closest_pocket_indices = np.unique(closest_indices.flatten())
    selected_pocket_points = pocket_3d_pos[closest_pocket_indices]
    aa_dist = sc_spatial.distance.pdist(selected_pocket_points, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.max(aa_dist)


def get_space_size(pocket_3d_pos):
    aa_dist = sc_spatial.distance.pdist(pocket_3d_pos, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.median(aa_dist[:10])


def _get_bin_idx(space_size):
    bounds = CONFIG['bounds']
    for i in range(len(bounds)):
        if bounds[i] > space_size:
            return i
    return len(bounds)


def sample_atom_num(space_size):
    bin_idx = _get_bin_idx(space_size)
    num_atom_list, prob_list = CONFIG['bins'][bin_idx]
    prob_list = np.array(prob_list)
    prob_list = prob_list / prob_list.sum()
    return np.random.choice(num_atom_list, p=prob_list)

