# =============================================================================
# From: https://github.com/guanjq/targetdiff  
#
# MIT License
#
# Copyright (c) 2023 Jiaqi Guan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

import numpy as np
from scipy import spatial as sc_spatial

from utils.atom_num_config import CONFIG


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


# =============================================================================
# conDitar is copyrighted by the Ohio State University and covered by US 64/023,113.
#
# conDitar may be licensed solely for educational and research purposes by
# non-profit institutions and US government agencies only. For other proposed
# uses, contact tlcip@osu.edu. The software may not be sold or redistributed
# without prior approval.
#
# You may not use the software to train or process or input the software into
# or make it accessible to: automated software, services or tools, including,
# but not limited to, artificial intelligence solutions, algorithms, machine
# learning, large language models, robots, spiders, crawlers, search engines,
# text or data mining or any other aggregation functionality.
#
# One may make copies of the software for their use provided that the copies
# are not sold or distributed and are used under the same terms and conditions.
# As unestablished research software, this code is provided on an "as is" basis
# without warranty of any kind, either expressed or implied. The downloading or
# executing any part of this software constitutes an implicit agreement to these
# terms. These terms and conditions are subject to change at any time without
# prior notice.
# =============================================================================

def get_interaction_size(pocket_3d_pos, ligand_3d_pos):
    dists = sc_spatial.distance.cdist(pocket_3d_pos, ligand_3d_pos, metric='euclidean')
    closest_indices = np.argsort(dists, axis=0)[:ligand_3d_pos.shape[0]*3]
    closest_pocket_indices = np.unique(closest_indices.flatten())
    selected_pocket_points = pocket_3d_pos[closest_pocket_indices]
    aa_dist = sc_spatial.distance.pdist(selected_pocket_points, metric='euclidean')
    aa_dist = np.sort(aa_dist)[::-1]
    return np.max(aa_dist)

