# =============================================================================
# Extends: https://github.com/guanjq/targetdiff  (MIT License, © 2023 Jiaqi Guan)

# Copyright (c) 2023 Jiaqi Guan

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# From TargetDiff (minimal or no modification):
#   get_atomic_number_from_index, is_aromatic_from_index,
#   get_hybridization_from_index, get_index,
#   FeaturizeProteinAtom, FeaturizeLigandAtom, FeaturizeLigandBond, RandomRotation
#   MAP_ATOM_TYPE_FULL_TO_INDEX

# conDitar is copyrighted by the Ohio State University and covered by US 64/023,113.

# conDitar may be licensed solely for educational and research purposes by
# non-profit institutions and US government agencies only. For other proposed
# uses, contact tlcip@osu.edu. The software may not be sold or redistributed
# without prior approval.

# You may not use the software to train or process or input the software into
# or make it accessible to: automated software, services or tools, including,
# but not limited to, artificial intelligence solutions, algorithms, machine
# learning, large language models, robots, spiders, crawlers, search engines,
# text or data mining or any other aggregation functionality.

# One may make copies of the software for their use provided that the copies
# are not sold or distributed and are used under the same terms and conditions.
# As unestablished research software, this code is provided on an "as is" basis
# without warranty of any kind, either expressed or implied. The downloading or
# executing any part of this software constitutes an implicit agreement to these
# terms. These terms and conditions are subject to change at any time without
# prior notice.

# conDitar:
#   MAP_ATOM_TYPE_ONLY_TO_INDEX, MAP_ATOM_TYPE_AROMATIC_TO_INDEX,
#   MAP_INDEX_TO_BOND_TYPE, RESIDUE_INDEX_DICT, AUTODOCK, ATOM_INDEX_DICT,
#   PocketRandomMask      
# =============================================================================

import torch
import torch.nn.functional as F
import numpy as np
from utils import data as utils_data
import pdb
import random 

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

MAP_INDEX_TO_BOND_TYPE = {v: k for k, v in utils_data.BOND_TYPES.items()}

MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    9: 5,
    15: 6,
    16: 7,
    17: 8,
    35: 9,
    53: 10,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False):   0,
    (5, False):   1,
    (6, False):   2,
    (6, True):    3,
    (7, False):   4,
    (7, True):    5,
    (8, False):   6,
    (8, True):    7,
    (9, False):   8,
    (15, False):  9,
    (15, True):  10,
    (16, False): 11,
    (16, True):  12,
    (17, False): 13,
    (35, False): 14,
    (53, False): 15,
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}


ATOM_INDEX_DICT = {'H': 0, 'C': 1, 'S': 2, 'O': 3, 'N': 4, '[MASK]': 5}
# BOND_INDEX_DICT = {'C-C': 0, 'C-S': 1, 'C-O':2, 'C-N':3, 'S-S':4, 'S-O':5, 'S-N':6, 'O-O':7, 'O-N':8, 'N-N':9}
# BOND_INDEX_DICT = {'S-C': 1, 'O-C':2, 'N-C':3, 'O-S':5, 'N-S':6, 'N-O':8}
# BOND_INDEX_DICT = {'C-C': 0, 'C-S': 1, 'C-O':2, 'C-N':3, 'S-S':4, 'S-O':5, 'S-N':6, 'O-O':7, 'O-N':8, 'N-N':9}

RESIDUE_INDEX_DICT = {'D': 0,
 'K': 1,
 'M': 2,
 'B': 3,
 'U': 4,
 'Z': 5,
 'H': 6,
 'Q': 7,
 'L': 8,
 'E': 9,
 'X': 10,
 'S': 11,
 'C': 12,
 'T': 13,
 'W': 14,
 'J': 15,
 'F': 16,
 'R': 17,
 'Y': 18,
 'A': 19,
 'G': 20,
 'I': 21,
 'O': 22,
 'V': 23,
 'N': 24,
 'P': 25,
 'UNKNOWN': 26}

AUTODOCK = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53, 11, 12, 19, 20, 25, 26, 29, 30]


def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] if i < len(MAP_INDEX_TO_ATOM_TYPE_AROMATIC) else 6 for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] if i < len(MAP_INDEX_TO_ATOM_TYPE_AROMATIC) else False for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]



class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)

    def __call__(self, data):
        element_list = data.ligand_element
        hybridization_list = data.ligand_hybridization
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type - 1, num_classes=len(utils_data.BOND_TYPES))
        return data


class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        return data

class PocketRandomMask(object):
    def __init__(self, vocab,
        noise_type: str = 'normal',
        noise: float = 1.0,
        mask_prob: float = 0.5,
        unmask_prob: float = 0.0,
        mask_mode: str = 'atom',
        mask_backbone: bool = True):

        self.mask_prob_upper = mask_prob
        self.mask_prob = mask_prob
        self.unmask_prob = unmask_prob
        self.mask_mode = mask_mode

        self.noise = noise
        self.noise_type = noise_type
        self.mask_backbone = mask_backbone

        self.vocab = vocab
    
    def noise_f(self, num_mask):
        if self.noise_type == "trunc_normal":
            return np.clip(
                np.random.randn(num_mask, 3) * self.noise,
                a_min=-self.noise * 2.0,
                a_max=self.noise * 2.0,
            )
        elif self.noise_type == "normal":
            return np.random.randn(num_mask, 3) * self.noise
        elif self.noise_type == "uniform":
            return np.random.uniform(
                low=-self.noise, high=self.noise, size=(num_mask, 3)
            )
        else:
            return 0.0

    def __call__(self, data):
        self.mask_prob = self.mask_prob_upper
        if self.mask_mode == 'residue':
            residue = data.pocket_residue_index.tolist()
            sz = len(residue)
            res_list = list(set(residue))
            res_sz = len(res_list)

            num_mask = int(
                    self.mask_prob * res_sz
                    + np.random.rand()
                )

            mask_res = np.random.choice(res_list, num_mask, replace=False).tolist()
            mask = np.isin(residue, mask_res)
        elif self.mask_mode == 'atom':
            atom = data.pocket_element.tolist()
            sz = len(atom)
            num_mask = int(
                    self.mask_prob * sz
                    + np.random.rand()
                )
            mask = np.zeros(sz, dtype=bool)
            mask_indices = np.random.choice(sz, num_mask, replace=False)
            mask[mask_indices] = True
        else:
            raise ValueError(f'Not supported mask mode: {self.mask_mode}')
        
        if self.mask_backbone == False:
            mask = mask & np.array((data.pocket_is_backbone == False).tolist())
            
        if self.unmask_prob > 0:
            rand_unmask = mask & (np.random.rand(sz) < self.unmask_prob)
        else:
            rand_unmask = None

        coordinates = data.pocket_coordinate.numpy()
        num_mask = mask.astype(np.int32).sum()

        coordinates_new = np.copy(coordinates)
        coordinates_new[mask, :] += self.noise_f(num_mask)
        atoms = data.pocket_element.numpy()
        atoms_new = np.copy(atoms)
        atoms_new[mask] = self.vocab['[MASK]']
        residue_type_new = np.copy(data.pocket_residue_type.numpy())
        residue_type_new[mask] = 20

        if rand_unmask is not None:
            num_rand = rand_unmask.sum()
            if num_rand > 0:
                atoms_new[rand_unmask] = np.random.choice(
                    len(self.vocab)-1,
                    num_rand,
                )
                residue_type_new[rand_unmask] = np.random.choice(
                    20,
                    num_rand,
                )


        data_new = {}
        data_new['coordinate'] = coordinates
        data_new['corrupted_coordinate'] = coordinates_new
        data_new['element'] = atoms 
        data_new['corrupted_element'] = atoms_new
        data_new['residue_type'] = data.pocket_residue_type
        if self.mask_mode == 'residue':
            data_new['corrupted_residue_type'] = residue_type_new
        else:
            data_new['corrupted_residue_type'] = data.pocket_residue_type
        data_new['residue_index'] = data.pocket_residue_index
        data_new['atom_name'] = data.pocket_atom_name
        data_new['is_backbone'] = data.pocket_is_backbone

        try:
            data_new['embd_scalar'] = data.pocket_embd_scalar
            data_new['embd_vector'] = data.pocket_embd_vector
        except: 
            return data_new, torch.from_numpy(mask)
        
        return data_new, torch.from_numpy(mask)
        

