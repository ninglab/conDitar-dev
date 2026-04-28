import torch
import torch.nn.functional as F
import numpy as np
#from datasets.shape_mol_data import ShapeMolData
from utils import data as utils_data
import pdb
import random 

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

MAP_INDEX_TO_BOND_TYPE = {v: k for k, v in utils_data.BOND_TYPES.items()}

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
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
    """ randomly mask atoms in pockets
        please use https://github.com/dptech-corp/Uni-Mol/blob/97797d53dd4be20a9dd79024fcac159a784cafce/unimol/unimol/data/mask_points_dataset.py#L138
        as a reference
    """
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
        

# class LigandCountNeighbors(object):
#
#     @staticmethod
#     def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
#         assert symmetry == True, 'Only support symmetrical edges.'
#
#         if num_nodes is None:
#             num_nodes = maybe_num_nodes(edge_index)
#
#         if valence is None:
#             valence = torch.ones([edge_index.size(1)], device=edge_index.device)
#         valence = valence.view(edge_index.size(1))
#
#         return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()
#
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, data):
#         data.ligand_num_neighbors = self.count_neighbors(
#             data.ligand_bond_index,
#             symmetry=True,
#             num_nodes=data.ligand_element.size(0),
#         )
#         data.ligand_atom_valence = self.count_neighbors(
#             data.ligand_bond_index,
#             symmetry=True,
#             valence=data.ligand_bond_type,
#             num_nodes=data.ligand_element.size(0),
#         )
#         return data
#
#
# class LigandRandomMask(object):
#
#     def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0):
#         super().__init__()
#         self.min_ratio = min_ratio
#         self.max_ratio = max_ratio
#         self.min_num_masked = min_num_masked
#         self.min_num_unmasked = min_num_unmasked
#
#     def __call__(self, data:ProteinLigandData):
#         ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
#         num_atoms = data.ligand_element.size(0)
#         num_masked = int(num_atoms * ratio)
#
#         if num_masked < self.min_num_masked:
#             num_masked = self.min_num_masked
#         if (num_atoms - num_masked) < self.min_num_unmasked:
#             num_masked = num_atoms - self.min_num_unmasked
#
#         idx = np.arange(num_atoms)
#         np.random.shuffle(idx)
#         idx = torch.LongTensor(idx)
#         masked_idx = idx[:num_masked]
#         context_idx = idx[num_masked:]
#
#         data.ligand_masked_element = data.ligand_element[masked_idx]
#         data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction
#         data.ligand_masked_pos = data.ligand_pos[masked_idx]
#
#         data.ligand_context_element = data.ligand_element[context_idx]
#         data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
#         data.ligand_context_pos = data.ligand_pos[context_idx]
#
#         data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
#             context_idx,
#             data.ligand_bond_index,
#             edge_attr = data.ligand_bond_feature,
#             relabel_nodes = True,
#         )
#         data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
#             data.ligand_context_bond_index,
#             symmetry=True,
#             num_nodes = context_idx.size(0),
#         )
#
#         # print(context_idx)
#         # print(data.ligand_context_bond_index)
#
#         # mask = torch.logical_and(
#         #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
#         #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
#         # )
#         # print(data.ligand_bond_index[:, mask])
#
#         # print(data.ligand_context_num_neighbors)
#         # print(data.ligand_num_neighbors[context_idx])
#
#
#         data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]
#
#         data._mask = 'random'
#
#         return data
#
#
# class LigandBFSMask(object):
#
#     def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, inverse=False):
#         super().__init__()
#         self.min_ratio = min_ratio
#         self.max_ratio = max_ratio
#         self.min_num_masked = min_num_masked
#         self.min_num_unmasked = min_num_unmasked
#         self.inverse = inverse
#
#     @staticmethod
#     def get_bfs_perm(nbh_list):
#         num_nodes = len(nbh_list)
#         num_neighbors = torch.LongTensor([len(nbh_list[i]) for i in range(num_nodes)])
#
#         bfs_queue = [random.randint(0, num_nodes-1)]
#         bfs_perm = []
#         num_remains = [num_neighbors.clone()]
#         bfs_next_list = {}
#         visited = {bfs_queue[0]}
#
#         num_nbh_remain = num_neighbors.clone()
#
#         while len(bfs_queue) > 0:
#             current = bfs_queue.pop(0)
#             for nbh in nbh_list[current]:
#                 num_nbh_remain[nbh] -= 1
#             bfs_perm.append(current)
#             num_remains.append(num_nbh_remain.clone())
#             next_candid = []
#             for nxt in nbh_list[current]:
#                 if nxt in visited: continue
#                 next_candid.append(nxt)
#                 visited.add(nxt)
#
#             random.shuffle(next_candid)
#             bfs_queue += next_candid
#             bfs_next_list[current] = copy.copy(bfs_queue)
#
#         return torch.LongTensor(bfs_perm), bfs_next_list, num_remains
#
#     def __call__(self, data):
#         bfs_perm, bfs_next_list, num_remaining_nbs = self.get_bfs_perm(data.ligand_nbh_list)
#
#         ratio = np.clip(random.uniform(self.min_ratio, self.max_ratio), 0.0, 1.0)
#         num_atoms = data.ligand_element.size(0)
#         num_masked = int(num_atoms * ratio)
#         if num_masked < self.min_num_masked:
#             num_masked = self.min_num_masked
#         if (num_atoms - num_masked) < self.min_num_unmasked:
#             num_masked = num_atoms - self.min_num_unmasked
#
#         if self.inverse:
#             masked_idx = bfs_perm[:num_masked]
#             context_idx = bfs_perm[num_masked:]
#         else:
#             masked_idx = bfs_perm[-num_masked:]
#             context_idx = bfs_perm[:-num_masked]
#
#         data.ligand_masked_element = data.ligand_element[masked_idx]
#         data.ligand_masked_feature = data.ligand_atom_feature[masked_idx]   # For Prediction
#         data.ligand_masked_pos = data.ligand_pos[masked_idx]
#
#         data.ligand_context_element = data.ligand_element[context_idx]
#         data.ligand_context_feature_full = data.ligand_atom_feature_full[context_idx]   # For Input
#         data.ligand_context_pos = data.ligand_pos[context_idx]
#
#         data.ligand_context_bond_index, data.ligand_context_bond_feature = subgraph(
#             context_idx,
#             data.ligand_bond_index,
#             edge_attr = data.ligand_bond_feature,
#             relabel_nodes = True,
#         )
#         data.ligand_context_num_neighbors = LigandCountNeighbors.count_neighbors(
#             data.ligand_context_bond_index,
#             symmetry=True,
#             num_nodes = context_idx.size(0),
#         )
#
#         # print(context_idx)
#         # print(data.ligand_context_bond_index)
#
#         # mask = torch.logical_and(
#         #     (data.ligand_bond_index[0].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
#         #     (data.ligand_bond_index[1].view(-1, 1) == context_idx.view(1, -1)).any(dim=-1),
#         # )
#         # print(data.ligand_bond_index[:, mask])
#
#         # print(data.ligand_context_num_neighbors)
#         # print(data.ligand_num_neighbors[context_idx])
#
#         data.ligand_frontier = data.ligand_context_num_neighbors < data.ligand_num_neighbors[context_idx]
#
#         data._mask = 'invbfs' if self.inverse else 'bfs'
#
#         return data
#
#
# class LigandMaskAll(LigandRandomMask):
#
#     def __init__(self):
#         super().__init__(min_ratio=1.0)
#
#
# class LigandMixedMask(object):
#
#     def __init__(self, min_ratio=0.0, max_ratio=1.2, min_num_masked=1, min_num_unmasked=0, p_random=0.5, p_bfs=0.25, p_invbfs=0.25):
#         super().__init__()
#
#         self.t = [
#             LigandRandomMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked),
#             LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=False),
#             LigandBFSMask(min_ratio, max_ratio, min_num_masked, min_num_unmasked, inverse=True),
#         ]
#         self.p = [p_random, p_bfs, p_invbfs]
#
#     def __call__(self, data):
#         f = random.choices(self.t, k=1, weights=self.p)[0]
#         return f(data)
#
#
# def get_mask(cfg):
#     if cfg.type == 'bfs':
#         return LigandBFSMask(
#             min_ratio=cfg.min_ratio,
#             max_ratio=cfg.max_ratio,
#             min_num_masked=cfg.min_num_masked,
#             min_num_unmasked=cfg.min_num_unmasked,
#         )
#     elif cfg.type == 'random':
#         return LigandRandomMask(
#             min_ratio=cfg.min_ratio,
#             max_ratio=cfg.max_ratio,
#             min_num_masked=cfg.min_num_masked,
#             min_num_unmasked=cfg.min_num_unmasked,
#         )
#     elif cfg.type == 'mixed':
#         return LigandMixedMask(
#             min_ratio=cfg.min_ratio,
#             max_ratio=cfg.max_ratio,
#             min_num_masked=cfg.min_num_masked,
#             min_num_unmasked=cfg.min_num_unmasked,
#             p_random = cfg.p_random,
#             p_bfs = cfg.p_bfs,
#             p_invbfs = cfg.p_invbfs,
#         )
#     elif cfg.type == 'all':
#         return LigandMaskAll()
#     else:
#         raise NotImplementedError('Unknown mask: %s' % cfg.type)
#
#
# class ContrastiveSample(object):
#
#     def __init__(self, num_real=50, num_fake=50, pos_real_std=0.05, pos_fake_std=2.0, elements=None):
#         super().__init__()
#         self.num_real = num_real
#         self.num_fake = num_fake
#         self.pos_real_std = pos_real_std
#         self.pos_fake_std = pos_fake_std
#         if elements is None:
#             # elements = torch.LongTensor([
#             #     1, 3, 5, 6, 7, 8, 9,
#             #     12, 13, 14, 15, 16, 17,
#             #     21, 23, 24, 26, 27, 29, 33, 34, 35,
#             #     39, 42, 44, 50, 53, 74, 79, 80
#             # ])
#             elements = [1,6,7,8,9,15,16,17]
#         self.elements = torch.LongTensor(elements)
#
#     @property
#     def num_elements(self):
#         return self.elements.size(0)
#
#     def __call__(self, data:ProteinLigandData):
#         # Positive samples
#         pos_real_mode = data.ligand_masked_pos
#         element_real = data.ligand_masked_element
#         ind_real = data.ligand_masked_feature
#         cls_real = data.ligand_masked_element.view(-1, 1) == self.elements.view(1, -1)
#         assert (cls_real.sum(-1) > 0).all(), 'Unexpected elements.'
#
#         real_sample_idx = np.random.choice(np.arange(pos_real_mode.size(0)), size=self.num_real)
#         data.pos_real = pos_real_mode[real_sample_idx]
#         data.pos_real += torch.randn_like(data.pos_real) * self.pos_real_std
#         data.element_real = element_real[real_sample_idx]
#         data.cls_real = cls_real[real_sample_idx]
#         data.ind_real = ind_real[real_sample_idx]
#
#         # Negative samples
#         pos_fake_mode = torch.cat([data.ligand_context_pos, data.protein_pos], dim=0)
#         fake_sample_idx = np.random.choice(np.arange(pos_fake_mode.size(0)), size=self.num_fake)
#         data.pos_fake = pos_fake_mode[fake_sample_idx]
#         data.pos_fake += torch.randn_like(data.pos_fake) * self.pos_fake_std
#
#         return data
#
#
# def get_contrastive_sampler(cfg):
#     return ContrastiveSample(
#         num_real = cfg.num_real,
#         num_fake = cfg.num_fake,
#         pos_real_std = cfg.pos_real_std,
#         pos_fake_std = cfg.pos_fake_std,
#     )
