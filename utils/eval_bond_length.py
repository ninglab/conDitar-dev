# =============================================================================
# From: https://github.com/guanjq/targetdiff

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
# =============================================================================

"""Utils for evaluating bond length."""

import collections
from typing import Tuple, Sequence, Dict, Optional

import numpy as np
from scipy import spatial as sci_spatial
import matplotlib.pyplot as plt

from utils import eval_bond_length_config
import utils.data as utils_data

BondType = Tuple[int, int, int]  # (atomic_num, atomic_num, bond_type)
BondLengthData = Tuple[BondType, float]  # (bond_type, bond_length)
BondLengthProfile = Dict[BondType, np.ndarray]  # bond_type -> empirical distribution


def get_distribution(distances: Sequence[float], bins=eval_bond_length_config.DISTANCE_BINS) -> np.ndarray:
    """Get the distribution of distances.

    Args:
        distances (list): List of distances.
        bins (list): bins of distances
    Returns:
        np.array: empirical distribution of distances with length equals to DISTANCE_BINS.
    """
    bin_counts = collections.Counter(np.searchsorted(bins, distances))
    bin_counts = [bin_counts[i] if i in bin_counts else 0 for i in range(len(bins) + 1)]
    bin_counts = np.array(bin_counts) / np.sum(bin_counts)
    return bin_counts


def _format_bond_type(bond_type: BondType) -> BondType:
    atom1, atom2, bond_category = bond_type
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1
    return atom1, atom2, bond_category


def get_bond_length_profile(bond_lengths: Sequence[BondLengthData]) -> BondLengthProfile:
    bond_length_profile = collections.defaultdict(list)
    for bond_type, bond_length in bond_lengths:
        bond_type = _format_bond_type(bond_type)
        bond_length_profile[bond_type].append(bond_length)
    bond_length_profile = {k: get_distribution(v) for k, v in bond_length_profile.items()}
    return bond_length_profile


def _bond_type_str(bond_type: BondType) -> str:
    atom1, atom2, bond_category = bond_type
    return f'{atom1}-{atom2}|{bond_category}'


def eval_bond_length_profile(bond_length_profile: BondLengthProfile) -> Dict[str, Optional[float]]:
    metrics = {}

    # Jensen-Shannon distances
    for bond_type, gt_distribution in eval_bond_length_config.EMPIRICAL_DISTRIBUTIONS.items():
        if bond_type not in bond_length_profile:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = None
        else:
            metrics[f'JSD_{_bond_type_str(bond_type)}'] = sci_spatial.distance.jensenshannon(gt_distribution,
                                                                                             bond_length_profile[
                                                                                                 bond_type])

    return metrics


def get_pair_length_profile(pair_lengths):
    cc_dist = [d[1] for d in pair_lengths if d[0] == (6, 6) and d[1] < 2]
    all_dist = [d[1] for d in pair_lengths if d[1] < 12]
    pair_length_profile = {
        'CC_2A': get_distribution(cc_dist, bins=np.linspace(0, 2, 100)),
        'All_12A': get_distribution(all_dist, bins=np.linspace(0, 12, 100))
    }
    return pair_length_profile


def eval_pair_length_profile(pair_length_profile):
    metrics = {}
    for k, gt_distribution in eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS.items():
        if k not in pair_length_profile:
            metrics[f'JSD_{k}'] = None
        else:
            metrics[f'JSD_{k}'] = sci_spatial.distance.jensenshannon(gt_distribution, pair_length_profile[k])
    return metrics


def plot_distance_hist(pair_length_profile, metrics=None, save_path=None):
    gt_profile = eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS
    plt.figure(figsize=(6 * len(gt_profile), 4))

    for idx, (k, gt_distribution) in enumerate(eval_bond_length_config.PAIR_EMPIRICAL_DISTRIBUTIONS.items()):
        plt.subplot(1, len(gt_profile), idx + 1)
        x = eval_bond_length_config.PAIR_EMPIRICAL_BINS[k]
        plt.step(x, gt_profile[k][1:])
        plt.step(x, pair_length_profile[k][1:])
        plt.legend(['True', 'Learned'])
        if metrics is not None:
            plt.title(f'{k} JS div: {metrics["JSD_" + k]:.4f}')
        else:
            plt.title(k)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def pair_distance_from_pos_v(pos, elements):
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    dist_list = []
    for s in range(len(pos)):
        for e in range(s + 1, len(pos)):
            s_sym = elements[s]
            e_sym = elements[e]
            d = pdist[s, e]
            dist_list.append(((s_sym, e_sym), d))
    return dist_list


def bond_distance_from_mol(mol):
    pos = mol.GetConformer().GetPositions()
    pdist = pos[None, :] - pos[:, None]
    pdist = np.sqrt(np.sum(pdist ** 2, axis=-1))
    all_distances = []
    for bond in mol.GetBonds():
        s_sym = bond.GetBeginAtom().GetAtomicNum()
        e_sym = bond.GetEndAtom().GetAtomicNum()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = utils_data.BOND_TYPES[bond.GetBondType()]
        distance = pdist[s_idx, e_idx]
        all_distances.append(((s_sym, e_sym, bond_type), distance))
    return all_distances


import torch
from tqdm import tqdm

if __name__ == '__main__':
    # TargetDiff_path = '/fs/ess/PCON0041/Ziqi/ShapeGeneration/baselines/pocket_baseline_results/targetdiff/targetdiff_vina_docked.pt'
    # TargetDiff_result = torch.load(TargetDiff_path)

    # results = {
    #     'bond_length': [],
    #     'all_results': [],
    # }

    # for i, pocket_result in tqdm(enumerate(TargetDiff_result)):
    #     for j, ligand_result in enumerate(pocket_result):
    #         bond_dist = bond_distance_from_mol(ligand_result['mol'])
    #         results['all_results'].append(TargetDiff_result[i][j])
    #         results['bond_length'] += bond_dist

    # TargetDiff_path

    # print(results.keys())
    # print(results['all_results'][0].keys())
    # print(len(results['all_results']))
    # print(len(results['bond_length']))

    # torch.save(results, '/fs/scratch/PCON0041/gruoxi/TargetDiff/results/eval_results/metrics_-1.pt')



    import os
    from rdkit import Chem
    from tqdm import tqdm

    split = torch.load('/fs/ess/PCON0041/gruoxi/split_by_name.pt')

    parent_dir = "/fs/scratch/PCON0041/gruoxi/crossdocked_pocket10/"

    all_bond_dist = []

    proteins = []
    ligands = []
    for complex in tqdm(split['train']):
        file_path = complex[1]
        rdmol = next(iter(Chem.SDMolSupplier(parent_dir+file_path, removeHs=True)))
        if rdmol != None:
            # smiles = Chem.MolToSmiles(rdmol)
            bond_dist = bond_distance_from_mol(rdmol)
            all_bond_dist += bond_dist
            # proteins.append(os.path.basename(file_path)[:10])
            # ligands.append(smiles)
    torch.save(all_bond_dist, '/fs/scratch/PCON0041/gruoxi/TargetDiff/results/train_bond_length.pt')
    print(all_bond_dist)



