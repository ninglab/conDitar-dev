# import qm9.dataset as dataset
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp_stats
from utils.transforms import MAP_INDEX_TO_ATOM_TYPE_AROMATIC, MAP_INDEX_TO_ATOM_TYPE_ONLY

# atom_decoder = ['H', 'C', 'N', 'O', 'F']

atom_encoder = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
atom_decoder = {v: k for k, v in atom_encoder.items()}


#Bond lengths from http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
#and from https://cccbdb.nist.gov/diatomicexpbondx.asp
bonds1 = {'H':  {'H': 74,  'C': 109, 'N': 101, 'O': 96,  'F': 92,  'P': 144, 'S': 134, 'Cl': 127, 'Br': 141, 'I': 161},
          'C':  {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194, 'I': 214},
          'N':  {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136, 'P': 177, 'S': 168, 'Cl': 175, 'Br': -1, 'I': -1},
          'O':  {'H': 96,  'C': 143, 'N': 140, 'O': 148, 'F': 142, 'P': 163, 'S': 151, 'Cl': 164, 'Br': -1, 'I': -1},
          'F':  {'H': 92,  'C': 135, 'N': 136, 'O': 142, 'F': 142, 'P': 156, 'S': 158, 'Cl': 166, 'Br': -1, 'I': 191},
          'P':  {'H': 144, 'C': 184, 'N': 177, 'O': 163, 'F': 156, 'P': 221, 'S': 210, 'Cl': 203, 'Br': -1, 'I': -1},
          'S':  {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'F': 158, 'P': 210, 'S': 204, 'Cl': 207, 'Br': -1, 'I': -1},
          'Cl': {'H': 127, 'C': 177, 'N': 175, 'O': 164, 'F': 166, 'P': 203, 'S': 207, 'Cl': 199, 'Br': -1, 'I': 232},
          'Br': {'H': 141, 'C': 194, 'N': -1,  'O': -1,  'F': -1,  'P': -1,  'S': -1,  'Cl': -1, 'Br': 228, 'I': -1},
          'I':  {'H': 161, 'C': 214, 'N': -1,  'O': -1,  'F': 191, 'P': -1,  'S': -1,  'Cl': 232, 'Br': -1, 'I': 267},
          }

bonds2 = {'H':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'C':  {'H': -1, 'C': 134, 'N': 129, 'O': 120, 'F': -1, 'P': -1,  'S': 160, 'Cl': -1, 'Br': -1, 'I': -1},
          'N':  {'H': -1, 'C': 129, 'N': 125, 'O': 121, 'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'O':  {'H': -1, 'C': 120, 'N': 121, 'O': 121, 'F': -1, 'P': 150, 'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'F':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'P':  {'H': -1, 'C': -1,  'N': -1,  'O': 150, 'F': -1, 'P': -1,  'S': 186, 'Cl': -1, 'Br': -1, 'I': -1},
          'S':  {'H': -1, 'C': 160, 'N': -1,  'O': -1,  'F': -1, 'P': 186, 'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'Cl': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'Br': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          'I':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1,  'S': -1,  'Cl': -1, 'Br': -1, 'I': -1},
          }

bonds3 = {'H':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'C':  {'H': -1, 'C': 120, 'N': 116, 'O': 113, 'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'N':  {'H': -1, 'C': 116, 'N': 110, 'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'O':  {'H': -1, 'C': 113, 'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'F':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'P':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'S':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'Cl': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'Br': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'I':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
        }

# calculated
bonds4 = {'H':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'C':  {'H': -1, 'C': 139, 'N': 134, 'O': 135, 'F': -1, 'P': -1, 'S': 174, 'Cl': -1, 'Br': -1, 'I': -1},
          'N':  {'H': -1, 'C': 134, 'N': 133, 'O': 135,  'F': -1, 'P': -1, 'S': 162, 'Cl': -1, 'Br': -1, 'I': -1},
          'O':  {'H': -1, 'C': 135, 'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'F':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'P':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'S':  {'H': -1, 'C': 174,  'N': 162,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'Cl': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'Br': {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
          'I':  {'H': -1, 'C': -1,  'N': -1,  'O': -1,  'F': -1, 'P': -1, 'S': -1, 'Cl': -1, 'Br': -1, 'I': -1},
        }
stdv = {'H': 5, 'C': 1, 'N': 1, 'O': 2, 'F': 3}
margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 4, 'Cl': 1, 'Br': 1, 'I': 1}

def construct_bond_tensors(atom_type, device='cuda'):
    if atom_type == 'add_aromatic':
        dicts = MAP_INDEX_TO_ATOM_TYPE_AROMATIC
    elif atom_type == 'basic':
        dicts = MAP_INDEX_TO_ATOM_TYPE_ONLY
    
    type_size = len(dicts)
    single_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    double_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    triple_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    aromatic_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    
    atom_type_dicts = {idx: (atom_decoder[dicts[idx][0]], dicts[idx][1]) for idx in dicts}
    for i in range(type_size):
        for j in range(i, type_size):
            atom_i, aroma_i = atom_type_dicts[i]
            atom_j, aroma_j = atom_type_dicts[j]

            if aroma_i and aroma_j:
                single_bond_tensor[i, j] = bonds1[atom_i][atom_j]
                single_bond_tensor[j, i] = bonds1[atom_i][atom_j]
                double_bond_tensor[i, j] = -1
                double_bond_tensor[j, i] = -1
                triple_bond_tensor[i, j] = -1
                triple_bond_tensor[i, j] = -1
                aromatic_bond_tensor[i, j] = bonds4[atom_i][atom_j]
                aromatic_bond_tensor[j, i] = bonds4[atom_i][atom_j]
            else:
                single_bond_tensor[i, j] = bonds1[atom_i][atom_j]
                single_bond_tensor[j, i] = bonds1[atom_i][atom_j]
                double_bond_tensor[i, j] = bonds2[atom_i][atom_j]
                double_bond_tensor[j, i] = bonds2[atom_i][atom_j]
                triple_bond_tensor[i, j] = bonds3[atom_i][atom_j]
                triple_bond_tensor[i, j] = bonds3[atom_i][atom_j]
                aromatic_bond_tensor[i, j] = -1
                aromatic_bond_tensor[j, i] = -1
    return single_bond_tensor, double_bond_tensor, triple_bond_tensor, aromatic_bond_tensor
    
def construct_bond_tensors(atom_type, device='cuda'):
    if atom_type == 'add_aromatic':
        dicts = MAP_INDEX_TO_ATOM_TYPE_AROMATIC
    elif atom_type == 'basic':
        dicts = MAP_INDEX_TO_ATOM_TYPE_ONLY
    
    type_size = len(dicts)
    single_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    double_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    triple_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    aromatic_bond_tensor = torch.ones((type_size, type_size)).to(device) * -1
    
    atom_type_dicts = {idx: (atom_decoder[dicts[idx][0]], dicts[idx][1]) for idx in dicts}
    for i in range(type_size):
        for j in range(i, type_size):
            atom_i, aroma_i = atom_type_dicts[i]
            atom_j, aroma_j = atom_type_dicts[j]
            
            if atom_i not in bonds1 or atom_j not in bonds1:
                single_bond_tensor[i, j] = -1
                single_bond_tensor[i, j] = -1
                double_bond_tensor[i, j] = -1
                double_bond_tensor[i, j] = -1
                triple_bond_tensor[i, j] = -1
                triple_bond_tensor[i, j] = -1
                aromatic_bond_tensor[i, j] = -1
                aromatic_bond_tensor[i, j] = -1
            elif aroma_i and aroma_j:
                single_bond_tensor[i, j] = bonds1[atom_i][atom_j]
                single_bond_tensor[j, i] = bonds1[atom_i][atom_j]
                double_bond_tensor[i, j] = -1
                double_bond_tensor[j, i] = -1
                triple_bond_tensor[i, j] = -1
                triple_bond_tensor[i, j] = -1
                aromatic_bond_tensor[i, j] = bonds4[atom_i][atom_j]
                aromatic_bond_tensor[j, i] = bonds4[atom_i][atom_j]
            else:
                single_bond_tensor[i, j] = bonds1[atom_i][atom_j]
                single_bond_tensor[j, i] = bonds1[atom_i][atom_j]
                double_bond_tensor[i, j] = bonds2[atom_i][atom_j]
                double_bond_tensor[j, i] = bonds2[atom_i][atom_j]
                triple_bond_tensor[i, j] = bonds3[atom_i][atom_j]
                triple_bond_tensor[i, j] = bonds3[atom_i][atom_j]
                aromatic_bond_tensor[i, j] = -1
                aromatic_bond_tensor[j, i] = -1
    return single_bond_tensor, double_bond_tensor, triple_bond_tensor, aromatic_bond_tensor


def normalize_histogram(hist):
    hist = np.array(hist)
    prob = hist / np.sum(hist)
    return prob


def coord2distances(x):
    x = x.unsqueeze(2)
    x_t = x.transpose(1, 2)
    dist = (x - x_t) ** 2
    dist = torch.sqrt(torch.sum(dist, 3))
    dist = dist.flatten()
    return dist


def earth_mover_distance(h1, h2):
    p1 = normalize_histogram(h1)
    p2 = normalize_histogram(h2)

    distance = sp_stats.wasserstein_distance(p1, p2)
    return distance


def kl_divergence(p1, p2):
    return np.sum(p1*np.log(p1 / p2))


def kl_divergence_sym(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    kl = kl_divergence(p1, p2)
    kl_flipped = kl_divergence(p2, p1)

    return (kl + kl_flipped) / 2.


def js_divergence(h1, h2):
    p1 = normalize_histogram(h1) + 1e-10
    p2 = normalize_histogram(h2) + 1e-10

    M = (p1 + p2)/2
    js = (kl_divergence(p1, M) + kl_divergence(p2, M)) / 2
    return js


def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric
    
    # margin1, margin2 and margin3 have been tuned to maximize the stability of the QM9 true samples
    if distance < bonds1[atom1][atom2] + margin1:
        thr_bond2 = bonds2[atom1][atom2] + margin2
        if distance < thr_bond2:
            thr_bond3 = bonds3[atom1][atom2] + margin3
            if distance < thr_bond3:
                return 3
            return 2
        return 1
    return 0


def check_stability(positions, atom_type, debug=False, hs=False, return_nr_bonds=False):
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    nr_bonds = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = atom_decoder[atom_type[i]], atom_decoder[atom_type[j]]
            if atom1 not in bonds1 or atom2 not in bonds1: continue
            order = get_bond_order(atom1, atom2, dist)
            # if i == 0:
            #     print(j, order)
            nr_bonds[i] += order
            nr_bonds[j] += order

    nr_stable_bonds = 0
    for atom_type_i, nr_bonds_i in zip(atom_type, nr_bonds):
        atom_i = atom_decoder[atom_type_i]
        if atom_i not in allowed_bonds: continue
        
        if hs:
            is_stable = allowed_bonds[atom_decoder[atom_type_i]] == nr_bonds_i
        else:
            is_stable = (allowed_bonds[atom_decoder[atom_type_i]] >= nr_bonds_i > 0)
        if is_stable == False and debug:
            print("Invalid bonds for molecule %s with %d bonds" % (atom_decoder[atom_type_i], nr_bonds_i))
        nr_stable_bonds += int(is_stable)

    molecule_stable = nr_stable_bonds == len(x)
    if return_nr_bonds:
        return molecule_stable, nr_stable_bonds, len(x), nr_bonds
    else:
        return molecule_stable, nr_stable_bonds, len(x)


def analyze_stability_for_molecules(molecule_list):
    n_samples = len(molecule_list)
    molecule_stable_list = []

    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0

    for one_hot, x in molecule_list:
        atom_type = one_hot.argmax(2).squeeze(0).cpu().detach().numpy()
        x = x.squeeze(0).cpu().detach().numpy()

        validity_results = check_stability(x, atom_type)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

        if validity_results[0]:
            molecule_stable_list.append((x, atom_type))

    # Validity
    fraction_mol_stable = molecule_stable / float(n_samples)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
    }

    #print('Validity:', validity_dict)

    return validity_dict, molecule_stable_list


if __name__ == '__main__':
    matplotlib.use('macosx')

