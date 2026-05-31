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

import pdb
import torch
from utils.transforms import *
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

FOLLOW_BATCH = ('protein_element', 'ligand_element', 'ligand_bond_type', 'id')



def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
            if output[k].dtype not in [torch.int8, torch.int16, torch.int32, torch.int64]:
                output[k] = output[k].to(torch.float32)
        elif isinstance(v, float):
            output[k] = torch.tensor(v, dtype=torch.float32)
        else:
            output[k] = v
    return output


def parse_protein_pocket(pocket_data, hydrogen=False):
    pocket_dict = {}
    positions = pocket_data['protein_pos'] - pocket_data['protein_pos'].mean(0)
    atom_types = [ATOM_INDEX_DICT[atom[0]] for atom in pocket_data['protein_atom_name']]

    
    if hydrogen == False:
        positions_selected = np.array(list(map(lambda x: x == 0, atom_types))) == False
        atom_types = np.array(atom_types) - 1
    else:
        positions_selected = np.full(len(atom_types), True, dtype=bool)

    pocket_dict['coordinate'] = np.array(positions[positions_selected])
    pocket_dict['element'] = np.array(atom_types)[positions_selected]

    def identify_residue_boundaries(atom_list):
        """ Identify the starting index of each residue based on the 'N', 'CA', 'C', 'O' pattern """
        boundaries = []
        n = len(atom_list)
        
        for i in range(n - 3):
            if (atom_list[i] == 'N' and atom_list[i + 1] == 'CA' and 
                atom_list[i + 2] == 'C' and atom_list[i + 3] == 'O'):
                boundaries.append(i)
        
        if boundaries:
            boundaries.append(n) 
        
        return boundaries

    
    def get_residue_indices(atom_list):
        boundaries = identify_residue_boundaries(atom_list)
        residue_indices = []
        i = 0
        for start, end in zip(boundaries, boundaries[1:] + [len(atom_list)]):
            residue_indices.extend([i] * (end - start))
            i = i + 1
        return residue_indices
    

    residue_indices = get_residue_indices(pocket_data['protein_atom_name'])
    pocket_dict['residue_type'] = np.array(pocket_data['protein_atom_to_aa_type'])[positions_selected]
    pocket_dict['residue_index'] = np.array(residue_indices)[positions_selected]
    pocket_dict['is_backbone'] = pocket_data['protein_is_backbone'][positions_selected]
    pocket_dict['atom_name'] = [s for s, flag in zip(pocket_data['protein_atom_name'], positions_selected) if flag]

    try:
        pocket_dict['embd_scalar'] = np.array(pocket_data['protein_embd_scaler'])
        pocket_dict['embd_vector'] = np.array(pocket_data['protein_embd_vector'])
    except:
        return pocket_dict

    
    return pocket_dict

def parse_ligand_pocket(data, hydrogen=False):
    ligand_dict = {}
    try:
        if 'protein_pos' in data:
            if 'ligand_center' in data:
                positions = data['ligand_pos'] + data['ligand_center'] - data['protein_pos'].mean(0)
            else:
                positions = data['ligand_pos'] - data['protein_pos'].mean(0)
        else:
            positions = data['ligand_pos']
    except Exception as e:
        print(e)
        pdb.set_trace()
    atom_types = data['ligand_element']

    if hydrogen == False:
        positions_selected = np.array(list(map(lambda x: x == 0, atom_types.tolist()))) == False
    else:
        positions_selected = np.full(len(atom_types.tolist()), True, dtype=bool)

    def update_bond(positions_selected, bond_indices, bond_types):
        new_indices = np.cumsum(positions_selected) - 1 

        kept_bonds_mask = np.all(positions_selected[bond_indices], axis=0)

        filtered_bond_indices = bond_indices[:, kept_bonds_mask]
        filtered_bond_types = bond_types[kept_bonds_mask]

        updated_bond_indices = new_indices[filtered_bond_indices]

        return {'bond_index': updated_bond_indices, 'bond_type': filtered_bond_types}
    
    for key, item in data.items():
        if key.startswith("ligand"):
            name = key.split("ligand_")[1]
            if name == 'nbh_list':
                continue
            if name == 'pos':
                ligand_dict['coordinate'] = np.array(positions)[positions_selected]
            else:
                if name != 'smiles' and 'bond' not in name and 'center' not in name and name != 'shape_emb':
                    if isinstance(item, list):
                        ligand_dict[name] = np.array(item)[positions_selected].tolist()
                    elif isinstance(item, torch.Tensor) and len(item.shape) > 0:
                        try:
                            ligand_dict[name] = item[torch.tensor(positions_selected, dtype=torch.bool)]
                        except:
                            pdb.set_trace()
                    else:
                        ligand_dict[name] = item
                else:
                    ligand_dict[name] = item
    
    ligand_dict['element'] = np.array(atom_types)[positions_selected]
    try:
        if 'protein_pos' in data: ligand_dict['protein_pos_center'] = data['protein_pos'].mean(0)
    except:
        pdb.set_trace()
    updated_bond = update_bond(positions_selected, ligand_dict['bond_index'], ligand_dict['bond_type'])
    ligand_dict.update(updated_bond)
    
    return ligand_dict


class PocketData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_pocket_dicts(pocket_dict, **kwargs):
        instance = PocketData(**kwargs)

        for key, item in pocket_dict.items():
            if key.startswith('pocket_'):
                instance[key] = item
            else:
                instance['pocket_' + key] = item
        
        instance['num_nodes'] = instance.num_nodes
        return instance
    
    @property
    def num_nodes(self):
        if hasattr(self, 'pocket_element'):
            return len(self.pocket_element)
        elif hasattr(self, 'pocket_embd_scaler'):
            return self.pocket_embd_scaler.size(0)
        else:
            return None
    
    def __inc__(self, key, value, data=None, store=None):
        if key == 'pocket_residue_index':
            return self.pocket_residue_index.max() + 1
        else:
            return super().__inc__(key, value)
        
class LigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_ligand_dicts(ligand_dict, **kwargs):
        instance = LigandData(**kwargs)

        for key, item in ligand_dict.items():
            if key.startswith('ligand_'):
                instance[key] = item
            else:
                instance['ligand_' + key] = item
        
        instance['num_nodes'] = instance.num_nodes
        return instance
    
    @property
    def num_nodes(self):
        if hasattr(self, 'ligand_coordinate'):
            try:
                return self.ligand_coordinate.shape[0]
            except:
                return self.ligand_coordinate.size(0)
        else:
            return None

    @property
    def num_edges(self):
        if hasattr(self, 'ligand_bond_index'):
            try:
                return self.ligand_bond_index.shape[1]
            except:
                return self.ligand_bond_index.size(1)
        else:
            return None


class ComplexData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_ligand_dicts(ligand_dict=None, protein_dict=None, **kwargs):
        instance = ComplexData(**kwargs)

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item
                
        if ligand_dict is not None:
            instance['ligand_nbh_list'] = {i.item(): [j.item() for k, j in enumerate(instance.ligand_bond_index[1])
                                                  if instance.ligand_bond_index[0, k].item() == i]
                                       for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_index':
            return self['ligand_index']
        elif key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        # elif key == 'ligand_context_bond_index':
        #     return self['ligand_context_element'].size(0)
        else:
            return super().__inc__(key, value)
