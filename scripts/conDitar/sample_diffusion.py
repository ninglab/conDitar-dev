# =============================================================================
# conDitar is copyrighted by the Ohio State University and covered by US 64/023,113. 
#
# conDitar may be licensed solely for educational and research purposes by non-profit institutions and US government agencies only. 
#
# For other proposed uses, contact tlcip@osu.edu. 
#
# The software may not be sold or redistributed without prior approval. 
#
# You may not use the software to train or process or input the software into or make it accessible to: automated software, services or tools, including, but not limited to, artificial intelligence solutions, algorithms, machine learning, large language models, robots, spiders, crawlers, search engines, text or data mining or any other aggregation functionality.   
#
# One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. 
#
# These terms and conditions are subject to change at any time without prior notice.
# =============================================================================

import pdb
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
import utils.misc as misc
import utils.transforms as trans
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from torch_scatter import scatter_sum, scatter_mean
import numpy as np
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical, index_to_log_onehot
from utils import atom_num
from utils import reconstruct
from datasets.mol_data import FOLLOW_BATCH
from rdkit import Chem, DataStructs
    

def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='none', sample_func=None,
                            sample_num_atoms='prior', sample_num_atoms_average=False):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_pos_cond_traj, all_pred_v_cond_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    
    # current_i = data['ligand'].ligand_element.shape[0]
    
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        pocket_batch = Batch.from_data_list([data['pocket'].clone() for _ in range(n_data)], follow_batch=list(FOLLOW_BATCH)).to(device)
        ligand_batch = Batch.from_data_list([data['ligand'].clone() for _ in range(n_data)], follow_batch=list(FOLLOW_BATCH) + ['bound']).to(device)
        t1 = time.time()
        with torch.no_grad():
            if sample_num_atoms == 'size':
                assert sample_func is not None
                ligand_num_atoms = sample_func(n_data)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = ligand_batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            elif sample_num_atoms == 'pocket':
                if hasattr(ligand_batch, "ligand_element_batch"):
                    batch_ligand = ligand_batch.ligand_element_batch
                    pocket_size_lp = atom_num.get_interaction_size(data['pocket'].pocket_coordinate.detach().cpu().numpy(), data['ligand'].ligand_coordinate.detach().cpu().numpy())
                    pocket_size_p = atom_num.get_space_size(data['pocket'].pocket_coordinate.detach().cpu().numpy())
                    if sample_num_atoms_average:
                        pocket_size = (pocket_size_lp + pocket_size_p) / 2
                    else:
                        pocket_size = pocket_size_lp
                else:
                    pocket_size = atom_num.get_space_size(data['pocket'].pocket_coordinate.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                print(ligand_num_atoms)
            else:
                raise ValueError
            
            # init ligand pos
            all_ligand_atoms = sum(ligand_num_atoms)
            
            init_ligand_pos = torch.randn(all_ligand_atoms, 3).to(device)
            
            # init ligand v
            if pos_only:
                # init_ligand_v = F.one_hot(batch.ligand_atom_feature_full, num_classes=model.num_classes).float()
                init_ligand_v = ligand_batch.ligand_atom_feature_full
            else:
                if model.v_mode == 'gaussian':
                    init_ligand_v = torch.randn(len(batch_ligand), model.num_classes).to(device)
                else:
                    uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                    init_ligand_v = log_sample_categorical(uniform_logits)
            
            r = model.sample_diffusion(
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                pocket_data=pocket_batch,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            ligand_pos_cond_traj, ligand_pos_uncond_traj = r['pos_cond_traj'], r['pos_uncond_traj']
            ligand_v_cond_traj, ligand_v_uncond_traj = r['v_cond_traj'], r['v_uncond_traj']

            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            try:
                all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]
            except:
                pdb.set_trace()

        ligand_bond = r['bond']
        if ligand_bond is not None:
            ligand_bond_array = ligand_bond.cpu().numpy().astype(int)
            ligand_bond_idx = np.argsort(ligand_bond_array[:, 0])
            ligand_bond_array = ligand_bond_array[ligand_bond_idx, :]

            n_idx, st_b_idx = 0, 0
            for b_idx in range(len(ligand_bond_array)):
                idx1, _, _ = ligand_bond_array[b_idx]
                if idx1 >= ligand_cum_atoms[n_idx+1]:
                    single_bond_indices = ligand_bond_array[st_b_idx:b_idx, :2] - ligand_cum_atoms[n_idx]
                    single_bond = np.concatenate([single_bond_indices, ligand_bond_array[st_b_idx:b_idx, -1].reshape(-1, 1)], axis=1)
                    
                    all_pred_bond.append(single_bond) #ligand_bond_array[st_b_idx:b_idx, :])
                    st_b_idx = b_idx
                    n_idx += 1
        
        all_step_pos = [[] for _ in range(n_data)]
        all_step_cond_pos = [[] for _ in range(n_data)]
        #all_step_uncond_pos = [[] for _ in range(n_data)]

        for pos, cond_pos in zip(ligand_pos_traj, ligand_pos_cond_traj):  # step_i
            p_array = pos.detach().cpu().numpy().astype(np.float64)
            cond_p_array = cond_pos.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
                all_step_cond_pos[k].append(cond_p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        all_step_cond_pos = [np.stack(step_pos) for step_pos in all_step_cond_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        
        all_pred_pos_traj += [p for p in all_step_pos]
        all_pred_pos_cond_traj += [p for p in all_step_cond_pos]

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]

        all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
        all_pred_v_traj += [v for v in all_step_v]

        all_step_cond_v = unbatch_v_traj(ligand_v_cond_traj, n_data, ligand_cum_atoms)
        all_pred_v_cond_traj += [v for v in all_step_cond_v]

        if not pos_only:
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
            all_pred_v0_traj += [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
            all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        # current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, \
        all_pred_pos_cond_traj, all_pred_v_cond_traj
