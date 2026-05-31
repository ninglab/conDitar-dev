#################################################################
# Copyright (c) 2026 Regents of the University of Minnesota.
# All rights reserved.
#
# paOPT is source-available research software covered by
# U.S. Patent Application Serial No. 64/072,275.
#
# Use, copying, redistribution, and other activities are governed by the
# license terms in LICENSE.txt. This software is provided "as is" without
# warranty of any kind.
#################################################################

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

def sample_diffusion_ligand_opt(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='none', sample_func=None,
                            sample_num_atoms='prior',
                            init_ligand_pos=None, gaussian_noise_traj=None,
                            init_ligand_v=None, gumbel_noise_traj=None,
                            ligand_num_atoms=None,
                            ligand_cum_atoms=None,
                            batch_ligand=None,
                            sample_num_atoms_average=False):
    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_pos_cond_traj, all_pred_v_cond_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    
    current_i = data['ligand'].ligand_element.shape[0]
    
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        pocket_batch = Batch.from_data_list([data['pocket'].clone() for _ in range(n_data)], follow_batch=list(FOLLOW_BATCH)).to(device)
        ligand_batch = Batch.from_data_list([data['ligand'].clone() for _ in range(n_data)], follow_batch=list(FOLLOW_BATCH) + ['bound']).to(device)

        t1 = time.time()
        with torch.no_grad():
            if ligand_num_atoms is None or ligand_cum_atoms is None or batch_ligand is None:
                if sample_num_atoms == 'size':
                    assert sample_func is not None
                    ligand_num_atoms = sample_func(n_data)
                    batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                elif sample_num_atoms == 'ref':
                    batch_ligand = ligand_batch.ligand_element_batch
                    ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
                elif sample_num_atoms == 'pocket':
                    batch_ligand = ligand_batch.ligand_element_batch
                    pocket_size_lp = atom_num.get_interaction_size(data['pocket'].pocket_coordinate.detach().cpu().numpy(), data['ligand'].ligand_coordinate.detach().cpu().numpy())
                    pocket_size_p = atom_num.get_space_size(data['pocket'].pocket_coordinate.detach().cpu().numpy())
                    if sample_num_atoms_average:
                        pocket_size = (pocket_size_lp + pocket_size_p) / 2
                    else:
                        pocket_size = pocket_size_lp
                    ligand_num_atoms = [atom_num.sample_atom_num(pocket_size) for _ in range(n_data)]
                    batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
                else:
                    raise ValueError
            
            # init ligand pos
            all_ligand_atoms = sum(ligand_num_atoms)
            if init_ligand_pos is None:
                init_ligand_pos = torch.randn(all_ligand_atoms, 3).to(device)
            
            # init ligand v
            if init_ligand_v is None:
                if pos_only:
                    # init_ligand_v = F.one_hot(batch.ligand_atom_feature_full, num_classes=model.num_classes).float()
                    init_ligand_v = ligand_batch.ligand_atom_feature_full
                else:
                    if model.v_mode == 'gaussian':
                        init_ligand_v = torch.randn(len(batch_ligand), model.num_classes).to(device)
                    else:
                        uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                        init_ligand_v = log_sample_categorical(uniform_logits)
            
            if num_steps is None:
                num_steps = model.num_timesteps
            time_seq = list(reversed(range(model.num_timesteps - num_steps, model.num_timesteps)))       

            if gaussian_noise_traj is None:
                gaussian_noise_traj = [torch.randn_like(init_ligand_pos) for _ in time_seq]
            if gumbel_noise_traj is None:
                log_ligand_v = index_to_log_onehot(init_ligand_v, model.num_classes+int(model.v_mode=='tomask'))
                gumbel_noise_traj = [-torch.log(-torch.log(torch.rand_like(log_ligand_v)+1e-30)+1e-30) for _ in time_seq]

            r = model.sample_diffusion(
                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                pocket_data=pocket_batch,
                num_steps=num_steps,
                center_pos_mode=center_pos_mode,
                gaussian_noise_traj=gaussian_noise_traj,
                gumbel_noise_traj=gumbel_noise_traj
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            ligand_pos_cond_traj, ligand_pos_uncond_traj = r['pos_cond_traj'], r['pos_uncond_traj']
            ligand_v_cond_traj, ligand_v_uncond_traj = r['v_cond_traj'], r['v_uncond_traj']

            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            try:
                all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]] for k in range(n_data)]  # num_samples * [num_atoms_i, 3]
            except:
                print('Error in unbatching pos, current_i:', current_i, 'n_data:', n_data, 'ligand_cum_atoms:', ligand_cum_atoms)
                print('ligand_pos_array shape:', ligand_pos_array.shape)
                print('ligand_num_atoms:', ligand_num_atoms)
                raise ValueError

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
                    
                    all_pred_bond.append(single_bond) 
                    st_b_idx = b_idx
                    n_idx += 1
        
        all_step_pos = [[] for _ in range(n_data)]
        all_step_cond_pos = [[] for _ in range(n_data)]

        for pos, cond_pos in zip(ligand_pos_traj, ligand_pos_cond_traj):  
            p_array = pos.detach().cpu().numpy().astype(np.float64)
            cond_p_array = cond_pos.detach().cpu().numpy().astype(np.float64)
            for k in range(n_data):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])
                all_step_cond_pos[k].append(cond_p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k+1]])

        all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]  
        all_step_cond_pos = [np.stack(step_pos) for step_pos in all_step_cond_pos]  
        
        all_pred_pos_traj += [p for p in all_step_pos]
        all_pred_pos_cond_traj += [p for p in all_step_cond_pos]

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
        current_i += n_data

    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list, \
        all_pred_pos_cond_traj, all_pred_v_cond_traj, init_ligand_pos, gaussian_noise_traj, init_ligand_v, gumbel_noise_traj, \
        ligand_num_atoms, ligand_cum_atoms, batch_ligand


