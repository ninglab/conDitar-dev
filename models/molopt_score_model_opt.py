# =============================================================================
# Copyright (c) 2026 Regents of the University of Minnesota.
# All rights reserved.
#
# paOPT is copyrighted by Regents of the University of Minnesota and covered by US 64/072,275. Regents of the University of Minnesota will license the use of paOPT solely for educational and research purposes by non-profit institutions and US government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. 
#
# You may not use the software to train or process or input the software into or make it accessible to: automated software, services or tools, including, but not limited to, artificial intelligence solutions, algorithms, machine learning, large language models, robots, spiders, crawlers, search engines, text or data mining or any other aggregation functionality. 
#
# One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice. (from https://research.umn.edu/units/techcomm/university-inventors/releasing-software)
# =============================================================================

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from models.molopt_score_model import ScorePosNet3D, center_pos, extract, index_to_log_onehot, log_sample_categorical


class ScorePosNet3D_opt(ScorePosNet3D):
    def __init__(self, config, ligand_atom_feature_dim, ligand_bond_feature_dim):
        super(ScorePosNet3D_opt, self).__init__(config, ligand_atom_feature_dim, ligand_bond_feature_dim)

    @torch.no_grad()
    def sample_diffusion(self, init_ligand_pos, init_ligand_v, batch_ligand, pocket_data, ligand_shape=None,
                         mask_shape_emb=None, num_steps=None, center_pos_mode=None, use_mesh_data=None, use_pointcloud_data=None,
                         use_pocket_data=None, grad_step=500, pred_bond=False, gaussian_noise_traj=None, gumbel_noise_traj=None):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_ligand.max().item() + 1
        print('sample center pos mode: ', center_pos_mode)

        init_ligand_pos, offset = center_pos(init_ligand_pos, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        pos_cond_traj, v_cond_traj = [], []
        pos_uncond_traj, v_uncond_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        ligand_pos, ligand_v = init_ligand_pos, init_ligand_v
        pred_bonds = None
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        idx = 0

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq), disable=True):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=ligand_pos.device)
            with torch.no_grad():
                preds_with_cond = self(
                    ligand_pos_perturbed=ligand_pos,
                    ligand_v_perturbed=ligand_v,
                    batch_ligand=batch_ligand,
                    pocket_data=pocket_data,
                    ligand_shape=ligand_shape,
                    mask_shape_emb=mask_shape_emb,
                    if_test=True,
                    pred_bond=True if self.pred_bond_type and pred_bond and i == 0 else False,
                    time_step=t
                )

            if self.pred_bond_type and pred_bond and i == 0:
                pred_bonds = torch.argmax(preds_with_cond['bond_pred'], dim=-1).view(-1, 1)
                pred_edge_indices = preds_with_cond['edge_index'].transpose(0, 1)
                pred_bonds = torch.cat([pred_edge_indices, pred_bonds], dim=1)

            preds = preds_with_cond

            pos_cond_traj.append(preds_with_cond['pred_ligand_pos'])
            v_cond_traj.append(preds_with_cond['pred_ligand_v'])

            if self.v_mode == 'tomask':
                preds['pred_ligand_v'][:, -1] = -1.e5

            with torch.no_grad():
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']

                pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
                pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
                nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
                ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * gaussian_noise_traj[idx]

                ligand_pos = ligand_pos_next

                log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
                log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes+int(self.v_mode=='tomask'))
                log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
                ligand_v_next = log_sample_categorical(log_model_prob, gumbel_noise=gumbel_noise_traj[idx])
                idx += 1

                v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
                vt_pred_traj.append(log_model_prob.clone().cpu())
                ligand_v = ligand_v_next

                if center_pos_mode != 'none':
                    ori_ligand_pos = ligand_pos + offset[batch_ligand]
                else:
                    ori_ligand_pos = ligand_pos

            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        if center_pos_mode != 'none':
            ligand_pos = ligand_pos + offset[batch_ligand]

        if pred_bond:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
        else:
            return {
                'pos': ligand_pos,
                'v': ligand_v,
                'bond': pred_bonds,
                'pos_traj': pos_traj,
                'pos_cond_traj': pos_cond_traj,
                'pos_uncond_traj': pos_uncond_traj,
                'v_traj': v_traj,
                'v_cond_traj': v_cond_traj,
                'v_uncond_traj': v_uncond_traj,
                'v0_traj': v0_pred_traj,
                'vt_traj': vt_pred_traj
            }
