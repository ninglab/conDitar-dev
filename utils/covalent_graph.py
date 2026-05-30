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
import openbabel.openbabel as ob

from .transforms import get_atomic_number_from_index

def connect_covalent_graph(ligand_pos, ligand_v, atom_mode='add_aromatic', gamma=0.2):
    atomic_index = torch.where(ligand_v > 0)[1]
    atomic_nums = get_atomic_number_from_index(atomic_index, atom_mode)
    covalent_radius = torch.FloatTensor([ob.GetCovalentRad(atomic_num) for atomic_num in atomic_nums]).unsqueeze(0).to(ligand_pos)
    
    pair_dists = torch.cdist(ligand_pos, ligand_pos, p=2)

    covalent_dists = covalent_radius + covalent_radius.transpose(1, 0) + gamma
    
    edge_mask = (pair_dists < covalent_dists) & (~torch.eye(len(atomic_nums)).to(ligand_pos).bool())
    
    edges = torch.vstack(torch.where(edge_mask))
    return edges
