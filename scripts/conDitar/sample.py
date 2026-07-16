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
import os
import argparse
import torch
from tqdm.auto import tqdm
import time
import utils.misc as misc
import utils.transforms as trans
from utils import misc, reconstruct
from datasets import get_dataset
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
import utils.data as utils_data
import numpy as np
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils import atom_num
from utils.device import resolve_device
from utils.scoring_func import *
from rdkit import Chem
from scripts.conDitar.sample_diffusion import sample_diffusion_ligand


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default=os.environ.get('CONDITAR_DEVICE', 'auto'))
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--batch_size', type=int, default=100)
    # Number of molecules to be generated
    parser.add_argument('--num_samples', type=int, default=100)
    # Radius
    parser.add_argument('--pocket_radius', type=int, default=10)
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--tmp_dir', type=str, default='../tmp')
    parser.add_argument('--protein_root', type=str, default='examples')
    # With reference ligand
    # parser.add_argument('--pdb_filename', type=str, default='4aua/4aua_protein.pdb')
    # parser.add_argument('--sdf_filename', type=str, default='4aua/4aua_ligand.sdf')
    # Without reference ligand
    parser.add_argument('--pdb_filename', type=str, default='xxxx/xxxx_pocket.pdb')
    parser.add_argument('--sdf_filename', type=str, default=None)


    args = parser.parse_args()
    # Resolve "auto" once and share it with helper code that reads CONDITAR_DEVICE.
    args.device = resolve_device(args.device)
    os.environ['CONDITAR_DEVICE'] = args.device

    logger = misc.get_logger('sample')
    logger.info(f"Using device: {args.device}")
    
    # Load config
    config = misc.load_config(args.config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])

    print(ckpt['config'])

    # Transforms
    if 'transform' in ckpt['config'].data:
        ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    else:
        ligand_atom_mode = 'full'
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # mol_path = os.path.join(args.result_path, os.path.dirname(args.pdb_filename))
    mol_path = args.result_path
    os.makedirs(mol_path, exist_ok=True)

    pdb_path = os.path.join(args.protein_root, args.pdb_filename)
    if args.sdf_filename != None:
        sdf_path = os.path.join(args.protein_root, args.sdf_filename)
    else:
        sdf_path = None
    test_set = get_dataset(config=config, name='test', files=[(pdb_path, sdf_path)], radius=args.pocket_radius, ligand_transform=transform)

    print(pdb_path)
    print(sdf_path)

    model = ScorePosNet3D(
        ckpt['config'],
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES),
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)

    data = test_set[0]

    print(data)

    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, \
        pred_pos_cond_traj, pred_v_cond_traj = sample_diffusion_ligand(
        model, data, args.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample_num_atoms
    )

    r = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
        'pred_ligand_v_cond_traj': pred_v_cond_traj,
    }

    all_pred_ligand_pos = r['pred_ligand_pos_traj']  
    all_pred_ligand_v = r['pred_ligand_v_traj']
        
    center = r['data']['ligand'].ligand_protein_pos_center.numpy() 

    for sample_idx, (pred_pos_all, pred_v_all) in enumerate(tqdm((zip(all_pred_ligand_pos, all_pred_ligand_v)))):
        pred_pos, pred_v = pred_pos_all[-1], pred_v_all[-1]   
        pred_pos = pred_pos + center
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)

        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            logger.warning('Reconstruct failed %s' % f'{sample_idx}')
            continue
        
        if '.' in smiles:
            continue

        print(smiles)

        writer = Chem.SDWriter(os.path.join(mol_path, f"{os.path.basename(args.pdb_filename)}_generated_{sample_idx}.sdf"))
        mol.SetProp("SMILES", smiles)
        writer.write(mol)
        writer.close()


    


        
