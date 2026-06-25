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

import argparse
import os
import re
from pathlib import Path
from utils import misc, reconstruct, transforms
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from utils import scoring_func
from utils.docking_qvina import QVinaDockingTask
from utils.docking_vina import VinaDockingTask
from multiprocessing import Pool


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

def is_vina_compatible(mol):
    allowed = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed:
            return False
    return True



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_path', type=str, default='test')
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_id', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--tmp_dir', type=str, default='../tmp')
    parser.add_argument('--protein_root', type=str, default='test_data')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none', 'all'], default='vina_score')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    current_path = args.sample_path
    all_sdf_files = []

    pattern = os.path.join(args.sample_path, "*", "*.sdf")
    all_sdf = glob(pattern, recursive=True)
    all_sdf_files = sorted([f for f in all_sdf])

    logger.info(f"Found {len(all_sdf_files)} .sdf files.")
    
    results = []

    for example_idx, r_name in enumerate(tqdm(all_sdf_files, desc='Eval')):
        folder = os.path.basename(os.path.dirname(r_name))   
        file_base = os.path.basename(r_name).split("_generated_")[0]
        if ".pdb" in file_base:
            protein_filename = os.path.join(folder, file_base)
            ligand_filename = os.path.join(folder, file_base.split(".pdb")[0] + ".sdf")
        elif ".sdf" in file_base:
            ligand_filename = os.path.join(folder, file_base)
            protein_filename = os.path.join(folder, file_base.split(".sdf")[0] + ".pdb")

        logger.info(str(os.path.basename(r_name)))
 
        mols = Chem.SDMolSupplier(r_name)

        for mol in mols:
            try:
                smiles = Chem.MolToSmiles(mol)
            except:
                continue

            if not is_vina_compatible(mol):
                continue
            
            if '.' in smiles:
                if args.verbose:
                    logger.warning('Molecule with seperate fragments')
                continue
        
            # try:
            chem_results = scoring_func.get_chem(mol)
        
            if args.docking_mode == 'qvina':
                qvina_task = QVinaDockingTask.from_generated_mol(
                    mol, ligand_filename, protein_filename=protein_filename, protein_root=args.protein_root, tmp_dir=args.tmp_dir)
                qvina_results = qvina_task.run_sync(exhaustiveness=args.exhaustiveness)
                q_vina_results.append(qvina_results[0]['affinity'])
                vina_results = {
                    'qvina': qvina_results
                }
    
            elif args.docking_mode in ['vina_score', 'vina_dock']:
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, ligand_filename, protein_filename=protein_filename, protein_root=args.protein_root, test=True, tmp_dir=args.tmp_dir)
                
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=10)
                vina_results = {
                    'score_only': score_only_results,
                }
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, cpu=10)
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                if args.docking_mode == 'vina_dock':
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results['dock'] = docking_results
            elif args.docking_mode == 'all':
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, ligand_filename, protein_filename=protein_filename, protein_root=args.protein_root, test=True, tmp_dir=args.tmp_dir)
                
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness, cpu=10)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness, cpu=10)

                qvina_task = QVinaDockingTask.from_generated_mol(
                    mol, ligand_filename, protein_filename=protein_filename, protein_root=args.protein_root,  tmp_dir=args.tmp_dir)
                qvina_results = qvina_task.run_sync(exhaustiveness=args.exhaustiveness)
                q_vina_results.append(qvina_results[0]['affinity'])

                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results,
                    'qvina': qvina_results
                }

            else:
                vina_results = None

            logger.info(vina_results)
            logger.info(chem_results)

            # except Exception as e:
            #     logger.info(e)
            #     if args.verbose:
            #         logger.warning('Evaluation failed')
            #     continue

            results.append({
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': str(os.path.basename(r_name)),
                'chem_results': chem_results,
                'vina': vina_results,
            })


    logger.info(f'Evaluate done!')


    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))

    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        # vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        # logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    result_path = os.path.join(args.sample_path, 'eval_results')
    pt_path = os.path.join(result_path, f'metrics_-1.pt')
    torch.save({
        'all_results': results
    }, pt_path)




    

