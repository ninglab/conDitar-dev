import pdb
import os, sys
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
import torch.nn.functional as F
import utils.data as utils_data
import numpy as np
import pdb
import cvxpy as cp
from models.molopt_score_model import ScorePosNet3D, ScorePosNet3D_opt, log_sample_categorical, extract, index_to_log_onehot, center_pos
from rdkit import Chem
from utils import eval_atom_type, scoring_func, analyze, eval_bond_length
from scripts.sample_diffusion import *
import json
from admet_ai import ADMETModel
from contextlib import contextmanager


def get_admet_ai(smiles_path):
    with open(smiles_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    print('admet list length:', len(smiles_list))
    @contextmanager
    def suppress_output():
        """A context manager that redirects stdout and stderr to devnull."""
        with open(os.devnull, 'w') as f:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = f
            sys.stderr = f
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    with suppress_output():
        admet_ai_model = ADMETModel()
        preds = admet_ai_model.predict(smiles=smiles_list)
    print('ADMET predictions BBBP:', preds['BBB_Martins'].values)
    return preds

def get_descent_dir(grads):
    # print('Computing MGDA descent direction...')
    # Flatten all gradients, in a way that the original shape can be recovered
    if grads['valid'] == False:
        print('No valid gradients, return None')
        return None
    grad_list = []
    grad_list_shape = []
    # print('grads keys: ', grads.keys())
    for iter, obj in enumerate(grads.keys()):
        if obj == 'valid':
            continue
        grad_sublist = []
        for grad in grads[obj]:
            # print(f'grad shape: {grad.shape}')
            grad_sublist.extend(grad.view(-1).tolist())
            if iter == 2:
                grad_list_shape.append(grad.shape)

        # print(f'Objective {obj}: {len(grad_sublist)}')
        grad_list.append(torch.tensor(grad_sublist))

    G = torch.stack(grad_list, dim=1).detach()  # Shape: (num_params, num_objs)
    # print(f'G shape: {G.shape}')

    # Number of objectives
    num_obj = G.shape[1]
    # print(f'Number of objectives: {num_obj}')
    # Define optimization variables
    lambda_vars = cp.Variable(num_obj)  # Lambda values for each objective
    # Compute the weighted sum of gradients
    G_np = G.cpu().numpy().astype(np.float32)  # Convert to NumPy for cvxpy

    descent_dir = cp.matmul(G_np, lambda_vars) # Shape: (num_params,)
    # Define the QP problem: min ||sum lambda_i * g_i ||^2
    objective = cp.Minimize(cp.sum_squares(descent_dir))

    # Constraints: lambda_i >= 0 and sum(lambda_i) = 1
    constraints = [lambda_vars >= 0, cp.sum(lambda_vars) == 1]

    # Solve the QP problem
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except:
        print('Solver failed, return None')
        return None
    # Get optimal lambda values
    optimal_lambdas = lambda_vars.value
    if optimal_lambdas is None:
        print('Solver failed, return None')
        return None
    print(f'Optimal lambdas: {optimal_lambdas}')
    # Compute final descent direction
    final_descent_dir = G_np @ optimal_lambdas  # Shape: (num_params,)
    final_descent_dir = torch.tensor(final_descent_dir, dtype=torch.float32)

    # compute norm
    norm = torch.norm(final_descent_dir, p=2)
    print(f'Final descent direction norm: {norm}')
    if norm < 1e-6:
        print('Final descent direction norm is too small, return None')
        return None
    # Normalize the descent direction
    if norm > 0:
        final_descent_dir = final_descent_dir / torch.norm(final_descent_dir, dim=-1, keepdim=True)

    # print('MGDA descent direction computed!')

    # recover the original shape
    reshaped_descent_dir = []
    start = 0
    # print(grad_list_shape)
    for shape in grad_list_shape:
        # print(shape)
        end = start + np.prod(shape)
        # print(end)
        # print(torch.tensor(final_descent_dir[start:end]))
        # print(torch.tensor(final_descent_dir[start:end].view(shape)))
        reshaped_descent_dir.append(torch.tensor(final_descent_dir[start:end].reshape(shape)))
        start = end
    # print(f'Final descent direction shape: {reshaped_descent_dir[0].shape}')

    return reshaped_descent_dir


def sample_wrapper(model, data, num_samples, batch_size, device, num_steps, center_pos_mode, sample_num_atoms, init_ligand_pos, init_ligand_v, gaussian_noise_traj, gumbel_noise_traj, ligand_num_atoms, ligand_cum_atoms, batch_ligand, i, args):
    print(f'gradient estimation run {i} sample started')
        # Load checkpoint

    init_ligand_pos = init_ligand_pos.to(device)
    init_ligand_v = init_ligand_v.to(device)
    gaussian_noise_traj = [gaussian_noise_traj[i].to(device) for i in range(len(gaussian_noise_traj))]
    gumbel_noise_traj = [gumbel_noise_traj[i].to(device) for i in range(len(gumbel_noise_traj))]
    batch_ligand = batch_ligand.to(device)
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, pred_pos_cond_traj, pred_v_cond_traj, _, _, _, _, _, _, _ = sample_diffusion_ligand_opt(
                    model, data, num_samples,
                    batch_size=batch_size, device=device,
                    num_steps=num_steps,
                    center_pos_mode=center_pos_mode,
                    sample_num_atoms=sample_num_atoms,
                    init_ligand_pos=init_ligand_pos,
                    init_ligand_v=init_ligand_v,
                    gaussian_noise_traj=gaussian_noise_traj,
                    gumbel_noise_traj=gumbel_noise_traj,
                    ligand_num_atoms=ligand_num_atoms,
                    ligand_cum_atoms=ligand_cum_atoms,
                    batch_ligand=batch_ligand
                    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
        'pred_ligand_v_cond_traj': pred_v_cond_traj,
    }


    all_pred_ligand_pos = result['pred_ligand_pos_traj']
    all_pred_ligand_v = result['pred_ligand_v_traj']
    # evaluate
    center = result['data']['ligand'].ligand_protein_pos_center.numpy()
    qed = []
    sa = []
    logp = []
    lipinski = []
    smiles_results = []
    # bbbp = []
    reconstruct_idx = 0
    for sample_idx, (pred_pos_all, pred_v_all) in enumerate(tqdm((zip(all_pred_ligand_pos, all_pred_ligand_v)))):
        pred_pos, pred_v = pred_pos_all[-1], pred_v_all[-1]
        pred_pos = pred_pos + center
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except:
            print('Reconstruct failed %s' % f'{sample_idx}')
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            smiles_results.append(None)
            reconstruct_idx += 1
            continue
        if '.' in smiles:
            print('Invalid SMILES: %s' % f'{smiles}')
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            smiles_results.append(None)
            reconstruct_idx += 1
            continue
        smiles_results.append(smiles)
        print(f'run {i} SMILES: {smiles}')
        try:
            chem_results = scoring_func.get_chem(mol)
        except:
            print('Chem scoring failed %s' % f'{sample_idx}')
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            reconstruct_idx += 1
            continue
        qed.append(chem_results['qed'])
        sa.append(chem_results['sa'])
        logp.append(chem_results['logp'])
        lipinski.append(chem_results['lipinski'])
        reconstruct_idx += 1
    eval_result = {
        'smiles': smiles_results,
        'QED': qed,
        'SA': sa,
        'LogP': logp,
        'Lipinski': [int(lipinski) if lipinski is not None else None for lipinski in lipinski],
        'Run ID': i
    }
    # save to json
    result_path = f'{args.result_path}/{args.pdb_filename}_eval_results_{i}_{str(int(time.time()))}.json'

    with open(result_path, 'w') as f:
        json.dump(eval_result, f)
    return result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_estimates', type=int, default=4)
    parser.add_argument('--result_path', type=str, default='outputs')
    parser.add_argument('--tmp_dir', type=str, default='../tmp')
    parser.add_argument('--protein_root', type=str, default='test_data')
    # Protein Target
    parser.add_argument('--pdb_filename', type=str, default='xxx2/xxx2_protein.pdb')
    # Reference Ligand
    parser.add_argument('--sdf_filename', type=str, default='xxx2/xxx2_ligand.sdf')
    # Optimization steps
    parser.add_argument('--opt_steps', type=int, default=3)
    # Optimization learning rate
    parser.add_argument('--opt_lr', type=float, default=0.1)
    # perturbation size
    parser.add_argument('--per_size', type=float, default=0.03)
    # Random seed
    parser.add_argument('--seed', type=int, default=2025)
    # Optimization keys
    parser.add_argument('--opt_keys', nargs='+', type=str, default=['Carcinogenicity'])
    # Optimization keys min
    parser.add_argument('--opt_keys_min', nargs='+', type=str, default=['Carcinogenicity'])
    parser.add_argument('--init_input_path', type=str, default=None)
    parser.add_argument('--init_input_step', type=int, default=8)

    args = parser.parse_args()

    logger = misc.get_logger('sample_with_opt')

    pdbid = os.path.basename(args.pdb_filename)[:4]
    os.makedirs(os.path.join(args.result_path, pdbid), exist_ok=True)
    
    # Load config
    config = misc.load_config(args.config)
    misc.seed_all(args.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    if 'train_config' in config.model:
        logger.info(f"Load training config from: {config.model['train_config']}")
        ckpt['config'] = misc.load_config(config.model['train_config'])

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
    print(ckpt['config'])
    pdb_path = os.path.join(args.protein_root, args.pdb_filename)
    sdf_path = os.path.join(args.protein_root, args.sdf_filename)
    test_set = get_dataset(config=config, name='test', files=[(pdb_path, sdf_path)], ligand_transform=transform)
    
    model = ScorePosNet3D_opt(
        ckpt['config'],
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES),
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    # model.share_memory()
    data = test_set[0]

    print(data)
    result_path_list = []
    OPT_KEYS = args.opt_keys
    OPT_KEYS_MIN = args.opt_keys_min
    ##### initial sample #####
    print('Initial sample started')
    # worst_init_obj = 2
    # while worst_init_obj > 0.6:
    if args.init_input_path is None:
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, pred_pos_cond_traj, pred_v_cond_traj,init_ligand_pos, gaussian_noise_traj, init_ligand_v, gumbel_noise_traj, ligand_num_atoms, ligand_cum_atoms, batch_ligand = sample_diffusion_ligand_opt(
        model, data, args.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
        )
        # print variable types
        print('init_ligand_pos type: ', type(init_ligand_pos))
        print('init_ligand_v type: ', type(init_ligand_v))
        print('gaussian_noise_traj type: ', type(gaussian_noise_traj))
        print('init_ligand_pos: ', init_ligand_pos.shape)
        print('init_ligand_v: ', init_ligand_v.shape)
        print('ligand_num_atoms: ', ligand_num_atoms)
        print('ligand_cum_atoms: ', ligand_cum_atoms)
        print('batch_ligand: ', batch_ligand)

    else:
        print('Load initial input from: ', args.init_input_path)
        # load npy
        init_ligand_pos = np.load(f'{args.init_input_path}_init_ligand_pos_step_{args.init_input_step}.npy')
        # transform to torch
        init_ligand_pos = torch.tensor(init_ligand_pos, dtype=torch.float32).to(args.device)
        init_ligand_v = np.load(f'{args.init_input_path}_init_ligand_v_step_{args.init_input_step}.npy')
        init_ligand_v = torch.tensor(init_ligand_v, dtype=torch.long).to(args.device)
        gaussian_noise_traj_npy = np.load(f'{args.init_input_path}_gaussian_noise_traj_step_{args.init_input_step}.npy', allow_pickle=True)
        gaussian_noise_traj = [torch.tensor(gaussian_noise_traj_npy[i], dtype=torch.float32).to(args.device) for i in range(len(gaussian_noise_traj_npy))]
        gumbel_noise_traj_npy = np.load(f'{args.init_input_path}_gumbel_noise_traj_step_{args.init_input_step}.npy', allow_pickle=True)
        gumbel_noise_traj = [torch.tensor(gumbel_noise_traj_npy[i], dtype=torch.float32).to(args.device) for i in range(len(gumbel_noise_traj_npy))]
        ligand_num_atoms = np.load(f'{args.init_input_path}_ligand_num_atoms_step_{args.init_input_step}.npy')
        # transform to a list
        ligand_num_atoms = [int(x) for x in ligand_num_atoms]
        ligand_cum_atoms = np.load(f'{args.init_input_path}_ligand_cum_atoms_step_{args.init_input_step}.npy')
        ligand_cum_atoms = [int(x) for x in ligand_cum_atoms]
        batch_ligand = np.load(f'{args.init_input_path}_batch_ligand_step_{args.init_input_step}.npy')
        batch_ligand = torch.tensor(batch_ligand, dtype=torch.long).to(args.device)
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, pred_pos_cond_traj, pred_v_cond_traj, _, _, _, _, _, _, _ = sample_diffusion_ligand_opt(
            model, data, args.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            gaussian_noise_traj=gaussian_noise_traj,
            gumbel_noise_traj=gumbel_noise_traj,
            ligand_num_atoms=ligand_num_atoms,
            ligand_cum_atoms=ligand_cum_atoms,
            batch_ligand=batch_ligand
            )


    print('time: ', time_list)
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list,
        'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
        'pred_ligand_v_cond_traj': pred_v_cond_traj,
    }
    all_pred_ligand_pos = result['pred_ligand_pos_traj']
    all_pred_ligand_v = result['pred_ligand_v_traj']
    # evaluate

    center = result['data']['ligand'].ligand_protein_pos_center.numpy()
    qed = []
    sa = []
    logp = []
    lipinski = []
    smiles_results = []
    reconstruct_idx = 0
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
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            smiles_results.append(None)
            reconstruct_idx += 1
            continue
        if '.' in smiles:
            logger.warning(f'Invalid SMILES: {smiles} at sample {sample_idx}')
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            smiles_results.append(None)
            reconstruct_idx += 1
            continue
        print(smiles)

        try:
            chem_results = scoring_func.get_chem(mol)
            qed.append(chem_results['qed'])
            sa.append(chem_results['sa'])
            logp.append(chem_results['logp'])
            lipinski.append(chem_results['lipinski'])
        except:
            logger.warning('Chem scoring failed %s' % f'{sample_idx}')
            qed.append(None)
            sa.append(None)
            logp.append(None)
            lipinski.append(None)
            smiles_results.append(smiles)
            reconstruct_idx += 1
            continue
        smiles_results.append(smiles)
        writer = Chem.SDWriter(f"{args.result_path}/{args.sdf_filename}_generated_init_{sample_idx}.sdf")
        writer.write(mol)
        writer.close()
        reconstruct_idx += 1
    eval_result = {
        'smiles': smiles_results,
        'QED': qed,
        'SA': sa,
        'LogP': logp,
        'Lipinski': [int(lipinski) if lipinski is not None else None for lipinski in lipinski],
    }

    # save smiles results to txt
    invalid_idx = []
    for idx, smiles in enumerate(smiles_results):
        if smiles is None:
            invalid_idx.append(idx)
    with open(f'{args.result_path}/{args.pdb_filename}_smiles_results_init.txt', 'w') as f:
        for idx, smiles in enumerate(smiles_results):
            if smiles is None:
                invalid_idx.append(idx)
                continue
            f.write(smiles + '\n')
    print('invalid idx: ', invalid_idx)
    # collect the admetAI results
    admet = get_admet_ai(f'{args.result_path}/{args.pdb_filename}_smiles_results_init.txt')

    if admet is None:
        print('ADMET AI prediction failed')
        eval_result['BBBP'] = [None] * len(smiles_results)
        eval_result['HIA'] = [None] * len(smiles_results)
        eval_result['hERG'] = [None] * len(smiles_results)
        eval_result['Ames'] = [None] * len(smiles_results)
        eval_result['Carcinogenicity'] = [None] * len(smiles_results)
        eval_result['CaCo2'] = [None] * len(smiles_results)
    else:
        for i in range(len(smiles_results)):
            if smiles_results[i] is None:
                continue
            if not smiles_results[i] in admet.index.values:
                invalid_idx.append(i)
        eval_result['BBBP'] = [None] * len(smiles_results)
        eval_result['HIA'] = [None] * len(smiles_results)
        eval_result['hERG'] = [None] * len(smiles_results)
        eval_result['Ames'] = [None] * len(smiles_results)
        eval_result['Carcinogenicity'] = [None] * len(smiles_results)
        eval_result['CaCo2'] = [None] * len(smiles_results)
        # parse the admet results to valid index
        idx = 0
        for i in range(len(smiles_results)):               
            if i in invalid_idx:
                continue
            eval_result['BBBP'][i] = admet['BBB_Martins'].iloc[idx]
            eval_result['HIA'][i] = admet['HIA_Hou'].iloc[idx]
            eval_result['hERG'][i] = admet['hERG'].iloc[idx]
            eval_result['Ames'][i] = admet['AMES'].iloc[idx]
            eval_result['Carcinogenicity'][i] = admet['Carcinogens_Lagunin'].iloc[idx]
            eval_result['CaCo2'][i] = 10 ** admet['Caco2_Wang'].iloc[idx]
            idx += 1


    print('Initial sample done!')
    # print('Initial sample results: ', eval_result)
    # save to json
    result_path = f'{args.result_path}/{args.pdb_filename}_eval_results_init.json'
    with open(result_path, 'w') as f:
        json.dump(eval_result, f)
    result_path_list.append(result_path)
    # calculate mean with None
    print('Initial batch mean QED with None: ', np.mean([x if x is not None else 0 for x in qed]))
    print('Initial batch mean SA with None: ', np.mean([x if x is not None else 0 for x in sa]))
    print('Initial batch mean LogP with None: ', np.mean([x if x is not None else 0 for x in logp]))
    print('Initial batch mean Lipinski with None: ', np.mean([x if x is not None else 0 for x in lipinski]))
    print('Initial batch mean BBBP with None: ', np.mean([x if x is not None else 0 for x in eval_result['BBBP']]))
    print('Initial batch mean HIA with None: ', np.mean([x if x is not None else 0 for x in eval_result['HIA']]))
    print('Initial batch mean hERG with None: ', np.mean([x if x is not None else 0 for x in eval_result['hERG']]))
    print('Initial batch mean Ames with None: ', np.mean([x if x is not None else 0 for x in eval_result['Ames']]))
    print('Initial batch mean Carcinogenicity with None: ', np.mean([x if x is not None else 0 for x in eval_result['Carcinogenicity']]))
    print('Initial batch mean CaCo2 with None: ', np.mean([x if x is not None else 0 for x in eval_result['CaCo2']]))
    # calculate mean without None
    print('Initial batch mean QED: ', np.mean([x for x in qed if x is not None]))
    print('Initial batch mean SA: ', np.mean([x for x in sa if x is not None]))
    print('Initial batch mean LogP: ', np.mean([x for x in logp if x is not None]))
    print('Initial batch mean Lipinski: ', np.mean([x for x in lipinski if x is not None]))
    print('Initial batch mean BBBP: ', np.mean([x for x in eval_result['BBBP'] if x is not None]))
    print('Initial batch mean HIA: ', np.mean([x for x in eval_result['HIA'] if x is not None]))
    print('Initial batch mean hERG: ', np.mean([x for x in eval_result['hERG'] if x is not None]))
    print('Initial batch mean Ames: ', np.mean([x for x in eval_result['Ames'] if x is not None]))
    print('Initial batch mean Carcinogenicity: ', np.mean([x for x in eval_result['Carcinogenicity'] if x is not None]))
    print('Initial batch mean CaCo2: ', np.mean([x for x in eval_result['CaCo2'] if x is not None]))
    print('Initial batch mean CaCo2 log10: ', np.mean([np.log10(x) for x in eval_result['CaCo2'] if x is not None]))

    for step in range(args.opt_steps):
        print('Optimization step: ', step)
        ################## Gradient Estimation ##################
        print('Starting Gradient Estimation')
        # Gradient Estimation
        init_pos_per = [0] * args.num_estimates
        gaussian_noise_per = [0] * args.num_estimates
        gumbel_noise_per = [0] * args.num_estimates
        # print('init_ligand_pos: ', init_ligand_pos)

        for i in range(args.num_estimates):
            init_pos_per[i] = torch.randn_like(init_ligand_pos)
            init_pos_per[i] = init_pos_per[i] / torch.norm(init_pos_per[i], dim=-1, keepdim=True) * args.per_size
            gaussian_noise_per[i] = [torch.randn_like(init_ligand_pos) for _ in range(len(gaussian_noise_traj))]
            
            gaussian_noise_per[i] = [gaussian_noise_per[i][j] / torch.norm(gaussian_noise_per[i][j], dim=-1, keepdim=True) * args.per_size for j in range(len(gaussian_noise_traj))]
            log_ligand_v = index_to_log_onehot(init_ligand_v, model.num_classes+int(model.v_mode=='tomask'))
            gumbel_noise_per[i] = [torch.randn_like(log_ligand_v) for _ in range(len(gumbel_noise_traj))]
            gumbel_noise_per[i] = [gumbel_noise_per[i][j] / torch.norm(gumbel_noise_per[i][j], dim=-1, keepdim=True) * args.per_size for j in range(len(gumbel_noise_traj))]

        # Unbatch the optimization variables and perturbations
        init_pos_list = []
        gaussian_noise_list = []
        gumbel_noise_list = []
        pos_per_list = []
        gaussian_per_list = []
        gumbel_per_list = []
        for i in range(args.num_samples):
            init_pos_list.append(init_ligand_pos[ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]])
            gaussian_noise_list.append([gaussian_noise_traj[j][ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] for j in range(len(gaussian_noise_traj))])
            gumbel_noise_list.append([gumbel_noise_traj[j][ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] for j in range(len(gumbel_noise_traj))])
            pos_per_list.append([init_pos_per[j][ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] for j in range(args.num_estimates)])
            gaussian_per_list_buffer = []
            gumbel_per_list_buffer = []
            for j in range(args.num_estimates):
                gaussian_per_list_buffer.append([gaussian_noise_per[j][k][ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] for k in range(len(gaussian_noise_traj))])
                gumbel_per_list_buffer.append([gumbel_noise_per[j][k][ligand_cum_atoms[i]:ligand_cum_atoms[i + 1]] for k in range(len(gumbel_noise_traj))])
            gaussian_per_list.append(gaussian_per_list_buffer)
            gumbel_per_list.append(gumbel_per_list_buffer)

        start = time.time()
        results = []
        for i in range(2 * args.num_estimates):
            # Use single device for sequential execution
            device = args.device
            if i < args.num_estimates:
                init_ligand_pos_perturbed = init_ligand_pos + init_pos_per[i]
                gaussian_noise_traj_perturbed = [gaussian_noise_traj[j] + gaussian_noise_per[i][j] for j in range(len(gaussian_noise_traj))]
                gumbel_noise_traj_perturbed = [gumbel_noise_traj[j] + gumbel_noise_per[i][j] for j in range(len(gumbel_noise_traj))]
            else:
                init_ligand_pos_perturbed = init_ligand_pos - init_pos_per[i - args.num_estimates]
                gaussian_noise_traj_perturbed = [gaussian_noise_traj[j] - gaussian_noise_per[i - args.num_estimates][j] for j in range(len(gaussian_noise_traj))]
                gumbel_noise_traj_perturbed = [gumbel_noise_traj[j] - gumbel_noise_per[i - args.num_estimates][j] for j in range(len(gumbel_noise_traj))]
            
            # Call sample_wrapper directly (sequential execution)
            result_path = sample_wrapper(model, data, args.num_samples,
                                        args.batch_size, device, config.sample.num_steps,
                                        config.sample.center_pos_mode, config.sample.sample_num_atoms,
                                        init_ligand_pos_perturbed, init_ligand_v, gaussian_noise_traj_perturbed, gumbel_noise_traj_perturbed, ligand_num_atoms, ligand_cum_atoms, batch_ligand, i, args)
            
            # Read result from file
            with open(result_path, 'r') as f:
                result = json.load(f)
            results.append(result)
            print(f'Sequential run {i} completed')

        print('all samples generated!')
        print('all samples generated time: ', time.time() - start)
        # sort the results by run id
        results.sort(key=lambda x: x['Run ID'])

        # print('results: ', results)

        # collect the result SMILES
        smiles_results = []
        for i in range(2 * args.num_estimates):
            if results[i]['smiles'] is None:
                smiles_results.append(None)
                continue
            smiles_results.extend(results[i]['smiles'])
        # print('smiles_results: ', smiles_results)
        # save the smiles_results to txt file
        with open(f'{args.result_path}/{args.pdb_filename}_smiles_results_step_{step}_intermediate.txt', 'w') as f:
            for smiles in smiles_results:
                if smiles is None:
                    f.write('NOSMILE' + '\n')
                    continue
                f.write(smiles + '\n')
        # collect the admet_ai results
        admet = get_admet_ai(f'{args.result_path}/{args.pdb_filename}_smiles_results_step_{step}_intermediate.txt')
        # try:
        print('admet SMILES:', admet.index.values)

        admet_results = {}
        admet_results['BBB'] = [None] * (2 * args.num_estimates)
        admet_results['HIA'] = [None] * (2 * args.num_estimates)
        admet_results['hERG'] = [None] * (2 * args.num_estimates)
        admet_results['Ames'] = [None] * (2 * args.num_estimates)
        admet_results['Carcinogenicity'] = [None] * (2 * args.num_estimates)
        admet_results['CaCo2'] = [None] * (2 * args.num_estimates)
        idx = 0
        for i in range(2 * args.num_estimates):
            admet_results['BBB'][i] = []
            admet_results['HIA'][i] = []
            admet_results['hERG'][i] = []
            admet_results['Ames'][i] = []
            admet_results['Carcinogenicity'][i] = []
            admet_results['CaCo2'][i] = []

            if results[i]['smiles'] is None:
                admet_results['BBB'][i] = [None] * args.num_samples
                admet_results['HIA'][i] = [None] * args.num_samples
                admet_results['hERG'][i] = [None] * args.num_samples
                admet_results['Ames'][i] = [None] * args.num_samples
                admet_results['Carcinogenicity'][i] = [None] * args.num_samples
                admet_results['CaCo2'][i] = [None] * args.num_samples
                continue
            for j in range(len(results[i]['smiles'])):
                if results[i]['smiles'][j] is None:
                    admet_results['BBB'][i].append(None)
                    admet_results['HIA'][i].append(None)
                    admet_results['hERG'][i].append(None)
                    admet_results['Ames'][i].append(None)
                    admet_results['Carcinogenicity'][i].append(None)
                    admet_results['CaCo2'][i].append(None)
                    continue
                if not results[i]['smiles'][j] in admet.index.values:
                    admet_results['BBB'][i].append(None)
                    admet_results['HIA'][i].append(None)
                    admet_results['hERG'][i].append(None)
                    admet_results['Ames'][i].append(None)
                    admet_results['Carcinogenicity'][i].append(None)
                    admet_results['CaCo2'][i].append(None)
                    continue
                admet_results['BBB'][i].append(admet['BBB_Martins'].iloc[idx])
                admet_results['HIA'][i].append(admet['HIA_Hou'].iloc[idx])
                admet_results['hERG'][i].append(admet['hERG'].iloc[idx])
                admet_results['Ames'][i].append(admet['AMES'].iloc[idx])
                admet_results['Carcinogenicity'][i].append(admet['Carcinogens_Lagunin'].iloc[idx])
                admet_results['CaCo2'][i].append(10 ** admet['Caco2_Wang'].iloc[idx])
                idx += 1
                print(idx)

        # put the admet results into the results
        for i in range(2 * args.num_estimates):
            results[i]['BBBP'] = admet_results['BBB'][i]
            results[i]['HIA'] = admet_results['HIA'][i]
            results[i]['hERG'] = admet_results['hERG'][i]
            results[i]['Ames'] = admet_results['Ames'][i]
            results[i]['Carcinogenicity'] = admet_results['Carcinogenicity'][i]
            results[i]['CaCo2'] = admet_results['CaCo2'][i]
        # print the results
        print(results)
            
        # Calculate the gradients
        grads_pos = [0] * args.num_samples
        grads_gaussian = [0] * args.num_samples
        grads_gumbel = [0] * args.num_samples
        grads = [0] * args.num_samples
        for i in range(args.num_samples):
            grads_pos[i] = {}
            grads_gaussian[i] = {}
            grads_gumbel[i] = {}
            grads[i] = {}
            grads_pos_list = {}
            grads_gaussian_list = {}
            grads_gumbel_list = {}
            for key in OPT_KEYS:
                grads_pos_list[key] = []
                grads_gaussian_list[key] = []
                grads_gumbel_list[key] = []

            for j in range(args.num_estimates):
                # calculate gradients for each key
                for key in OPT_KEYS:
                    try:
                        val_pos = results[j][key][i]
                        val_neg = results[j + args.num_estimates][key][i]
                    except:
                        val_pos = None
                        val_neg = None
                    if val_pos is None or val_neg is None:
                        continue
                    # if i == 0 and j == 0:
                    #     print('val_pos: ', val_pos)
                    #     print('val_neg: ', val_neg)
                    #     print('pos_per_list: ', pos_per_list[i])
                    if key in OPT_KEYS_MIN:
                        val_pos = -val_pos
                        val_neg = -val_neg
                    grad_pos_j = (val_pos - val_neg) * pos_per_list[i][j] / (2 * args.per_size)
                    grads_pos_list[key].append(grad_pos_j)
                    grad_gaussian_j = [(val_pos - val_neg) * gaussian_per_list[i][j][k] / (2 * args.per_size) for k in range(len(gaussian_per_list[i][j]))]
                    grads_gaussian_list[key].append(grad_gaussian_j)
                    grad_gumbel_j = [(val_pos - val_neg) * gumbel_per_list[i][j][k] / (2 * args.per_size) for k in range(len(gumbel_per_list[i][j]))]
                    grads_gumbel_list[key].append(grad_gumbel_j)
            # calculate the mean of the gradients
            grads[i]['valid'] = True
            for key in grads_pos_list.keys():
                if len(grads_pos_list[key]) == 0:
                    print(f'No gradients for {key} in sample {i}')
                    grads_pos[i][key] = torch.zeros_like(pos_per_list[i][0])
                    grads_gaussian[i][key] = [torch.zeros_like(gaussian_per_list[i][0][0]) for _ in range(len(gaussian_per_list[i][0]))]
                    grads_gumbel[i][key] = [torch.zeros_like(gumbel_per_list[i][0][0]) for _ in range(len(gumbel_per_list[i][0]))]
                    grads[i][key] = [grads_pos[i][key]] + grads_gaussian[i][key] + grads_gumbel[i][key]
                    grads[i]['valid'] = False
                    continue
                grads_pos[i][key] = sum([grads_pos_list[key][j] for j in range(len(grads_pos_list[key]))]) / len(grads_pos_list[key])
                grads_gaussian[i][key] = [sum([grads_gaussian_list[key][j][k] for j in range(len(grads_gaussian_list[key]))]) / len(grads_gaussian_list[key]) for k in range(len(gaussian_per_list[i][0]))]
                grads_gumbel[i][key] = [sum([grads_gumbel_list[key][j][k] for j in range(len(grads_gumbel_list[key]))]) / len(grads_gumbel_list[key]) for k in range(len(gumbel_per_list[i][0]))]
                grads[i][key] = [grads_pos[i][key]] + grads_gaussian[i][key] + grads_gumbel[i][key]
            

        descent_dir = [0] * args.num_samples
        descent_dir_pos = [0] * args.num_samples
        descent_dir_gaussian = [0] * args.num_samples
        descent_dir_gumbel = [0] * args.num_samples

        if len(OPT_KEYS) == 1:
            # If only one key, we can directly use the gradient as the descent direction
            for i in range(args.num_samples):
                descent_dir[i] = grads[i][OPT_KEYS[0]]
                # print(descent_dir[i])
                # pdb.set_trace()
                descent_dir_pos[i] = descent_dir[i][0]
                descent_dir_gaussian[i] = descent_dir[i][1:len(gaussian_per_list[i][0]) + 1]
                descent_dir_gumbel[i] = descent_dir[i][len(gaussian_per_list[i][0]) + 1:]
        else:
        # Calculate the descent direction
            for i in tqdm(range(args.num_samples)):
                # print(f'grads for sample {i}: ', grads[i])
                print(f'grads shape for sample {i}: ', len(grads[i]))
                descent_dir[i] = get_descent_dir(grads[i])
                if descent_dir[i] is None:
                    print('No valid descent direction, skip sample %s' % f'{i}')
                    descent_dir_pos[i] = torch.zeros_like(pos_per_list[i][0])
                    descent_dir_gaussian[i] = [torch.zeros_like(gaussian_per_list[i][0][0]) for _ in range(len(gaussian_per_list[i][0]))]
                    descent_dir_gumbel[i] = [torch.zeros_like(gumbel_per_list[i][0][0]) for _ in range(len(gumbel_per_list[i][0]))]
                    continue
                print(f'Descent direction shape for sample {i}: ', len(descent_dir[i]))
                descent_dir_pos[i] = descent_dir[i][0]
                descent_dir_gaussian[i] = descent_dir[i][1:len(gaussian_per_list[i][0]) + 1]
                descent_dir_gumbel[i] = descent_dir[i][len(gaussian_per_list[i][0]) + 1:]

        print('Descent direction calculated, updating noise')
        ### Update ###
        for i in range(args.num_samples):
            init_pos_list[i] = init_pos_list[i] + descent_dir_pos[i].to('cuda')
            init_pos_list[i].to(dtype=torch.float32)
            try:
                gaussian_noise_list[i] = [gaussian_noise_list[i][j] + descent_dir_gaussian[i][j].to('cuda') for j in range(len(gaussian_noise_list[i]))]
                gaussian_noise_list[i] = [gaussian_noise_list[i][j].to(dtype=torch.float32) for j in range(len(gaussian_noise_list[i]))]
                gumbel_noise_list[i] = [gumbel_noise_list[i][j] + descent_dir_gumbel[i][j].to('cuda') for j in range(len(gumbel_noise_list[i]))]
                gumbel_noise_list[i] = [gumbel_noise_list[i][j].to(dtype=torch.float32) for j in range(len(gumbel_noise_list[i]))]
            except:
                print('gaussian noise length', len(gaussian_noise_list[i]))
                print('gaussian descent direction length', len(descent_dir_gaussian[i]))
                print('gumbel noise length', len(gumbel_noise_list[i]))
                print('gumbel descent direction length', len(descent_dir_gumbel[i]))
                raise ValueError('Gaussian or Gumbel noise length mismatch')

        # Rebatch the optimization variables
        init_ligand_pos = torch.cat(init_pos_list, dim=0)
        gaussian_noise_traj = []
        for i in range(len(gaussian_noise_list[0])):
            gaussian_noise_traj.append(torch.cat([gaussian_noise_list[j][i] for j in range(args.num_samples)], dim=0))
        gumbel_noise_traj = []
        for i in range(len(gumbel_noise_list[0])):
            gumbel_noise_traj.append(torch.cat([gumbel_noise_list[j][i] for j in range(args.num_samples)], dim=0))
        init_ligand_pos = init_ligand_pos.to(device=args.device)
        gaussian_noise_traj = [noise.to(device=args.device) for noise in gaussian_noise_traj]
        gumbel_noise_traj = [noise.to(device=args.device) for noise in gumbel_noise_traj]
        # save the noise to npy files
        # np.save(f'{args.result_path}/{args.pdb_filename}_init_ligand_pos_step_{step}.npy', init_ligand_pos.cpu().numpy())
        # np.save(f'{args.result_path}/{args.pdb_filename}_init_ligand_v_step_{step}.npy', init_ligand_v.cpu().numpy())
        # np.save(f'{args.result_path}/{args.pdb_filename}_gaussian_noise_traj_step_{step}.npy', [noise.cpu().numpy() for noise in gaussian_noise_traj], allow_pickle=True)
        # np.save(f'{args.result_path}/{args.pdb_filename}_gumbel_noise_traj_step_{step}.npy', [noise.cpu().numpy() for noise in gumbel_noise_traj], allow_pickle=True)
        # np.save(f'{args.result_path}/{args.pdb_filename}_ligand_num_atoms_step_{step}.npy', ligand_num_atoms)
        # np.save(f'{args.result_path}/{args.pdb_filename}_ligand_cum_atoms_step_{step}.npy', ligand_cum_atoms)
        # np.save(f'{args.result_path}/{args.pdb_filename}_batch_ligand_step_{step}.npy', batch_ligand.cpu().numpy())

        print('Generating new samples')
        ### Generate new samples ###
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, pred_pos_cond_traj, pred_v_cond_traj, init_ligand_pos, gaussian_noise_traj, init_ligand_v, gumbel_noise_traj, ligand_num_atoms, ligand_cum_atoms, batch_ligand = sample_diffusion_ligand_opt(
            model.to(args.device), data, args.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            gaussian_noise_traj=gaussian_noise_traj,
            gumbel_noise_traj=gumbel_noise_traj,
            ligand_num_atoms=ligand_num_atoms,
            ligand_cum_atoms=ligand_cum_atoms,
            batch_ligand=batch_ligand
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'time': time_list,
            'pred_ligand_pos_cond_traj': pred_pos_cond_traj,
            'pred_ligand_v_cond_traj': pred_v_cond_traj,
        }

        all_pred_ligand_pos = result['pred_ligand_pos_traj']
        all_pred_ligand_v = result['pred_ligand_v_traj']
        # evaluate
        center = result['data']['ligand'].ligand_protein_pos_center.numpy()
        qed = []
        sa = []
        logp = []
        lipinski = []
        smiles_results = []
        # bbbp = []
        reconstruct_idx = 0
        for sample_idx, (pred_pos_all, pred_v_all) in enumerate(tqdm((zip(all_pred_ligand_pos, all_pred_ligand_v)))):
            pred_pos, pred_v = pred_pos_all[-1], pred_v_all[-1]
            pred_pos = pred_pos + center
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except:
                print('Reconstruct failed %s' % f'{sample_idx}')
                qed.append(None)
                sa.append(None)
                logp.append(None)
                lipinski.append(None)
                smiles_results.append(None)
                reconstruct_idx += 1
                continue
            if '.' in smiles:
                print('Invalid SMILES: %s' % f'{smiles}')
                qed.append(None)
                sa.append(None)
                logp.append(None)
                lipinski.append(None)
                smiles_results.append(None)
                reconstruct_idx += 1
                continue
            smiles_results.append(smiles)
            print(f'run {i} SMILES: {smiles}')
            try:
                chem_results = scoring_func.get_chem(mol)
            except:
                print('Chem scoring failed %s' % f'{sample_idx}')
                qed.append(None)
                sa.append(None)
                logp.append(None)
                lipinski.append(None)
                # bbbp.append(None)
                # smiles_results.append(smiles)
                reconstruct_idx += 1
                continue
            qed.append(chem_results['qed'])
            sa.append(chem_results['sa'])
            logp.append(chem_results['logp'])
            lipinski.append(chem_results['lipinski'])
            reconstruct_idx += 1
            writer = Chem.SDWriter(f"{args.result_path}/{args.sdf_filename}_generated_{step}_{sample_idx}.sdf")
            writer.write(mol)
            writer.close()
        eval_result = {
            'smiles': smiles_results,
            'QED': qed,
            'SA': sa,
            'LogP': logp,
            'Lipinski': [int(lipinski) if lipinski is not None else None for lipinski in lipinski],
        }
        # save smiles results to txt
        invalid_idx = []
        with open(f'{args.result_path}/{args.pdb_filename}_smiles_results_step_{step}.txt', 'w') as f:
            for idx, smiles in enumerate(smiles_results):
                if smiles is None:
                    invalid_idx.append(idx)
                    continue
                f.write(smiles + '\n')
        # collect the admetAI results
        admet = get_admet_ai(f'{args.result_path}/{args.pdb_filename}_smiles_results_step_{step}.txt')
        if admet is None:
            print('ADMET AI prediction failed')
            eval_result['BBBP'] = [None] * len(smiles_results)
            eval_result['HIA'] = [None] * len(smiles_results)
            eval_result['hERG'] = [None] * len(smiles_results)
            eval_result['Ames'] = [None] * len(smiles_results)
            eval_result['Carcinogenicity'] = [None] * len(smiles_results)
            eval_result['CaCo2'] = [None] * len(smiles_results)
        else:
            for i in range(len(smiles_results)):
                if smiles_results[i] is None:
                    invalid_idx.append(i)
                    continue
                if not smiles_results[i] in admet.index.values:
                    invalid_idx.append(i)
                    continue
            eval_result['BBBP'] = [None] * len(smiles_results)
            eval_result['HIA'] = [None] * len(smiles_results)
            eval_result['hERG'] = [None] * len(smiles_results)
            eval_result['Ames'] = [None] * len(smiles_results)
            eval_result['Carcinogenicity'] = [None] * len(smiles_results)
            eval_result['CaCo2'] = [None] * len(smiles_results)
            print('SMILES results length: ', len(smiles_results))
            print('ADMET AI SMILES length: ', len(admet.index.values))
            print('invalid idx: ', invalid_idx)
            # parse the admet results to valid index
            idx = 0
            for i in range(len(smiles_results)):
                if i in invalid_idx:
                    continue
                try:
                    eval_result['BBBP'][i] = admet['BBB_Martins'].iloc[idx]
                except:
                    print(f'ADMET AI prediction failed for sample {i}, index {idx}')
                    print(eval_result['smiles'])
                    print(admet.index.values)
                eval_result['HIA'][i] = admet['HIA_Hou'].iloc[idx]
                eval_result['hERG'][i] = admet['hERG'].iloc[idx]
                eval_result['Ames'][i] = admet['AMES'].iloc[idx]
                eval_result['Carcinogenicity'][i] = admet['Carcinogens_Lagunin'].iloc[idx]
                eval_result['CaCo2'][i] = admet['Caco2_Wang'].iloc[idx]
                idx += 1

        # save to json
        result_path = f'{args.result_path}/{args.pdb_filename}_eval_results_{step}.json'
        with open(result_path, 'w') as f:
            json.dump(eval_result, f)
        result_path_list.append(result_path)
        print('Optimization step %s done!' % f'{step}')

        # calculate mean with None
        print(f'lr: {args.opt_lr}, per_size: {args.per_size}, step: {step}')
        print(f'mean QED with None: ', np.mean([x if x is not None else 0 for x in qed]))
        print(f'mean SA with None: ', np.mean([x if x is not None else 0 for x in sa]))
        print(f'mean LogP with None: ', np.mean([x if x is not None else 0 for x in logp]))
        print(f'mean Lipinski with None: ', np.mean([x if x is not None else 0 for x in lipinski]))
        print(f'mean BBBP with None: ', np.mean([x if x is not None else 0 for x in eval_result['BBBP']]))
        print(f'mean HIA with None: ', np.mean([x if x is not None else 0 for x in eval_result['HIA']]))
        print(f'mean hERG with None: ', np.mean([x if x is not None else 0 for x in eval_result['hERG']]))
        print(f'mean Ames with None: ', np.mean([x if x is not None else 0 for x in eval_result['Ames']]))
        print(f'mean Carcinogenicity with None: ', np.mean([x if x is not None else 0 for x in eval_result['Carcinogenicity']]))
        print(f'mean CaCo2 with None: ', np.mean([x if x is not None else 0 for x in eval_result['CaCo2']]))

        # calculate mean without None
        print(f'mean QED: ', np.mean([x for x in qed if x is not None]))
        print(f'mean SA: ', np.mean([x for x in sa if x is not None]))
        print(f'mean LogP: ', np.mean([x for x in logp if x is not None]))
        print(f'mean Lipinski: ', np.mean([x for x in lipinski if x is not None]))
        print(f'mean BBBP: ', np.mean([x for x in eval_result['BBBP'] if x is not None]))
        print(f'mean HIA: ', np.mean([x for x in eval_result['HIA'] if x is not None]))
        print(f'mean hERG: ', np.mean([x for x in eval_result['hERG'] if x is not None]))
        print(f'mean Ames: ', np.mean([x for x in eval_result['Ames'] if x is not None]))
        print(f'mean Ames: ', np.mean([x for x in eval_result['Ames'] if x is not None]))
        print(f'mean Carcinogenicity: ', np.mean([x for x in eval_result['Carcinogenicity'] if x is not None]))
        print(f'mean CaCo2: ', np.mean([x for x in eval_result['CaCo2'] if x is not None]))



if __name__ == '__main__':
    main()
