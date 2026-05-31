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
import shutil
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import Subset

from datasets import get_dataset
from datasets.mol_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
import utils.transforms as trans
import utils.misc as misc
import utils.train as utils_train
import utils.data as utils_data
from utils import analyze
from rdkit import Chem
import time
from sklearn.metrics import roc_auc_score
import pickle
from scripts.sample_diffusion import sample_diffusion_ligand
from utils import reconstruct
from utils.docking_vina import VinaDockingTask
from utils import scoring_func
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



def validate_via_sample(model, batch_data, protein_root="", logger=None, num_samples=50, atom_enc_mode='add_aromatic'):
    n_recon_success, n_complete = 0, 0
    mols = []
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    all_sample_nums = 0
    all_chem_results = []
    all_vina_results = []
    for data in batch_data:
        all_ligand_pos, all_ligand_v, _, _, _, _, _, _, _ = sample_diffusion_ligand(
            model, data, num_samples, batch_size=num_samples, sample_num_atoms='ref'
        )
        all_sample_nums += len(all_ligand_pos)
        
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_ligand_pos, all_ligand_v)):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
            
            if 'ligand_protein_pos_center' in data['ligand']: pred_pos = pred_pos + data['ligand']['ligand_protein_pos_center'].numpy()
            
            try:
                mol_cal_bond = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol_cal_bond)
                mols.append(mol_cal_bond)
            except:
                continue
            try:
                chem_results = scoring_func.get_chem(mol_cal_bond)
                all_chem_results.append(chem_results)
            except:
                continue
            
            n_recon_success += 1

            if "." not in smiles:
                n_complete += 1

            if data['pocket'] is not None:
                try:
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol_cal_bond, data['ligand']['ligand_filename'], protein_root=protein_root, tmp_dir='/fs/ess/PCON0041/gruoxi/tmp')
                    vina_results = vina_task.run(mode='score_only', exhaustiveness=16)
                    vina_task.remove_tmp_file()
                except Exception as e:
                    print(e)
                    continue
                print(vina_results)
                all_vina_results.append(vina_results)

    mean_qed = np.mean([results['qed'] for results in all_chem_results]) if len(all_chem_results) > 0 else 0
    mean_sa = np.mean([results['sa'] for results in all_chem_results]) if len(all_chem_results) > 0 else 0
    mean_vina_score = np.mean([result[0]['affinity'] for result in all_vina_results])
    fraction_mol_stable = all_mol_stable / all_sample_nums
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / all_sample_nums
    fraction_complete = n_complete / all_sample_nums


    results = {
        'recon_success': fraction_recon,
        'complete': fraction_complete,
        'mol_stability': fraction_mol_stable, 
        'atom_stability': fraction_atm_stable,
        'mean_qed': mean_qed,
        'mean_sa': mean_sa,
        'mean_vina_score': mean_vina_score,
    }

    out_string = "[Sample Validate] "
    for key in results:
        if results[key] is None:
            out_string += "%s: none | " % (key)
        else:
            out_string += "%s: %.6f | " % (key, results[key])
        
    logger.info(out_string)

def get_auroc(y_true, y_pred, feat_mode=None, pred_type='atom'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        if pred_type == 'atom':
            mapping = {
                'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
                'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
                'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL,
            }
            logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
        elif pred_type == 'bond':
            mapping = trans.MAP_INDEX_TO_BOND_TYPE
            logger.info(f'bond: {mapping[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_bond_auroc(y_true, y_pred):
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        bond_type = {
            0: 'none',
            1: 'single',
            2: 'double',
            3: 'triple',
            4: 'aromatic',
        }
        logger.info(f'bond: {bond_type[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--change_log_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--continue_train_iter', type=int, default=-1)
    parser.add_argument('--logdir', type=str, default='diffusion_logs')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--model_ckpt_path', type=str, default='/fs/ess/PCON0041/gruoxi/SBDDcode/checkpoints/Diff.pt')
    args = parser.parse_args()

    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    if args.change_log_dir is not None:
        log_dir = args.change_log_dir
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = misc.get_logger('train_diff', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    
    if args.change_log_dir is None:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    print('Loading dataset...')
    
    dataset, subsets = get_dataset(config=config.data, name='train', train=True, ckpt_path=config.data.pocket_checkpoint, ligand_transform=transform)
    train_set, val_set = subsets['train'], subsets['valid']
    print(f'Training: {len(train_set)} Validation: {len(val_set)}')
    
    collate_exclude_keys = ['ligand_nbh_list']
    
    train_loader = DataLoader(train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
       )
    val_loader = DataLoader(val_set, 
                            config.train.batch_size, 
                            shuffle=False,
                            num_workers=config.train.num_workers, 
                            follow_batch=FOLLOW_BATCH,
                            exclude_keys=collate_exclude_keys,
                            )
    
    train_iterator = utils_train.inf_iterator(train_loader)

    logger.info('Building model...')
    model = ScorePosNet3D(
        config,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES)
    ).to(args.device)

    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    if args.model_ckpt_path is not None:
        try:
            print('Loading...')
            ckpt = torch.load(args.model_ckpt_path, map_location=args.device)
            model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
            logger.info(f'Successfully load the model! {args.model_ckpt_path}')
            ckpt_stem = os.path.basename(args.model_ckpt_path).split(".pt")[0]
            try:
                continue_train_iter = int(ckpt_stem)
                start_iter = continue_train_iter + 1
            except ValueError:
                start_iter = 1

            ckpt['optimizer']['param_groups'][-1]['lr'] = 1e-3
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            raise ValueError(
                f'Failed to load ckpt: {args.model_ckpt_path} '
                f'({type(e).__name__}: {e})'
            )
    else:
        start_iter = 1

    print(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    print(f'ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    def compute_batch_diffusion_loss(batch_ligand, batch_pocket, eval_mode=False):
        return model.get_diffusion_loss(
            ligand_data=batch_ligand,
            pocket_data=batch_pocket,
            eval_mode=eval_mode,
        )

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator)
            batch_ligand = batch['ligand'].to(args.device)
            batch_pocket = batch['pocket'].to(args.device)
            
            results = compute_batch_diffusion_loss(
                    batch_ligand=batch_ligand,
                    batch_pocket=batch_pocket,
                    eval_mode=False
            )
            
            loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux = \
                results['loss'], results['loss_pos'], results['loss_v'], results['loss_bond_final'], results['loss_bond_aux']
            loss_bond_dist, loss_bond_angle, loss_torsion_angle = \
                results['loss_bond_dist'], results['loss_bond_angle'], results['loss_torsion_angle']
            loss = loss / config.train.n_acc_batch
            loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

        if it % config.train.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | bond final %.6f | bond aux %.6f | bond_dist %.6f | bond_angle %.6f | torsion_angle %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux, loss_bond_dist, loss_bond_angle, loss_torsion_angle, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    def validate(it, sample_validate=False):
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_bond_final, sum_loss_bond_aux, sum_n = 0, 0, 0, 0, 0, 0
        sum_loss_bond_dist, sum_loss_bond_angle, sum_loss_torsion_angle = 0, 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch_ligand = batch['ligand'].to(args.device)
                batch_pocket = batch['pocket'].to(args.device)
                
                batch_size = batch_ligand.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []

                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    
                    results = compute_batch_diffusion_loss(
                            batch_ligand=batch_ligand,
                            batch_pocket=batch_pocket,
                            eval_mode=False
                    )
                    loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux = \
                        results['loss'], results['loss_pos'], results['loss_v'], results['loss_bond_final'], results['loss_bond_aux']
                    
                    loss_bond_dist, loss_bond_angle, loss_torsion_angle = \
                        results['loss_bond_dist'], results['loss_bond_angle'], results['loss_torsion_angle']

                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_bond_final += float(loss_bond_final) * batch_size
                    sum_loss_bond_aux += float(loss_bond_aux) * batch_size
                    sum_loss_bond_dist += float(loss_bond_dist) * batch_size
                    sum_loss_bond_angle += float(loss_bond_angle) * batch_size
                    sum_loss_torsion_angle += float(loss_torsion_angle) * batch_size
                    sum_n += batch_size
                    
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch_ligand.ligand_atom_feature_full.detach().cpu().numpy())
                    
                    if len(results['pred_bond_type']) != 0:
                        all_pred_bond_type.append(results['pred_bond_type'].detach().cpu().numpy())
                        all_gt_bond_type.append(results['gt_bond_type'].detach().cpu().numpy())
                

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_bond_final = sum_loss_bond_final / sum_n
        avg_loss_bond_aux = sum_loss_bond_aux / sum_n
        avg_loss_bond_dist = sum_loss_bond_dist / sum_n
        avg_loss_bond_angle = sum_loss_bond_angle / sum_n
        avg_loss_torsion_angle = sum_loss_torsion_angle / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)
        
        if len(all_pred_bond_type) != 0:
            bond_auroc = get_auroc(np.concatenate(all_gt_bond_type), np.concatenate(all_pred_bond_type, axis=0),
                                feat_mode=None, pred_type='bond')
        else:
            bond_auroc = 0.0
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss bond final %.6f | Loss bond aux %.6f | '
            'Loss bond_dist %.6f | Loss bond_angle %.6f | Loss torsion_angle %.6f | Avg atom auroc %.6f | Avg bond auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_bond_final, avg_loss_bond_aux, avg_loss_bond_dist, avg_loss_bond_angle, avg_loss_torsion_angle, atom_auroc, bond_auroc
            )
        )
        
        if sample_validate:
            sample_valid_data = [val_set[idx] for idx in np.random.choice(len(val_set), 2)]
            results = validate_via_sample(model, sample_valid_data, logger=logger, protein_root=config.data.path)

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/loss_bond_aux', avg_loss_bond_aux, it)
        writer.add_scalar('val/loss_bond_final', avg_loss_bond_final, it)
        writer.add_scalar('val/loss_bond_dist', avg_loss_bond_dist, it)
        writer.add_scalar('val/loss_bond_angle', avg_loss_bond_angle, it)
        writer.add_scalar('val/loss_torsion_angle', avg_loss_torsion_angle, it)
        writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it == start_iter: continue

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                if it % config.train.sample_freq == 0:
                    val_loss = validate(it, sample_validate=False)
                else:
                    val_loss = validate(it, sample_validate=False)

                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
