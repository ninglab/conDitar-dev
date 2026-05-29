# =============================================================================
# Copyright (c) The Ohio State University. All rights reserved.
# Licensed under the terms in LICENSE.txt.
# =============================================================================

import pdb
import os
import shutil
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from torch_geometric.transforms import Compose

from torch_geometric.loader import DataLoader

from datasets import get_dataset
from models.pocket_modelAE import PocketModel
import utils.transforms as trans
import utils.misc as misc
import utils.train as utils_train
import datetime
from collections import OrderedDict



def train_epoch(model, train_loader, optimizer, epoch, device):
    model.train() 
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
    
    for batch in progress_bar:
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        loss, type_loss, coord_loss, acc, diff = model.get_loss(batch['pocket'])
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()


        progress_bar.set_postfix(
                                OrderedDict([
                                    ('avg_loss', f"{total_loss / (progress_bar.n + 1):.4f}"),  
                                    ('loss', f"{loss.item():.4f}"),  
                                    ('diff', f"{diff.item():.4f}"),  
                                    ('acc', f"{acc.item():.4f}")
                                    ])
                                )
        
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            loss, _, _, _, _ = model.get_loss(batch['pocket']) 
            loss = loss.mean()
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PocketModel(config.model, hydrogen=config.data.hydrogen)
    model = model.to(device)
    if args.ckpt_path != None:
        try:
            checkpoint = torch.load(args.ckpt_path)
            model.load_state_dict(checkpoint["state_dict"])
        except:
            checkpoint = torch.load(args.ckpt_path)
            model.load_state_dict(checkpoint["model_state_dict"])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    best_loss = 10

    for epoch in range(config.train.epoch):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, device)
        val_loss = validate(model, test_loader, device)
        scheduler.step(val_loss)
        
        current_time = datetime.datetime.now().strftime('%m_%d__%H_%M_%S')
        print(current_time)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}.')
        
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'{model_dir}/checkpoint_{epoch+1}.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='pocket_enc_logs')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--ckpt_path', type=str, default=None)


    args = parser.parse_args()

    logger = misc.get_logger('pocket_enc')

    cuda_version = torch.version.cuda if torch.cuda.is_available() else "CUDA is not available"
    

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    current_time = datetime.datetime.now().strftime('%m_%d__%H_%M_%S')

    model_dir = os.path.join(args.logdir, f'{current_time}/checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    logger.info(model_dir)

    # Transforms
    ligand_featurizer = trans.FeaturizeLigandAtom('add_aromatic')
    transform_list = [
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]

    transform = Compose(transform_list)

    dataset, subsets = get_dataset(config=config.data, name='train', train=True, ckpt_path=args.ckpt_path, ligand_transform=transform)
    train_set, val_set = subsets['train'], subsets['valid']

    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=args.nworkers,
    )
    test_loader = DataLoader(
        val_set, 
        100, 
        shuffle=False, 
        num_workers=args.nworkers,
    )

    main(config)

