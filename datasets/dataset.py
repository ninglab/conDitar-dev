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

import os
import pickle
import lmdb
from tqdm.auto import tqdm
import utils.misc as misc
import numpy as np
import abc

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Compose
from utils.data import *
from utils.transforms import *
from .dataset import *
from .mol_data import *
from models.pocket_modelAE import PocketModel
from datasets.mol_data import FOLLOW_BATCH
from utils.device import resolve_device


class AbstractRawDataset(Dataset):

    def __init__(self, config, files):
        super().__init__()
        self.config = config
        self.files = files
        
        self.hydrogen = config.hydrogen
    
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass


class RawDataset(AbstractRawDataset):
    def __init__(self, config, files, radius, ligand_transform):
        super().__init__(config.data, files)
        self.ckpt_path = config.model.checkpoint_pocket
        self.config = config
        self.radius = radius
        self.ligand_transform = ligand_transform
        self._build_model()

    def _build_model(self):
        
        self.model = PocketModel(config = self.config.pocket,
                                hydrogen = self.config.data.hydrogen)
        # map_location lets CUDA-trained checkpoints load during CPU-only container runs.
        checkpoint = torch.load(self.ckpt_path, map_location=resolve_device())
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data, data_ligand = parse_pdb(self.files[idx][0], self.files[idx][1], self.radius)
        data = ComplexData.from_ligand_dicts(
                        ligand_dict=torchify_dict(data_ligand),
                        protein_dict=torchify_dict(data)
                    ).to_dict()
        pocket_mean = data['protein_pos'].mean(0)
        data_pocket = parse_protein_pocket(data, hydrogen=self.hydrogen)
        if data_ligand != None:
            data_ligand = parse_ligand_pocket(data)
        else:
            data_ligand = {}
        
        pocket_data = PocketData.from_pocket_dicts(
            torchify_dict(data_pocket)
            )
        batch = Batch.from_data_list([pocket_data])
        with torch.no_grad():
            embd_scaler, embd_vector = self.model.get_embds(batch) 

        data_pocket['embd_scalar'] = np.array(embd_scaler)
        data_pocket['embd_vector'] = np.array(embd_vector)

        data_ligand['filename'] = self.files[idx][1]
        data_ligand['protein_pos_center'] = pocket_mean

        pocket_data = PocketData.from_pocket_dicts(
            torchify_dict(data_pocket)
            )
        ligand_data = LigandData.from_ligand_dicts(
            torchify_dict(data_ligand)
            )
        
        if self.ligand_transform is not None and self.files[idx][1] is not None:
            ligand_data = self.ligand_transform(ligand_data)

        pocket_data.id = idx
        ligand_data.id = idx
        
        return {'pocket': pocket_data, 'ligand': ligand_data}



class AbstractComplexDataset(Dataset):
    env = None

    def __init__(self, config, train):
        super().__init__()
        self.config = config
        self.raw_path = config.path.rstrip('/')
        self.processed_dir = config.processed_path
        self.train = train

        if train:
            self.processed_path = os.path.join(self.processed_dir,
                                            config.dataset + f'_train_processed.lmdb') 
        
        self.hydrogen = config.hydrogen
        if self.hydrogen == True:  
            self.atom_dict = ATOM_INDEX_DICT  
        else:
            self.atom_dict = {key: value-1 for key, value in ATOM_INDEX_DICT.items() if key != 'H'}

        if config.transform == True:                           
            self.transform = PocketRandomMask(self.atom_dict, 
                                            noise_type = config.noise_type,
                                            noise = config.noise,
                                            mask_prob = config.mask_prob,
                                            unmask_prob = config.unmask_prob,
                                            mask_mode = config.mask_mode,
                                            mask_backbone = config.backbone)
        else:
            self.transform = None


class ComplexDataset(AbstractComplexDataset):
    def __init__(self, config, train, ckpt_path, ligand_transform):
        super().__init__(config, train)
        self.ckpt_path = ckpt_path
        self.split = config.split

        if self.ckpt_path != None:
            self.all_config = misc.load_config(config.pocket_config_path)
            self._build_model()
            
        self.ligand_transform = ligand_transform

    def _build_model(self):
        self.model = PocketModel(config = self.all_config.model,
                                hydrogen = self.all_config.data.hydrogen)
        checkpoint = torch.load(self.ckpt_path)
        try:
            self.model.load_state_dict(checkpoint["state_dict"])
        except:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()


    def _connect_db(self):
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._preprocess_cr() 

        self.db = lmdb.open(
                self.processed_path,
                map_size=self.config.datasize * (1024 * 1024 * 1024), 
                create=False,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            self.size = len(self.keys)


    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if not hasattr(self, 'keys'):
            self._connect_db()
        return len(self.keys)

    def _preprocess_cr(self):
        split = torch.load(self.split)['train']

        print("%d complex to be processed" % len(split))

        self.db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,
            lock=True,
        )

        write_every = 5000  
        idx = 0
        failed = 0

        txn = self.db.begin(write=True)

        for index in tqdm(split):
            try:
                files = (
                    os.path.join(self.raw_path, index[0]),
                    os.path.join(self.raw_path, index[1])
                )

                data, data_ligand = parse_pdb(files[0], files[1])
                data = ComplexData.from_ligand_dicts(
                    ligand_dict=torchify_dict(data_ligand),
                    protein_dict=torchify_dict(data)
                ).to_dict()

                pocket_mean = data['protein_pos'].mean(0)

                data_pocket = parse_protein_pocket(data, hydrogen=self.hydrogen)
                data_ligand = parse_ligand_pocket(data)

                pocket_data = PocketData.from_pocket_dicts(
                    torchify_dict(data_pocket)
                )
                batch = Batch.from_data_list([pocket_data])

                with torch.no_grad():
                    embd_scalar, embd_vector = self.model.get_embds(batch)

                data_pocket['embd_scalar'] = embd_scalar.detach().cpu().numpy()
                data_pocket['embd_vector'] = embd_vector.detach().cpu().numpy()
                data_pocket['filename'] = index[0]

                data_ligand['filename'] = index[1]
                data_ligand['protein_pos_center'] = pocket_mean

                dict_data = {
                    'pocket': data_pocket,
                    'ligand': data_ligand
                }

                txn.put(
                    key=str(idx).encode(),
                    value=pickle.dumps(dict_data, protocol=pickle.HIGHEST_PROTOCOL)
                )

                idx += 1

                if idx % write_every == 0:
                    txn.commit()
                    txn = self.db.begin(write=True)

            except Exception as e:
                failed += 1
                print(e)

        txn.commit()
        self.db.close()

        print(f"Finished preprocessing: {idx} entries, {failed} failed")

    def __getitem__(self, idx):
        if not hasattr(self, 'keys'):
            self._connect_db()
            print('connect db')

        with self.db.begin() as txn:
            try:
                data = pickle.loads(txn.get(idx))
            except:
                data = pickle.loads(txn.get(self.keys[idx]))

        data_pocket = data['pocket']
        data_ligand = data['ligand']

        pocket_data = PocketData.from_pocket_dicts(
            torchify_dict(data_pocket)
            )
        
        if self.transform is not None: 
            data_dict, mask = self.transform(pocket_data)
            pocket_data = PocketData.from_pocket_dicts(
                torchify_dict(data_dict)
            )
            pocket_data.mask = mask

        ligand_data = LigandData.from_ligand_dicts(
            torchify_dict(data_ligand)
            )
        
        if self.ligand_transform is not None:
            ligand_data = self.ligand_transform(ligand_data)
        
        pocket_data.id = idx
        ligand_data.id = idx

        out = {'pocket': pocket_data, 'ligand': ligand_data}
        for key, value in data.items():
            if key in {'pocket', 'ligand'}:
                continue
            out[key] = value
        return out
 
