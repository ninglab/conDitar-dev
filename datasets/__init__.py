import torch
from torch.utils.data import Subset
from. dataset import *

def get_dataset(config, name, *args, **kwargs):
    if name == 'train':
        dataset = ComplexDataset(config, *args, **kwargs) 
    elif name == 'test':
        dataset = RawDataset(config, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split_val' in config and kwargs['train'] == True:
        subsets = {}
        try:
            split = torch.load(config.split_val)
        except:
            split = torch.load(config.data.split_val)
        subsets['train'] = Subset(dataset, indices=split['train'])
        subsets['valid'] = Subset(dataset, indices=split['valid'])
        return dataset, subsets
    else:
        return dataset


