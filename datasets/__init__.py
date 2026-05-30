# =============================================================================
# conDitar is copyrighted by the Ohio State University and covered by US 64/023,113.

# conDitar may be licensed solely for educational and research purposes by
# non-profit institutions and US government agencies only. For other proposed
# uses, contact tlcip@osu.edu. The software may not be sold or redistributed
# without prior approval.

# You may not use the software to train or process or input the software into
# or make it accessible to: automated software, services or tools, including,
# but not limited to, artificial intelligence solutions, algorithms, machine
# learning, large language models, robots, spiders, crawlers, search engines,
# text or data mining or any other aggregation functionality.

# One may make copies of the software for their use provided that the copies
# are not sold or distributed and are used under the same terms and conditions.
# As unestablished research software, this code is provided on an "as is" basis
# without warranty of any kind, either expressed or implied. The downloading or
# executing any part of this software constitutes an implicit agreement to these
# terms. These terms and conditions are subject to change at any time without
# prior notice.
# =============================================================================

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


