# import os
# import abc
# import time
# import math
# import random
# import numpy as np
# import pickle
# import pdb
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.autograd import Variable
# # import pytorch_lightning as pl

# from utils import *
# from utils.transforms import *
# from models.common import *
# from models.uni_transformer import *
# import utils.train as utils_train
# from torch_geometric.nn import radius_graph, knn_graph


# class BaseGVPLayer(nn.Module):
#     def __init__(self, scaler_input_dim, vector_input_dim, output_dim=(64,64), hidden_dim=(128,128), n_heads=2, mess_gvp_layer_num=2, node_gvp_layer_num=2, \
#                  mlp_norm=True, output_norm=True, act_fn='relu'):
#         super().__init__()

#         self.n_heads = n_heads
#         self.output_norm = output_norm

#         hk_input_dim = hidden_dim[0] + hidden_dim[1]
#         hq_input_dim = scaler_input_dim[0] + vector_input_dim[0]

#         self.hk_func = MLP(hk_input_dim, hidden_dim[0], hidden_dim[0] * n_heads, norm=mlp_norm, act_fn=act_fn)
#         self.hq_func = MLP(hq_input_dim, hidden_dim[0], hidden_dim[0] * n_heads, norm=mlp_norm, act_fn=act_fn)
        
#         # initialize message layer to combine scalar features and vector features
#         message_layers = []
#         message_layers.append(
#             GVP([scaler_input_dim[0]+scaler_input_dim[1], vector_input_dim[0]+vector_input_dim[1]], hidden_dim[1], hidden_dim)
#         )
#         for i in range(mess_gvp_layer_num-1):
#             message_layers.append(
#                 GVP(hidden_dim, hidden_dim[1], hidden_dim)
#             )
#         self.message_layer = nn.Sequential(*message_layers)

#         node_output_layer_list = [GVP((hidden_dim[0]+scaler_input_dim[0], hidden_dim[1]+vector_input_dim[0]), hidden_dim[0], hidden_dim)] + \
#                                     [GVP(hidden_dim, hidden_dim[0], hidden_dim) for _ in range(node_gvp_layer_num-2)] + \
#                                     [GVP(hidden_dim, hidden_dim[0], output_dim)]
            
#         if self.output_norm:
#             self.node_norm = GVPLayerNorm(output_dim)

#         self.node_output_layer = nn.Sequential(*node_output_layer_list)

#     def embed_message_att_weight(self, mess_scalar_emb, mess_vec_emb, node_scalar_emb, node_vec_emb, edge_index):
#         mess_vec_emb_norm = norm_no_nan(mess_vec_emb, keepdims=False)
        
#         src, dst = edge_index
#         node_vec_emb_norm = norm_no_nan(node_vec_emb, keepdims=False)
        
#         # compute k
#         k = self.hk_func(torch.cat([mess_scalar_emb, mess_vec_emb_norm], -1))

#         # compute q
#         q = self.hq_func(torch.cat([node_scalar_emb, node_vec_emb_norm], -1))

#         # compute attention weight
#         att_weight = scatter_softmax((q * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)

#         # att_weight = scatter_softmax((k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)

#         return att_weight
        
#     def forward(self, scalar_emb, vec_emb, scalar_edge_feat, vec_edge_feat, edge_index):
#         N = scalar_emb.size(0)
#         src, dst = edge_index

#         scalar_emb_i = scalar_emb[src]
#         vec_emb_i = vec_emb[src]
        
#         # message embedding
#         if scalar_edge_feat is not None:
#             mess_scalar_i_in = torch.cat([scalar_emb_i, scalar_edge_feat], -1)
#         else:
#             mess_scalar_i_in = scalar_emb_i

#         if vec_edge_feat is not None:
#             mess_vec_i_in = torch.cat([vec_emb_i, vec_edge_feat], -2)
#         else:
#             mess_vec_i_in = vec_emb_i
        
#         if torch.isnan(mess_scalar_i_in).any(): pdb.set_trace()
#         if torch.isnan(mess_vec_i_in).any(): pdb.set_trace()
#         mess_scalar_i_out, mess_vec_i_out = self.message_layer((mess_scalar_i_in, mess_vec_i_in))

#         ## attention weight
#         att_weight = self.embed_message_att_weight(mess_scalar_i_out, mess_vec_i_out, scalar_emb_i, vec_emb_i, edge_index)
        
#         ## aggregate messages into atom embeddings
#         w_mess_scalar = att_weight.unsqueeze(-1) * mess_scalar_i_out
#         w_mess_vec = att_weight.view(-1, 1, 1) * mess_vec_i_out

#         scalar_output = scatter_sum(w_mess_scalar, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
#         vec_output = scatter_sum(w_mess_vec, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        
#         scalar_output = torch.cat([scalar_emb, scalar_output], -1)
        
#         vec_output = torch.cat([vec_emb, vec_output], -2)

#         if torch.isnan(scalar_output).any(): pdb.set_trace()
#         if torch.isnan(vec_output).any(): pdb.set_trace()
#         scalar_output, vec_output = self.node_output_layer((scalar_output, vec_output))

#         if self.output_norm:
#             scalar_output, vec_output = self.node_norm((scalar_output, vec_output))
        
#         return scalar_output, vec_output, 
 
# # class BaseModel(pl.LightningModule, abc.ABC):

# #     def __init__(self, config):
# #         super().__init__()
# #         self.config_train = config

# #     def training_step(self, batch, batch_idx):
# #         self.current_batch = batch
# #         loss, type_loss, coord_loss, _, _ = self.get_loss(batch)

# #         return {'loss': loss, 'type_loss': type_loss, 'coord_loss': coord_loss}
    
# #     def training_epoch_end(self, outputs):
# #         result_dict = {}
# #         try:
# #             for key, _ in outputs[0].items():
# #                 result_dict[key] = torch.stack([torch.tensor(x[key]) for x in outputs]).mean()
# #         except:
# #             for key, _ in outputs[0].items():
# #                 result_dict[key] = self.trainer.callback_metrics.get(key)

# #         self.log_dict(result_dict, on_step=False, on_epoch=True)

# #     def configure_optimizers(self):
# #         config = {
# #             "optimizer": self.optimizer
# #         }

# #         config["lr_scheduler"] = {
# #             "scheduler": self.scheduler,
# #             "frequency": 1,
# #             "interval": "epoch",    
# #             'monitor': 'loss',
# #         }

# #         return config
    
# #     @abc.abstractmethod
# #     def get_loss(batch):
# #         pass
    
# #     def validation_step(self, batch, batch_idx):
# #         with torch.no_grad():
# #             loss, type_loss, coord_loss, _, _ = self.get_loss(batch)

# #         return {"val_loss": loss, 'val_type_loss': type_loss, "val_coord_loss": coord_loss}
    
# #     def validation_epoch_end(self, outputs):
# #         try:
# #             avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
# #             avg_type_loss = torch.stack([x['val_type_loss'] for x in outputs]).mean()
# #             avg_coord_loss = torch.stack([x['val_coord_loss'] for x in outputs]).mean()

# #             self.log_dict({"val_loss": avg_loss, "val_type_loss": avg_type_loss, "val_coord_loss": avg_coord_loss}, on_step=False, on_epoch=True)
# #         except:
# #             avg_loss = self.trainer.callback_metrics.get('val_loss')
# #             avg_type_loss = self.trainer.callback_metrics.get('val_type_loss')
# #             avg_coord_loss = self.trainer.callback_metrics.get('val_coord_loss')

# #             # self.log_dict({"val_loss": avg_loss, "val_type_loss": avg_type_loss, "val_coord_loss": avg_coord_loss}, on_step=False, on_epoch=True)

# #         return {'val_loss': avg_loss, 'val_type_loss': avg_type_loss, 'val_coord_loss': avg_coord_loss}

# #     def test_step(self, batch, batch_idx):
# #         with torch.no_grad():
# #             loss, type_loss, coord_loss, acc, diff = self.get_loss(batch)

# #         return {'test_loss': loss, 'test_type_loss': type_loss, 'test_coord_loss': coord_loss, \
# #                  'test_type_acc': acc, 'test_coord_diff': diff}

# #     def test_epoch_end(self, outputs):
# #         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
# #         avg_type_loss = torch.stack([x['test_type_loss'] for x in outputs]).mean()
# #         avg_coord_loss = torch.stack([x['test_coord_loss'] for x in outputs]).mean()
# #         avg_type_acc = torch.stack([x['test_type_acc'] for x in outputs]).mean()
# #         avg_coord_diff = torch.stack([x['test_coord_diff'] for x in outputs]).mean()

# #         print(
# #                 'Loss: %.6f | Type_loss: %.4f | Coord_loss: %.4f | Type_acc: %.2f | Coord_diff: %.4f' % (
# #                     avg_loss, avg_type_loss, avg_coord_loss, avg_type_acc*100, avg_coord_diff
# #                 )
# #             )

# #         return avg_loss


# class Gating(nn.Module):
#     def __init__(self, size):
#         super(Gating, self).__init__()
#         self.gate = nn.Linear(size * 2, size)

#     def forward(self, emb_atom, emb_residue):
#         combined = torch.cat((emb_atom, emb_residue), dim=-1)  
#         gate = torch.sigmoid(self.gate(combined))  

#         return gate
    
# class PocketModel(nn.Module):

#     def __init__(self, config, hydrogen=False, edge_feat_dim=1):
#         super().__init__()

#         self.dim = config.point_dim
#         self.layer_num = config.layer_num
#         self.embedding_dim = config.embedding_dim
#         self.hidden_size = config.hidden_size
#         self.output_size = config.output_size
#         self.k = config.num_k # k-nearest neighbor
#         self.k_residue = config.num_k_residue
#         self.noise = config.noise
#         self.atom_mode = 'add_aromatic'
#         self.residue_center = config.residue_center
#         if hydrogen == True:
#             self.dictionary_dim = len(ATOM_INDEX_DICT)
#         else:
#             self.dictionary_dim = len(ATOM_INDEX_DICT) - 1
            
#         self.loss_type_weight = config.loss_type_weight

#         # Todo: fill in these variables
#         self.padding_idx = None
#         self.element_embedding = nn.Embedding(
#             self.dictionary_dim, self.embedding_dim, self.padding_idx
#         )

#         self.residue_embedding = nn.Embedding(
#             20, self.embedding_dim, self.padding_idx
#         )

#         # Todo: dimension of model needs to be updated based on input features later
#         self.encoder_layers = nn.ModuleList(
#             [BaseGVPLayer(
#                     scaler_input_dim = [self.embedding_dim, edge_feat_dim], vector_input_dim = [1, 1], output_dim = [self.output_size, self.output_size], \
#                     hidden_dim = [self.hidden_size, self.hidden_size]
#                 )] +
#             [BaseGVPLayer(
#                     scaler_input_dim = [self.output_size, 0], vector_input_dim = [self.output_size, 0], output_dim = [self.output_size, self.output_size],\
#                     hidden_dim = [self.hidden_size, self.hidden_size]
#                 ) for i in range(self.layer_num-1)]
#             )
        
#         self.pocket_encoder_layers = nn.ModuleList(
#             [BaseGVPLayer(
#                     scaler_input_dim = [self.embedding_dim, edge_feat_dim], vector_input_dim = [1, 1], output_dim = [self.output_size, self.output_size], \
#                     hidden_dim = [self.hidden_size, self.hidden_size]
#                 )] +
#             [BaseGVPLayer(
#                     scaler_input_dim = [self.output_size, 0], vector_input_dim = [self.output_size, 0], output_dim = [self.output_size, self.output_size],\
#                     hidden_dim = [self.hidden_size, self.hidden_size]
#                 ) for i in range(self.layer_num-1)]
#             )
        
#         # Todo: use the coor_head and type_head to predict coordinates and atom type from embeddings of encoder_layers        
#         # self.coor_head = VNLinearLeakyReLU(in_channels=self.output_size*3+1, out_channels=4, dim=self.dim)
#         self.coor_head = MLP(in_dim=self.output_size*3+1, out_dim=self.dim, hidden_dim=self.output_size, act_last=False)
#         self.type_head = MLP(in_dim=self.output_size, out_dim=self.dictionary_dim-1, hidden_dim=self.output_size, act_last=False)

#         if config.noise_type == "trunc_normal":
#             self.clip = ClipLayer(-2., 2.)
#         elif config.noise_type == "uniform":
#             self.clip = ClipLayer(-1., 1.)
#         elif config.noise_type == "normal":
#             self.clip = nn.Identity()
#         else:
#             raise ValueError(f'Not supported noise type: {config.noise_type}')

#         self.loss_atom_type = nn.CrossEntropyLoss(reduce=False)
#         self.loss_atom_coord = nn.MSELoss(reduce=False)

#         self.scaler_gating = Gating(self.output_size)
#         self.vec_gating = Gating(self.output_size)

#         # self.optimizer = utils_train.get_optimizer(config_train.optimizer, self)
#         # self.scheduler = utils_train.get_scheduler(config_train.scheduler, self.optimizer)

#     def _connect_edge(self, element, coordinate, batch, k, cutoff_mode='knn'):
#         if cutoff_mode == 'knn':
#             edge_index = knn_graph(coordinate, k=k, batch=batch, flow='source_to_target')
#         elif cutoff_mode == 'radius':
#             edge_index = radius_graph(coordinate, r=k, batch=batch, flow='source_to_target')
#             # edge_index = connect_covalent_graph(coordinate, element, atom_mode=self.atom_enc_mode)
#         else:
#             raise ValueError(f'Not supported cutoff mode: {cutoff_mode}')

#         # print(k, coordinate.size(0), edge_index.size(1)/coordinate.size(0))
#         return edge_index

#     def _get_edge_feature(self, element, coordinate, edge_index):
#         """
#         Todo: calculate the distances between connected atoms in edge_index
#         """
#         source_coords = coordinate[edge_index[0,:]]
#         target_coords = coordinate[edge_index[1,:]]
#         distances = torch.sqrt(torch.sum((source_coords - target_coords) ** 2, dim=-1))
#         differences = source_coords - target_coords
        
#         if torch.isnan(distances).any():
#             pdb.set_trace()
#         return distances.unsqueeze(1), (source_coords-target_coords).unsqueeze(1)
    
#     def forward_residue(self, batch, train=True):
#         # if train:
#         #     element = batch.pocket_residue_type
#         #     coordinate = batch.pocket_corrupted_coordinate
#         #     index = batch.pocket_residue_index
#         #     batch_pocket = batch.batch
#         # else:
#         #     element = batch.pocket_residue_type
#         #     coordinate = batch.pocket_coordinate
#         #     index = batch.pocket_residue_index
#         #     batch_pocket = batch.batch
#         # print(batch.mask.sum()/batch.mask.size(0))
#         if train:
#             element = batch.pocket_corrupted_residue_type
#             coordinate = batch.pocket_corrupted_coordinate
#             index = batch.pocket_residue_index
#             batch_pocket = batch.batch
#             # pos_CA = batch.pocket_pos_CA
#             # pos_N = batch.pocket_pos_N
#             # pos_C = batch.pocket_pos_C
#         else:
#             element = batch.pocket_residue_type
#             coordinate = batch.pocket_coordinate
#             index = batch.pocket_residue_index
#             batch_pocket = batch.batch
#             # pos_CA = batch.pocket_pos_CA
#             # pos_N = batch.pocket_pos_N
#             # pos_C = batch.pocket_pos_C

#         # def collapse_residues(batch_pocket, element, coordinate, index):
#         #     clean_residues = []
#         #     clean_positions = []
#         #     clean_batches = []
#         #     lengths = []

#         #     for pocket in batch_pocket.unique():
#         #         pocket_mask = batch_pocket == pocket
#         #         pocket_residues = element[pocket_mask]
#         #         pocket_positions = coordinate[pocket_mask]
#         #         pocket_residue_index = index[pocket_mask]
#         #         pocket_pos_CA = pos_CA[pocket_mask]
#         #         pocket_pos_N = pos_N[pocket_mask]
#         #         pocket_pos_C = pos_C[pocket_mask]

#         #         for residue_i in pocket_residue_index.unique():
#         #             mask = pocket_residue_index == residue_i
#         #             mean_pos = pocket_positions[mask].mean(0)
#         #             residue = pocket_residues[mask][0]
#         #             CA = pocket_pos_CA[mask].mean(0)
#         #             N = pocket_pos_N[mask].mean(0)
#         #             C = pocket_pos_C[mask].mean(0)
#         #             if self.residue_center == 'mean':
#         #                 clean_positions.append(mean_pos)
#         #             elif self.residue_center == 'carbon':
#         #                 clean_positions.append(CA)
#         #             clean_residues.append(residue)
#         #             clean_batches.append(pocket)
#         #             lengths.append(mask.sum())

#         #     return torch.stack(clean_residues), torch.stack(clean_positions), torch.stack(clean_batches), torch.tensor(lengths)

#         # element, coordinate, batch_pocket, lengths = collapse_residues(batch_pocket, element, coordinate, index)

#         element = scatter_mean(element, index, dim=0)
#         coordinate = scatter_mean(coordinate, index, dim=0)
#         batch_pocket = scatter_mean(batch_pocket, index, dim=0)
#         lengths = torch.bincount(index)

#         input_scalar_emb = self.residue_embedding(element)
#         input_vec_emb = coordinate.unsqueeze(1)
#         scalar_emb, vec_emb = input_scalar_emb, input_vec_emb

#         edge_index = self._connect_edge(element, coordinate, batch_pocket, self.k_residue)
#         scaler_edge_feature, vector_edge_feature = self._get_edge_feature(element, coordinate, edge_index)

#         for i, layer in enumerate(self.pocket_encoder_layers):
#             if i == 0:
#                 scalar_emb, vec_emb = layer(scalar_emb, vec_emb, scaler_edge_feature, vector_edge_feature, edge_index)
#             else:
#                 scalar_emb, vec_emb = layer(scalar_emb, vec_emb, None, None, edge_index)
                
#         scalar_emb_full = scalar_emb.repeat_interleave(lengths.to(scalar_emb.device), dim=0)
#         vec_emb_full = vec_emb.repeat_interleave(lengths.to(vec_emb.device), dim=0)

#         # first_occurrences = (index != torch.roll(index, 1)).nonzero().squeeze()
#         # print(first_occurrences)
#         # pocket_residue_scalar_emb = scalar_emb_full
#         # changes = (pocket_residue_scalar_emb[1:] != pocket_residue_scalar_emb[:-1]).any(dim=1)
#         # first_occurrences = torch.cat([torch.tensor([0], device=pocket_residue_scalar_emb.device), 
#         #                         (changes.nonzero().squeeze() + 1)])
#         # print(first_occurrences)
        
#         return scalar_emb_full, vec_emb_full
        
#     def forward_atom(self, batch, train=True):
#         if train:
#             element = batch.pocket_corrupted_element
#             coordinate = batch.pocket_corrupted_coordinate
#             batch_pocket = batch.batch
#         else:
#             element = batch.pocket_element
#             coordinate = batch.pocket_coordinate
#             batch_pocket = batch.batch
    
        
#         input_scalar_emb = self.element_embedding(element)
#         input_vec_emb = coordinate.unsqueeze(1)

#         edge_index = self._connect_edge(element, coordinate, batch_pocket, self.k)
#         scaler_edge_feature, vector_edge_feature = self._get_edge_feature(element, coordinate, edge_index)
        
#         scalar_emb, vec_emb = input_scalar_emb, input_vec_emb
        
#         for i, layer in enumerate(self.encoder_layers):
#             if i == 0:
#                 scalar_emb, vec_emb = layer(scalar_emb, vec_emb, scaler_edge_feature, vector_edge_feature, edge_index)
#             else:
#                 scalar_emb, vec_emb = layer(scalar_emb, vec_emb, None, None, edge_index)

#         return scalar_emb, vec_emb, edge_index
    
#     def forward(self, batch):
#         scalar_emb_atom, vec_emb_atom, edge_index =  self.forward_atom(batch)
#         scalar_emb_residue, vec_emb_residue =  self.forward_residue(batch)

#         scaler_gate = self.scaler_gating(scalar_emb_atom, scalar_emb_residue)
#         vec_gate = self.vec_gating(norm_no_nan(vec_emb_atom), norm_no_nan(vec_emb_residue)).unsqueeze(-1)

#         # TODO: calculate weight
#         scalar_emb = scaler_gate * scalar_emb_atom + (1 - scaler_gate) * scalar_emb_residue
#         vec_emb = vec_gate * vec_emb_atom + (1 - vec_gate) * vec_emb_residue

#         pred_type = self.type_head(scalar_emb)
#         pred_coor = self.get_pred_coor(batch.pocket_corrupted_coordinate.unsqueeze(1), scalar_emb, vec_emb, edge_index)

#         return pred_type, pred_coor

#     def get_pred_coor(self, input_coor, scalar_emb, vec_emb, edge_index):
#         N = scalar_emb.size(0)
#         src, dst = edge_index
        
#         vec_emb_diff = norm_no_nan(vec_emb[dst] - vec_emb[src])
#         input_diff = input_coor[dst] - input_coor[src]
        
#         # input_weight = torch.cat([scalar_emb[dst], scalar_emb[src], norm_no_nan(input_diff), vec_emb_diff], axis=1)
#         weight = self.coor_head(torch.cat([scalar_emb[dst], scalar_emb[src], norm_no_nan(input_diff), vec_emb_diff], axis=1))
        
#         weighted_diff = input_diff.squeeze(1) * weight
#         output_diff = scatter_mean(weighted_diff, dst, dim=0, dim_size=N) # (N, heads, H_per_head)
#         return input_coor.squeeze(1) + self.noise*output_diff 

#     def get_embds(self, batch):

#         scalar_emb_atom, vec_emb_atom, edge_index =  self.forward_atom(batch, train=False)
#         scalar_emb_residue, vec_emb_residue =  self.forward_residue(batch, train=False)

#         scaler_gate = self.scaler_gating(scalar_emb_atom, scalar_emb_residue)
#         vec_gate = self.vec_gating(norm_no_nan(vec_emb_atom), norm_no_nan(vec_emb_residue)).unsqueeze(-1)

#         scalar_emb = scaler_gate * scalar_emb_atom + (1 - scaler_gate) * scalar_emb_residue
#         vec_emb = vec_gate * vec_emb_atom + (1 - vec_gate) * vec_emb_residue

#         # changes = (scalar_emb_residue[1:] != scalar_emb_residue[:-1]).any(dim=1)
#         # first_occurrences = torch.cat([torch.tensor([0], device=scalar_emb_residue.device), 
#         #                         (changes.nonzero().squeeze() + 1)])
#         # print(first_occurrences)
#         # first_occurrences = (batch.pocket_residue_index != torch.roll(batch.pocket_residue_index, 1)).nonzero().squeeze()
#         # print(first_occurrences)

#         return scalar_emb, vec_emb
    
#     def get_embds_atom(self, batch):

#         scalar_emb, vec_emb, _ =  self.forward_atom(batch, train=False)

#         return scalar_emb, vec_emb

#     def get_loss(self, batch):
#         # get masked atom type and their corresponding predictions
#         # similar for coordinates
#         # use cross-entropy loss and MSE loss for these predictions, respectively.
        
#         pred_type, pred_coord = self(
#             batch
#         )
#         # pred_coord = batch.pocket_corrupted_coordinate - pred_noise*self.noise

#         # pred_type = torch.where(mask.unsqueeze(1).expand(-1, self.dictionary_dim-1), pred_type, 1e5*F.one_hot(pocket_element, num_classes=self.dictionary_dim-1).to(pred_type.dtype))
#         # pred_coord = torch.where(batch.mask.unsqueeze(1).expand(-1, self.dim), pred_coord, batch.pocket_coordinate)

#         type_loss = self.loss_atom_type(pred_type, batch.pocket_element)
#         coord_loss = self.loss_atom_coord(pred_coord, batch.pocket_coordinate)

#         type_loss = torch.where(batch.mask, type_loss, torch.zeros_like(type_loss))
#         coord_loss = torch.where(batch.mask.unsqueeze(1).expand(-1, self.dim), coord_loss, torch.zeros_like(coord_loss))

#         batch_size = batch.batch.max() + 1
#         batch.batch[batch.mask==False] = batch_size

#         type_loss = scatter_mean(type_loss, batch.batch, dim=0)
#         coord_loss = scatter_mean(coord_loss.mean(-1), batch.batch, dim=0)

#         type_loss = type_loss[:-1].mean(0)
#         coord_loss = coord_loss[:-1].mean(0)

#         loss = self.loss_type_weight * type_loss + coord_loss

#         pred_index = pred_type.argmax(dim=1)
#         acc = (pred_index[batch.mask] == batch.pocket_element[batch.mask]).sum().item()/batch.mask.sum()
#         diff = (pred_coord[batch.mask]-batch.pocket_coordinate[batch.mask]).abs().mean()
        
#         return loss, type_loss, coord_loss, acc, diff





import os
import abc
import time
import math
import random
import numpy as np
import pickle
import pdb
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
# import pytorch_lightning as pl

from utils import *
from utils.transforms import *
from models.common import *
from models.uni_transformer import *
import utils.train as utils_train
from torch_geometric.nn import radius_graph, knn_graph


class BaseGVPLayer(nn.Module):
    def __init__(self, scaler_input_dim, vector_input_dim, output_dim=(64,64), hidden_dim=(128,128), n_heads=2, mess_gvp_layer_num=2, node_gvp_layer_num=2, \
                 mlp_norm=True, output_norm=True, act_fn='relu'):
        super().__init__()

        self.n_heads = n_heads
        self.output_norm = output_norm

        hk_input_dim = hidden_dim[0] + hidden_dim[1]
        hq_input_dim = scaler_input_dim[0] + vector_input_dim[0]

        self.hk_func = MLP(hk_input_dim, hidden_dim[0], hidden_dim[0] * n_heads, norm=mlp_norm, act_fn=act_fn)
        self.hq_func = MLP(hq_input_dim, hidden_dim[0], hidden_dim[0] * n_heads, norm=mlp_norm, act_fn=act_fn)
        
        # initialize message layer to combine scalar features and vector features
        message_layers = []
        message_layers.append(
            GVP([scaler_input_dim[0]+scaler_input_dim[1], vector_input_dim[0]+vector_input_dim[1]], hidden_dim[1], hidden_dim)
        )
        for i in range(mess_gvp_layer_num-1):
            message_layers.append(
                GVP(hidden_dim, hidden_dim[1], hidden_dim)
            )
        self.message_layer = nn.Sequential(*message_layers)

        node_output_layer_list = [GVP((hidden_dim[0]+scaler_input_dim[0], hidden_dim[1]+vector_input_dim[0]), hidden_dim[0], hidden_dim)] + \
                                    [GVP(hidden_dim, hidden_dim[0], hidden_dim) for _ in range(node_gvp_layer_num-2)] + \
                                    [GVP(hidden_dim, hidden_dim[0], output_dim)]
            
        if self.output_norm:
            self.node_norm = GVPLayerNorm(output_dim)

        self.node_output_layer = nn.Sequential(*node_output_layer_list)

    def embed_message_att_weight(self, mess_scalar_emb, mess_vec_emb, node_scalar_emb, node_vec_emb, edge_index):
        mess_vec_emb_norm = norm_no_nan(mess_vec_emb, keepdims=False)
        
        src, dst = edge_index
        node_vec_emb_norm = norm_no_nan(node_vec_emb, keepdims=False)
        
        # compute k
        k = self.hk_func(torch.cat([mess_scalar_emb, mess_vec_emb_norm], -1))

        # compute q
        q = self.hq_func(torch.cat([node_scalar_emb, node_vec_emb_norm], -1))

        # compute attention weight
        att_weight = scatter_softmax((q * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)

        # att_weight = scatter_softmax((k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0)

        return att_weight
        
    def forward(self, scalar_emb, vec_emb, scalar_edge_feat, vec_edge_feat, edge_index):
        N = scalar_emb.size(0)
        src, dst = edge_index

        scalar_emb_i = scalar_emb[src]
        vec_emb_i = vec_emb[src]
        
        # message embedding
        if scalar_edge_feat is not None:
            mess_scalar_i_in = torch.cat([scalar_emb_i, scalar_edge_feat], -1)
        else:
            mess_scalar_i_in = scalar_emb_i

        if vec_edge_feat is not None:
            mess_vec_i_in = torch.cat([vec_emb_i, vec_edge_feat], -2)
        else:
            mess_vec_i_in = vec_emb_i
        
        if torch.isnan(mess_scalar_i_in).any(): pdb.set_trace()
        if torch.isnan(mess_vec_i_in).any(): pdb.set_trace()
        mess_scalar_i_out, mess_vec_i_out = self.message_layer((mess_scalar_i_in, mess_vec_i_in))

        ## attention weight
        att_weight = self.embed_message_att_weight(mess_scalar_i_out, mess_vec_i_out, scalar_emb_i, vec_emb_i, edge_index)
        
        ## aggregate messages into atom embeddings
        w_mess_scalar = att_weight.unsqueeze(-1) * mess_scalar_i_out
        w_mess_vec = att_weight.view(-1, 1, 1) * mess_vec_i_out

        scalar_output = scatter_sum(w_mess_scalar, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        vec_output = scatter_sum(w_mess_vec, dst, dim=0, dim_size=N)  # (N, heads, H_per_head)
        
        scalar_output = torch.cat([scalar_emb, scalar_output], -1)
        
        vec_output = torch.cat([vec_emb, vec_output], -2)

        if torch.isnan(scalar_output).any(): pdb.set_trace()
        if torch.isnan(vec_output).any(): pdb.set_trace()
        scalar_output, vec_output = self.node_output_layer((scalar_output, vec_output))

        if self.output_norm:
            scalar_output, vec_output = self.node_norm((scalar_output, vec_output))
        
        return scalar_output, vec_output, 
 
# class BaseModel(pl.LightningModule, abc.ABC):

#     def __init__(self, config):
#         super().__init__()
#         self.config_train = config

#     def training_step(self, batch, batch_idx):
#         self.current_batch = batch
#         loss, type_loss, coord_loss, _, _ = self.get_loss(batch)

#         return {'loss': loss, 'type_loss': type_loss, 'coord_loss': coord_loss}
    
#     def training_epoch_end(self, outputs):
#         result_dict = {}
#         try:
#             for key, _ in outputs[0].items():
#                 result_dict[key] = torch.stack([torch.tensor(x[key]) for x in outputs]).mean()
#         except:
#             for key, _ in outputs[0].items():
#                 result_dict[key] = self.trainer.callback_metrics.get(key)

#         self.log_dict(result_dict, on_step=False, on_epoch=True)

#     def configure_optimizers(self):
#         config = {
#             "optimizer": self.optimizer
#         }

#         config["lr_scheduler"] = {
#             "scheduler": self.scheduler,
#             "frequency": 1,
#             "interval": "epoch",    
#             'monitor': 'loss',
#         }

#         return config
    
#     @abc.abstractmethod
#     def get_loss(batch):
#         pass
    
#     def validation_step(self, batch, batch_idx):
#         with torch.no_grad():
#             loss, type_loss, coord_loss, _, _ = self.get_loss(batch)

#         return {"val_loss": loss, 'val_type_loss': type_loss, "val_coord_loss": coord_loss}
    
#     def validation_epoch_end(self, outputs):
#         try:
#             avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#             avg_type_loss = torch.stack([x['val_type_loss'] for x in outputs]).mean()
#             avg_coord_loss = torch.stack([x['val_coord_loss'] for x in outputs]).mean()

#             self.log_dict({"val_loss": avg_loss, "val_type_loss": avg_type_loss, "val_coord_loss": avg_coord_loss}, on_step=False, on_epoch=True)
#         except:
#             avg_loss = self.trainer.callback_metrics.get('val_loss')
#             avg_type_loss = self.trainer.callback_metrics.get('val_type_loss')
#             avg_coord_loss = self.trainer.callback_metrics.get('val_coord_loss')

#             # self.log_dict({"val_loss": avg_loss, "val_type_loss": avg_type_loss, "val_coord_loss": avg_coord_loss}, on_step=False, on_epoch=True)

#         return {'val_loss': avg_loss, 'val_type_loss': avg_type_loss, 'val_coord_loss': avg_coord_loss}

#     def test_step(self, batch, batch_idx):
#         with torch.no_grad():
#             loss, type_loss, coord_loss, acc, diff = self.get_loss(batch)

#         return {'test_loss': loss, 'test_type_loss': type_loss, 'test_coord_loss': coord_loss, \
#                  'test_type_acc': acc, 'test_coord_diff': diff}

#     def test_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         avg_type_loss = torch.stack([x['test_type_loss'] for x in outputs]).mean()
#         avg_coord_loss = torch.stack([x['test_coord_loss'] for x in outputs]).mean()
#         avg_type_acc = torch.stack([x['test_type_acc'] for x in outputs]).mean()
#         avg_coord_diff = torch.stack([x['test_coord_diff'] for x in outputs]).mean()

#         print(
#                 'Loss: %.6f | Type_loss: %.4f | Coord_loss: %.4f | Type_acc: %.2f | Coord_diff: %.4f' % (
#                     avg_loss, avg_type_loss, avg_coord_loss, avg_type_acc*100, avg_coord_diff
#                 )
#             )

#         return avg_loss


class Gating(nn.Module):
    def __init__(self, size):
        super(Gating, self).__init__()
        self.gate = nn.Linear(size * 2, size)

    def forward(self, emb_atom, emb_residue):
        combined = torch.cat((emb_atom, emb_residue), dim=-1)  
        gate = torch.sigmoid(self.gate(combined))  

        return gate
    
class PocketModel(nn.Module):

    def __init__(self, config, hydrogen=False, edge_feat_dim=1):
        super().__init__()

        self.dim = config.point_dim
        self.layer_num = config.layer_num
        self.embedding_dim = config.embedding_dim
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.k = config.num_k # k-nearest neighbor
        self.k_residue = config.num_k_residue
        self.noise = config.noise
        self.atom_mode = 'add_aromatic'
        self.residue_center = config.residue_center
        if hydrogen == True:
            self.dictionary_dim = len(ATOM_INDEX_DICT)
        else:
            self.dictionary_dim = len(ATOM_INDEX_DICT) - 1
            
        self.loss_type_weight = config.loss_type_weight

        # Todo: fill in these variables
        self.padding_idx = None
        self.element_embedding = nn.Embedding(
            self.dictionary_dim, self.embedding_dim, self.padding_idx
        )

        self.residue_embedding = nn.Embedding(
            20, self.embedding_dim, self.padding_idx
        )

        # Todo: dimension of model needs to be updated based on input features later
        self.encoder_layers = nn.ModuleList(
            [BaseGVPLayer(
                    scaler_input_dim = [self.embedding_dim, edge_feat_dim], vector_input_dim = [1, 1], output_dim = [self.output_size, self.output_size], \
                    hidden_dim = [self.hidden_size, self.hidden_size]
                )] +
            [BaseGVPLayer(
                    scaler_input_dim = [self.output_size, 0], vector_input_dim = [self.output_size, 0], output_dim = [self.output_size, self.output_size],\
                    hidden_dim = [self.hidden_size, self.hidden_size]
                ) for i in range(self.layer_num-1)]
            )
        
        self.pocket_encoder_layers = nn.ModuleList(
            [BaseGVPLayer(
                    scaler_input_dim = [self.embedding_dim, edge_feat_dim], vector_input_dim = [1, 1], output_dim = [self.output_size, self.output_size], \
                    hidden_dim = [self.hidden_size, self.hidden_size]
                )] +
            [BaseGVPLayer(
                    scaler_input_dim = [self.output_size, 0], vector_input_dim = [self.output_size, 0], output_dim = [self.output_size, self.output_size],\
                    hidden_dim = [self.hidden_size, self.hidden_size]
                ) for i in range(self.layer_num-1)]
            )
        
        # Todo: use the coor_head and type_head to predict coordinates and atom type from embeddings of encoder_layers        
        # self.coor_head = VNLinearLeakyReLU(in_channels=self.output_size*3+1, out_channels=4, dim=self.dim)
        self.coor_head = MLP(in_dim=self.output_size*3+1, out_dim=self.dim, hidden_dim=self.output_size, act_last=False)
        self.type_head = MLP(in_dim=self.output_size, out_dim=self.dictionary_dim-1, hidden_dim=self.output_size, act_last=False)

        if config.noise_type == "trunc_normal":
            self.clip = ClipLayer(-2., 2.)
        elif config.noise_type == "uniform":
            self.clip = ClipLayer(-1., 1.)
        elif config.noise_type == "normal":
            self.clip = nn.Identity()
        else:
            raise ValueError(f'Not supported noise type: {config.noise_type}')

        self.loss_atom_type = nn.CrossEntropyLoss(reduce=False)
        self.loss_atom_coord = nn.MSELoss(reduce=False)

        self.scaler_gating = Gating(self.output_size)
        self.vec_gating = Gating(self.output_size)

        # self.optimizer = utils_train.get_optimizer(config_train.optimizer, self)
        # self.scheduler = utils_train.get_scheduler(config_train.scheduler, self.optimizer)

    def _connect_edge(self, element, coordinate, batch, k, cutoff_mode='knn'):
        if cutoff_mode == 'knn':
            edge_index = knn_graph(coordinate, k=k, batch=batch, flow='source_to_target')
        elif cutoff_mode == 'radius':
            edge_index = radius_graph(coordinate, r=k, batch=batch, flow='source_to_target')
            # edge_index = connect_covalent_graph(coordinate, element, atom_mode=self.atom_enc_mode)
        else:
            raise ValueError(f'Not supported cutoff mode: {cutoff_mode}')

        # print(k, coordinate.size(0), edge_index.size(1)/coordinate.size(0))
        return edge_index

    def _get_edge_feature(self, element, coordinate, edge_index):
        """
        Todo: calculate the distances between connected atoms in edge_index
        """
        source_coords = coordinate[edge_index[0,:]]
        target_coords = coordinate[edge_index[1,:]]
        distances = torch.sqrt(torch.sum((source_coords - target_coords) ** 2, dim=-1))
        differences = source_coords - target_coords
        
        if torch.isnan(distances).any():
            pdb.set_trace()
        return distances.unsqueeze(1), (source_coords-target_coords).unsqueeze(1)
    
    def forward_residue(self, batch, train=True):
        # if train:
        #     element = batch.pocket_residue_type
        #     coordinate = batch.pocket_corrupted_coordinate
        #     index = batch.pocket_residue_index
        #     batch_pocket = batch.batch
        # else:
        #     element = batch.pocket_residue_type
        #     coordinate = batch.pocket_coordinate
        #     index = batch.pocket_residue_index
        #     batch_pocket = batch.batch
        # print(batch.mask.sum()/batch.mask.size(0))
        if train:
            element = batch.pocket_corrupted_residue_type
            coordinate = batch.pocket_corrupted_coordinate
            index = batch.pocket_residue_index
            batch_pocket = batch.batch
            # pos_CA = batch.pocket_pos_CA
            # pos_N = batch.pocket_pos_N
            # pos_C = batch.pocket_pos_C
        else:
            element = batch.pocket_residue_type
            coordinate = batch.pocket_coordinate
            index = batch.pocket_residue_index
            batch_pocket = batch.batch
            # pos_CA = batch.pocket_pos_CA
            # pos_N = batch.pocket_pos_N
            # pos_C = batch.pocket_pos_C

        # def collapse_residues(batch_pocket, element, coordinate, index):
        #     clean_residues = []
        #     clean_positions = []
        #     clean_batches = []
        #     lengths = []

        #     for pocket in batch_pocket.unique():
        #         pocket_mask = batch_pocket == pocket
        #         pocket_residues = element[pocket_mask]
        #         pocket_positions = coordinate[pocket_mask]
        #         pocket_residue_index = index[pocket_mask]
        #         pocket_pos_CA = pos_CA[pocket_mask]
        #         pocket_pos_N = pos_N[pocket_mask]
        #         pocket_pos_C = pos_C[pocket_mask]

        #         for residue_i in pocket_residue_index.unique():
        #             mask = pocket_residue_index == residue_i
        #             mean_pos = pocket_positions[mask].mean(0)
        #             residue = pocket_residues[mask][0]
        #             CA = pocket_pos_CA[mask].mean(0)
        #             N = pocket_pos_N[mask].mean(0)
        #             C = pocket_pos_C[mask].mean(0)
        #             if self.residue_center == 'mean':
        #                 clean_positions.append(mean_pos)
        #             elif self.residue_center == 'carbon':
        #                 clean_positions.append(CA)
        #             clean_residues.append(residue)
        #             clean_batches.append(pocket)
        #             lengths.append(mask.sum())

        #     return torch.stack(clean_residues), torch.stack(clean_positions), torch.stack(clean_batches), torch.tensor(lengths)

        # element, coordinate, batch_pocket, lengths = collapse_residues(batch_pocket, element, coordinate, index)

        element = scatter_mean(element, index, dim=0)
        coordinate = scatter_mean(coordinate, index, dim=0)
        batch_pocket = scatter_mean(batch_pocket, index, dim=0)
        lengths = torch.bincount(index)

        input_scalar_emb = self.residue_embedding(element)
        input_vec_emb = coordinate.unsqueeze(1)
        scalar_emb, vec_emb = input_scalar_emb, input_vec_emb

        edge_index = self._connect_edge(element, coordinate, batch_pocket, self.k_residue)
        scaler_edge_feature, vector_edge_feature = self._get_edge_feature(element, coordinate, edge_index)

        for i, layer in enumerate(self.pocket_encoder_layers):
            if i == 0:
                scalar_emb, vec_emb = layer(scalar_emb, vec_emb, scaler_edge_feature, vector_edge_feature, edge_index)
            else:
                scalar_emb, vec_emb = layer(scalar_emb, vec_emb, None, None, edge_index)
                
        scalar_emb_full = scalar_emb.repeat_interleave(lengths.to(scalar_emb.device), dim=0)
        vec_emb_full = vec_emb.repeat_interleave(lengths.to(vec_emb.device), dim=0)

        # first_occurrences = (index != torch.roll(index, 1)).nonzero().squeeze()
        # print(first_occurrences)
        # pocket_residue_scalar_emb = scalar_emb_full
        # changes = (pocket_residue_scalar_emb[1:] != pocket_residue_scalar_emb[:-1]).any(dim=1)
        # first_occurrences = torch.cat([torch.tensor([0], device=pocket_residue_scalar_emb.device), 
        #                         (changes.nonzero().squeeze() + 1)])
        # print(first_occurrences)
        
        return scalar_emb_full, vec_emb_full
        
    def forward_atom(self, batch, train=True):
        if train:
            element = batch.pocket_corrupted_element
            coordinate = batch.pocket_corrupted_coordinate
            batch_pocket = batch.batch
        else:
            element = batch.pocket_element
            coordinate = batch.pocket_coordinate
            batch_pocket = batch.batch
    
        
        input_scalar_emb = self.element_embedding(element)
        input_vec_emb = coordinate.unsqueeze(1)

        edge_index = self._connect_edge(element, coordinate, batch_pocket, self.k)
        scaler_edge_feature, vector_edge_feature = self._get_edge_feature(element, coordinate, edge_index)
        
        scalar_emb, vec_emb = input_scalar_emb, input_vec_emb
        
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                scalar_emb, vec_emb = layer(scalar_emb, vec_emb, scaler_edge_feature, vector_edge_feature, edge_index)
            else:
                scalar_emb, vec_emb = layer(scalar_emb, vec_emb, None, None, edge_index)

        return scalar_emb, vec_emb, edge_index
    
    def forward(self, batch):
        scalar_emb_atom, vec_emb_atom, edge_index =  self.forward_atom(batch)
        scalar_emb_residue, vec_emb_residue =  self.forward_residue(batch)

        scaler_gate = self.scaler_gating(scalar_emb_atom, scalar_emb_residue)
        vec_gate = self.vec_gating(norm_no_nan(vec_emb_atom), norm_no_nan(vec_emb_residue)).unsqueeze(-1)

        # TODO: calculate weight
        scalar_emb = scaler_gate * scalar_emb_atom + (1 - scaler_gate) * scalar_emb_residue
        vec_emb = vec_gate * vec_emb_atom + (1 - vec_gate) * vec_emb_residue

        pred_type = self.type_head(scalar_emb)
        pred_coor = self.get_pred_coor(batch.pocket_corrupted_coordinate.unsqueeze(1), scalar_emb, vec_emb, edge_index)

        return pred_type, pred_coor

    def get_pred_coor(self, input_coor, scalar_emb, vec_emb, edge_index):
        N = scalar_emb.size(0)
        src, dst = edge_index
        
        vec_emb_diff = norm_no_nan(vec_emb[dst] - vec_emb[src])
        input_diff = input_coor[dst] - input_coor[src]
        
        # input_weight = torch.cat([scalar_emb[dst], scalar_emb[src], norm_no_nan(input_diff), vec_emb_diff], axis=1)
        weight = self.coor_head(torch.cat([scalar_emb[dst], scalar_emb[src], norm_no_nan(input_diff), vec_emb_diff], axis=1))
        
        weighted_diff = input_diff.squeeze(1) * weight
        output_diff = scatter_mean(weighted_diff, dst, dim=0, dim_size=N) # (N, heads, H_per_head)
        return input_coor.squeeze(1) + self.noise*output_diff 

    def get_embds(self, batch):

        scalar_emb_atom, vec_emb_atom, edge_index =  self.forward_atom(batch, train=False)
        scalar_emb_residue, vec_emb_residue =  self.forward_residue(batch, train=False)

        scaler_gate = self.scaler_gating(scalar_emb_atom, scalar_emb_residue)
        vec_gate = self.vec_gating(norm_no_nan(vec_emb_atom), norm_no_nan(vec_emb_residue)).unsqueeze(-1)

        scalar_emb = scaler_gate * scalar_emb_atom + (1 - scaler_gate) * scalar_emb_residue
        vec_emb = vec_gate * vec_emb_atom + (1 - vec_gate) * vec_emb_residue

        # changes = (scalar_emb_residue[1:] != scalar_emb_residue[:-1]).any(dim=1)
        # first_occurrences = torch.cat([torch.tensor([0], device=scalar_emb_residue.device), 
        #                         (changes.nonzero().squeeze() + 1)])
        # print(first_occurrences)
        # first_occurrences = (batch.pocket_residue_index != torch.roll(batch.pocket_residue_index, 1)).nonzero().squeeze()
        # print(first_occurrences)

        return scalar_emb, vec_emb
    
    def get_embds_atom(self, batch):

        scalar_emb, vec_emb, _ =  self.forward_atom(batch, train=False)

        return scalar_emb, vec_emb

    def get_loss(self, batch):
        # get masked atom type and their corresponding predictions
        # similar for coordinates
        # use cross-entropy loss and MSE loss for these predictions, respectively.
        
        pred_type, pred_coord = self(
            batch
        )
        # pred_coord = batch.pocket_corrupted_coordinate - pred_noise*self.noise

        # pred_type = torch.where(mask.unsqueeze(1).expand(-1, self.dictionary_dim-1), pred_type, 1e5*F.one_hot(pocket_element, num_classes=self.dictionary_dim-1).to(pred_type.dtype))
        # pred_coord = torch.where(batch.mask.unsqueeze(1).expand(-1, self.dim), pred_coord, batch.pocket_coordinate)

        type_loss = self.loss_atom_type(pred_type, batch.pocket_element)
        coord_loss = self.loss_atom_coord(pred_coord, batch.pocket_coordinate)

        type_loss = torch.where(batch.mask, type_loss, torch.zeros_like(type_loss))
        coord_loss = torch.where(batch.mask.unsqueeze(1).expand(-1, self.dim), coord_loss, torch.zeros_like(coord_loss))

        batch_size = batch.batch.max() + 1
        batch.batch[batch.mask==False] = batch_size

        type_loss = scatter_mean(type_loss, batch.batch, dim=0)
        coord_loss = scatter_mean(coord_loss.mean(-1), batch.batch, dim=0)

        type_loss = type_loss[:-1].mean(0)
        coord_loss = coord_loss[:-1].mean(0)

        loss = self.loss_type_weight * type_loss + coord_loss

        pred_index = pred_type.argmax(dim=1)
        acc = (pred_index[batch.mask] == batch.pocket_element[batch.mask]).sum().item()/batch.mask.sum()
        diff = (pred_coord[batch.mask]-batch.pocket_coordinate[batch.mask]).abs().mean()
        
        return loss, type_loss, coord_loss, acc, diff