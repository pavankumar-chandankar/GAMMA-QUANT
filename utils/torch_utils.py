# ------------------------------------------------------------------------
# Torch utilities
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn

from utils.data_utils import apply_sliding_window, apply_context_sliding_window


class InertialDataset(Dataset):
    """
    Inertial dataset for time series classification.
    
    Args:
        data: numpy array
            Input data
        window_size: int
            Size of the sliding window
        window_overlap: int
            Overlap of the sliding window
        model: str
            Model type (default is 'deepconvlstm')
    """
    def __init__(self, data, window_size, window_overlap, model='deepconvlstm'):
        self.ids, self.features, self.labels = apply_sliding_window(data, window_size, window_overlap)
        self.classes = len(np.unique(self.labels))
        self.channels = self.features.shape[2] - 1
        self.window_size = window_size
        self.model = model

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if 'attendanddiscriminate' in self.model:
            data = torch.FloatTensor(self.features[index, :, 1:])
            target = torch.LongTensor([int(self.labels[index])])
            return data, target
        else:
            return self.features[index, :, 1:].astype(np.float32), self.labels[index].astype(np.uint8)
        

class ContextDataset(Dataset):
    def __init__(self, data, no_context_windows, window_size, window_overlap, model='deepconvlstm', dataset_path=None):
        ids, features, labels = apply_context_sliding_window(data, no_context_windows, window_size, window_overlap, dataset_path)
        self.ids = ids
        self.features = features
        self.labels = labels
        self.classes = len(np.unique(self.labels))
        self.channels = self.features.shape[2] - 1
        self.window_size = window_size
        self.no_context_windows = no_context_windows
        self.model = model
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index, :, 1:].astype(np.float32), self.labels[index].astype(np.uint8)
    

def init_weights(network, weight_init):
    """
    Weight initialization of network (initialises all LSTM, Conv2D and Linear layers according to weight_init parameter
    of network.

    Args:
        network: torch.nn.Module
            The network to initialize.
        weight_init: str
            The weight initialization method. Options are 'normal', 'orthogonal', 'xavier_uniform', 'xavier_normal',
            'kaiming_uniform', 'kaiming_normal'.
            
    Returns:
        network: torch.nn.Module
            The initialized network.
    """
    for m in network.modules():
        # conv initialisation
        if isinstance(m, nn.Conv2d):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                m.bias.data.fill_(0.0)
        # linear layers
        elif isinstance(m, nn.Linear):
            if weight_init == 'normal':
                nn.init.normal_(m.weight)
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight)
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            if torch.is_tensor(m.bias):                
                nn.init.constant_(m.bias, 0)
        # LSTM initialisation
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    if weight_init == 'normal':
                        torch.nn.init.normal_(param.data)
                    elif weight_init == 'orthogonal':
                        torch.nn.init.orthogonal_(param.data)
                    elif weight_init == 'xavier_uniform':
                        torch.nn.init.xavier_uniform_(param.data)
                    elif weight_init == 'xavier_normal':
                        torch.nn.init.xavier_normal_(param.data)
                    elif weight_init == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param.data)
                    elif weight_init == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param.data)
        elif isinstance(m, nn.LayerNorm):
            # Typically, the scale (weight) is initialized to 1 and the bias to 0.
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        # Transformer-related: MultiheadAttention and TransformerEncoderLayer
        elif isinstance(m, nn.MultiheadAttention):
            for attr in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                param = m
                for part in attr.split('.'):
                    param = getattr(param, part)
                if 'weight' in attr:
                    if weight_init == 'normal':
                        nn.init.normal_(param)
                    elif weight_init == 'orthogonal':
                        nn.init.orthogonal_(param)
                    elif weight_init == 'xavier_uniform':
                        nn.init.xavier_uniform_(param)
                    elif weight_init == 'xavier_normal':
                        nn.init.xavier_normal_(param)
                    elif weight_init == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(param)
                    elif weight_init == 'kaiming_normal':
                        nn.init.kaiming_normal_(param)
                else:
                    nn.init.constant_(param, 0)

        elif isinstance(m, nn.TransformerEncoderLayer):
            # Applies to self_attn, linear1, linear2
            init_weights(m.self_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)

        elif isinstance(m, nn.TransformerDecoderLayer):
            init_weights(m.self_attn, weight_init)
            init_weights(m.multihead_attn, weight_init)
            init_weights(m.linear1, weight_init)
            init_weights(m.linear2, weight_init)
            init_weights(m.norm1, weight_init)
            init_weights(m.norm2, weight_init)
            init_weights(m.norm3, weight_init)
    return network


def fix_random_seed(seed, include_cuda=True):
    """
    Fix random seed for reproducibility.
    
    Args:
        seed: int
            Random seed to fix.
        include_cuda: bool
            Whether to include CUDA in the random seed fixing.

    Returns:
        rng_generator: torch.Generator
            Random number generator.
    """
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def save_checkpoint(state, is_best, file_folder, file_name='checkpoint.pth.tar'):
    """
    Save model checkpoint to file
    
    Args:
        state: dict
            State dictionary containing model and optimizer state.
        is_best: bool
            Whether this is the best model so far.
        file_folder: str
            Folder to save the checkpoint.
        file_name: str
            Name of the checkpoint file.    
    """
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)