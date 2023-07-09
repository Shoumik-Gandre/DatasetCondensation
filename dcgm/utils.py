import random
from typing import Iterable, Tuple, Sequence
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, ConcatDataset


def has_batchnormalization(model: nn.Module) -> bool:
    for module in model.modules():
        if 'BatchNorm' in module._get_name():
            return True
    return False


def fix_batchnormalization_statistics(model: nn.Module) -> None:
    """Note: This function searches for batchnorm statistics and sets them to eval. 
    A simple \"model.train()\" on this model will unfix them again"""
    for module in model.modules():
        if 'BatchNorm' in module._get_name():
            module.eval()


def update_batchnorm_statistics(
        model: nn.Module, 
        labelwise_dataset: Tuple[Subset], 
        batchsize_perlabel: int) -> None:
    """Given a tuple of labelwise subsets of the dataset, 
    this function will update the model with a random subset of size num_labels*batchsize_perlabel"""

    batch_norm_dataset = []
    # Obtain labelwise random subsets from real data
    for subset in labelwise_dataset:
        random_labelwise_subset = Subset(
            dataset=subset, 
            indices=random.sample(
                range(len(subset)), 
                batchsize_perlabel
            )
        )
        batch_norm_dataset.append(random_labelwise_subset)
    
    batch_norm_dataset = ConcatDataset(batch_norm_dataset)
    # Load all of the data at once into the memory using dataloader
    dataloader = DataLoader(batch_norm_dataset, batch_size=len(batch_norm_dataset), num_workers=2)
    # Activate batchnormalization
    model.train()
    inputs, _ = next(iter(dataloader))
    model(inputs)  # Batchnormalization statistics updated


def distance_wb(gwr: torch.Tensor, gws: torch.Tensor):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(
        gw_syn: Sequence[torch.Tensor], 
        gw_real: Sequence[torch.Tensor], 
        distance_metric: str, 
        device: torch.device
    ):
    dis = torch.tensor(0.0).to(device)

    if distance_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif distance_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif distance_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%distance_metric)

    return dis
