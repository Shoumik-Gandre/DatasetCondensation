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
