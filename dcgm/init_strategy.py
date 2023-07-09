import abc
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import Dataset, TensorDataset


class DatasetInitStrategy(abc.ABC):
    
    @abc.abstractmethod
    def init(self) -> TensorDataset:
        ...


@dataclass
class RandomStratifiedInitStrategy(DatasetInitStrategy):
    """The dataset is intialized randomly with the property that 
    datapoints between ipc*i to ipc*(i+1) will have the label i"""
    dimensions: Tuple[int, ...]
    num_classes: int
    ipc: int
    device: torch.device
    
    def init(self) -> TensorDataset:
        ''' initialize the synthetic data '''
        syn_data = torch.randn(size=(self.num_classes*self.ipc, *self.dimensions), dtype=torch.float, requires_grad=True, device=self.device)
        syn_labels = torch.arange(0, self.num_classes).repeat_interleave(self.ipc)
        return TensorDataset(syn_data, syn_labels)