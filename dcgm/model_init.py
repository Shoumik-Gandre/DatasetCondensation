import abc
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset


class ModelInitStrategy(abc.ABC):
    
    @abc.abstractmethod
    def init(self) -> nn.Module:
        ...


@dataclass
class Lenet5InitStrategy(ModelInitStrategy):
    """The dataset is intialized randomly with the property that 
    datapoints between ipc*i to ipc*(i+1) will have the label i"""
    
    def init(self) -> TensorDataset:
        ''' initialize the model '''
        return TensorDataset(syn_data, syn_labels)