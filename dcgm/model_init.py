import abc
from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset


class ModelInitStrategy(abc.ABC):
    
    @abc.abstractmethod
    def init(self) -> nn.Module:
        ...
    

@dataclass
class HomogenousModelInitStrategy(ModelInitStrategy):
    """Strategy to initalize the model"""
    model_class: Type[nn.Module]
    model_args: Mapping[str, Any]
    
    def init(self) -> nn.Module:
        ''' initialize the model '''
        return self.model_class(**self.model_args)
    

@dataclass
class HeterogenousModelInitStrategy(ModelInitStrategy):
    """The dataset is intialized randomly with the property that 
    datapoints between ipc*i to ipc*(i+1) will have the label i"""
    model_class: Type[nn.Module]
    model_args: Mapping[str, Any]
    
    def init(self) -> nn.Module:
        ''' initialize the model '''
        return self.model_class(**self.model_args)