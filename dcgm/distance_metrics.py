from typing import Iterable, Sequence
from itertools import starmap
import torch
import torch.nn as nn


class DCGMDistance(nn.Module):
    """Custom distance between gradients defined by Dataset condensation with Gradient Matching Paper"""

    def forward(self, gwr: torch.Tensor, gws: torch.Tensor) -> torch.Tensor:
        shape = gwr.shape
        if len(shape) in (3, 4): # conv, out*in*h*w
            gwr = gwr.reshape(shape[0], -1)
            gws = gws.reshape(shape[0], -1)

        elif len(shape) == 2: # linear, out*in
            pass

        elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
            gwr = gwr.reshape(1, shape[0])
            gws = gws.reshape(1, shape[0])
            return torch.tensor(0, dtype=torch.float, device=gwr.device)

        return torch.sum(
            1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
        )
    

def match_loss(
        gw_syn: Iterable[torch.Tensor], 
        gw_real: Iterable[torch.Tensor], 
        distance_metric: nn.Module, 
    ):
    """Accumulates the distance provided by distance function"""
    return sum(starmap(distance_metric, zip(gw_real, gw_syn)))