import torch
from torchvision import datasets
from torchvision import transforms

from dcgm.synthesize import DCGMHyperparameters, DCGMSynthesizer, run
import fire


if __name__ == '__main__':
    fire.Fire(run)
