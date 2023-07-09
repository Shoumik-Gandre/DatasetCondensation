import copy
from dataclasses import dataclass
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset, SubsetRandomSampler
from torchvision import models, datasets
from torchvision.transforms import transforms
from tqdm import tqdm

from dcgm.init_strategy import DatasetInitStrategy, RandomStratifiedInitStrategy
from dcgm.utils import has_batchnormalization, fix_batchnormalization_statistics, match_loss


@dataclass
class DCGMHyperparameters:
    iterations: int
    outer_loops: int
    inner_loops: int
    batch_size: int
    lr_dataset: float
    momentum_dataset: float
    lr_nn: float
    batchnorm_batchsize_perclass: int = 16
    ipc = 10


@dataclass
class DCGMSynthesizer:
    dimensions: Tuple[int, ...]
    num_labels: int
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    dataset_init_strategy: DatasetInitStrategy
    hyperparams: DCGMHyperparameters

    def _get_labelwise_subsets(self, dataset: Dataset, num_labels: int) -> Tuple[Subset[Tuple[torch.Tensor, torch.Tensor]]]:
        """Classwise subset is a tuple that can be indexed by a label 
        to obtain the subset of the features with those labels."""
        
        # record the indexes by class in the dataset
        indexes_by_class = [[] for _ in range(num_labels)]
        for index, (_, label) in enumerate(dataset): # type: ignore
            indexes_by_class[int(label)].append(index)
        
        # generate classwise subsets with the above indexes
        labelwise_subsets = tuple(
            Subset(dataset, indexes_by_class[label])
            for label in range(num_labels)
        )

        return labelwise_subsets

    def synthesize(self) -> TensorDataset:
        syn_dataset = self.dataset_init_strategy.init()
        real_dataset_labelwise = self._get_labelwise_subsets(dataset=self.dataset, num_labels=self.num_labels)

        # * Optimizer used on the synthetic dataset's features
        dataset_optimizer = torch.optim.SGD(params=(syn_dataset.tensors[0],), 
                                            lr=self.hyperparams.lr_dataset, 
                                            momentum=self.hyperparams.momentum_dataset)
        dataset_optimizer.zero_grad()
        # * loss function computed on the model
        criterion = nn.CrossEntropyLoss()

        for iteration in range(self.hyperparams.iterations):
            model: nn.Module = models.resnet18().to(self.device)  # TODO: write a function to assign a model
            model_params = list(model.parameters())
            model_optimizer = torch.optim.SGD(model.parameters(), lr=self.hyperparams.lr_nn)

            for eta_data in range(self.hyperparams.outer_loops):
                
                # [HANDLE BATCHNORMALIZATION]
                # TODO: Write code that updates batch norm statistics from real data 
                # 1. detect batchnormalization
                if has_batchnormalization(model):
                    # Upon detection, we need to obtain batchnorm statistics from real data
                    # TODO: now obtain batch norm statistics
                    batch_norm_dataset = []
                    # Obtain labelwise random subsets from real data
                    for subset in real_dataset_labelwise:
                        random_labelwise_subset = Subset(
                            dataset=subset, 
                            indices=random.sample(
                                range(len(subset)), 
                                self.hyperparams.batchnorm_batchsize_perclass
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
                    fix_batchnormalization_statistics(model)

                loss = torch.tensor(0.0).to(self.device)
                for label in range(self.num_labels):
                    inputs_real, labels_real = next(iter(DataLoader(real_dataset_labelwise[label], batch_size=self.hyperparams.batch_size)))
                    inputs_syn, labels_syn = next(iter(DataLoader(
                        syn_dataset, 
                        batch_size=self.hyperparams.ipc, 
                        sampler=SubsetRandomSampler(list(range(self.hyperparams.ipc * (label), self.hyperparams.ipc * (label + 1))))
                        )))
                    loss_real = criterion(model(inputs_real), labels_real)
                    gw_real = torch.autograd.grad(loss_real, model_params)
                    gw_real = list((gradients.detach().clone() for gradients in gw_real))

                    loss_syn = criterion(model(inputs_syn), labels_syn)
                    gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, 'ours', self.device)
                
                dataset_optimizer.zero_grad()
                loss.backward()
                dataset_optimizer.step()
                
                # No need to update network at final outerloop
                if eta_data == self.hyperparams.outer_loops - 1:
                    break

                ''' update network '''
                image_syn_train = copy.deepcopy(inputs_syn.detach())  # type: ignore
                label_syn_train = copy.deepcopy(labels_syn.detach())  # type: ignore
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, batch_size=self.hyperparams.batch_size, shuffle=True, num_workers=2)
                
                for eta_model in range(self.hyperparams.inner_loops):
                    model.train()
                    running_loss = 0.0
                    for inputs, labels in tqdm(trainloader):
                        inputs.to(self.device)
                        labels.to(self.device)

                        loss: torch.Tensor = criterion(model(criterion), labels)
                        running_loss += loss.item()
                        model_optimizer.zero_grad()
                        loss.backward()
                        model_optimizer.step()     
                    print(running_loss / len(trainloader))                   

        return syn_dataset
    

def run():
    device = torch.device('cuda')
    dataset = datasets.MNIST(
        r'D:\USC\DeepUSC\project1\zskd baseline\Zero-shot_Knowledge_Distillation_Pytorch\data\real', 
        train=True, 
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
    )

    hyperparams = DCGMHyperparameters(
        iterations=1, 
        outer_loops=1, 
        inner_loops=1, 
        batch_size=64, 
        lr_dataset=0.1, 
        momentum_dataset=0.5, 
        lr_nn=0.01
    )

    synthesizer = DCGMSynthesizer(
        dimensions=(3, 32, 32),
        num_labels=10,
        dataset=dataset,
        device=device,
        dataset_init_strategy=RandomStratifiedInitStrategy(dimensions=(3, 32, 32), num_classes=10, ipc=10, device=device),
        hyperparams=hyperparams
    )

    synthesizer.synthesize()
