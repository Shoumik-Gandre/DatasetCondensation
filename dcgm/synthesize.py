from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, SubsetRandomSampler
from tqdm.auto import tqdm

from dcgm.init_strategy import DatasetInitStrategy
from dcgm.handle_batchnorm import has_batchnormalization, fix_batchnormalization_statistics, update_batchnorm_statistics
from dcgm.distance_metrics import DCGMDistance, match_loss
from dcgm.model_init import ModelInitStrategy
from dcgm.train_classifier import train_step


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
    ipc: int = 1


@dataclass
class DCGMSynthesizer:
    dimensions: Tuple[int, ...]
    num_labels: int
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    dataset_init_strategy: DatasetInitStrategy
    model_init_strategy: ModelInitStrategy
    hyperparams: DCGMHyperparameters
    distance_metric: nn.Module = DCGMDistance()

    def synthesize(self) -> TensorDataset:
        # Generate Synthetic Dataset
        syn_dataset = self.dataset_init_strategy.init()

        # Subset the real dataset labelwise
        real_dataset_labelwise = self._get_labelwise_subsets(dataset=self.dataset, num_labels=self.num_labels)

        # Optimizer used on the synthetic dataset's features
        dataset_optimizer = torch.optim.SGD(params=(syn_dataset.tensors[0],), 
                                            lr=self.hyperparams.lr_dataset, 
                                            momentum=self.hyperparams.momentum_dataset)
        dataset_optimizer.zero_grad()
        # loss function computed on the model
        criterion = nn.CrossEntropyLoss()

        # Loop for training synthetic data on multiple model initializations
        for iteration in tqdm(range(self.hyperparams.iterations), desc="iterations", position=0):
            
            # Initialize Model
            model: nn.Module = self.model_init_strategy.init().to(self.device)
            model_params = list(model.parameters())
            model_optimizer = torch.optim.SGD(model.parameters(), lr=self.hyperparams.lr_nn)

            # Partial Optimization loops for synthetic data
            for eta_data in range(self.hyperparams.outer_loops):

                # [HANDLE BATCHNORMALIZATION]
                # detect batchnormalization
                if has_batchnormalization(model):
                    # Upon detection, we need to obtain batchnorm statistics from real data
                    update_batchnorm_statistics(model, real_dataset_labelwise, self.hyperparams.batchnorm_batchsize_perclass)
                    fix_batchnormalization_statistics(model)

                gradient_distance = torch.tensor(0.0, device=self.device)
                for label in range(self.num_labels):

                    dataloader_real = DataLoader(
                        real_dataset_labelwise[label], 
                        batch_size=self.hyperparams.batch_size
                    )
                    dataloader_syn = DataLoader(
                        syn_dataset, 
                        batch_size=self.hyperparams.ipc, 
                        sampler=SubsetRandomSampler(
                            indices=range(
                                self.hyperparams.ipc * (label), 
                                self.hyperparams.ipc * (label + 1)
                            )
                        )
                    )
                    
                    # Compute real Gradients
                    inputs_real, labels_real = next(iter(dataloader_real))
                    inputs_real = inputs_real.to(self.device)
                    labels_real = labels_real.to(self.device)
                    loss_real = criterion(model(inputs_real), labels_real)
                    gw_real = torch.autograd.grad(loss_real, model_params)
                    gw_real = (gradients.detach() for gradients in gw_real)
                    
                    # Compute Synthetic Gradients
                    inputs_syn, labels_syn = next(iter(dataloader_syn))
                    inputs_syn = inputs_syn.to(self.device)
                    labels_syn = labels_syn.to(self.device)
                    loss_syn = criterion(model(inputs_syn), labels_syn)
                    gw_syn = torch.autograd.grad(loss_syn, model_params, create_graph=True)

                    # Aggregate distance between gradients
                    gradient_distance += match_loss(gw_syn, gw_real, self.distance_metric)
                
                dataset_optimizer.zero_grad()
                gradient_distance.backward()
                dataset_optimizer.step()
                
                # No need to update network at final outerloop
                if eta_data == self.hyperparams.outer_loops - 1:
                    break

                ''' update network '''
                trainloader = DataLoader(
                    TensorDataset(
                        inputs_syn.detach(),  # type: ignore
                        labels_syn.detach()  # type: ignore
                    ),
                    batch_size=self.hyperparams.batch_size, 
                    shuffle=True, num_workers=0
                )
                
                # Partial Optimization Loops for the Model
                for eta_model in range(self.hyperparams.inner_loops):
                    train_step(model, criterion, model_optimizer, trainloader, self.device, augment_flag=True)                 

        return syn_dataset

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
    