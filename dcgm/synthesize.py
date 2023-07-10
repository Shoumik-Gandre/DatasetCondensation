from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, ConcatDataset, SubsetRandomSampler
from torchvision import models, datasets, transforms
from tqdm.auto import tqdm

from dcgm.init_strategy import DatasetInitStrategy, RandomStratifiedInitStrategy
from dcgm.handle_batchnorm import has_batchnormalization, fix_batchnormalization_statistics, update_batchnorm_statistics
from dcgm.distance_metrics import DCGMDistance, match_loss
from dcgm.model_init import ModelInitStrategy, HomogenousModelInitStrategy

from networks import LeNet5
from dcgm.train_classifier import train_step, eval_step, train


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
        syn_dataset = self.dataset_init_strategy.init()
        real_dataset_labelwise = self._get_labelwise_subsets(dataset=self.dataset, num_labels=self.num_labels)

        # * Optimizer used on the synthetic dataset's features
        dataset_optimizer = torch.optim.SGD(params=(syn_dataset.tensors[0],), 
                                            lr=self.hyperparams.lr_dataset, 
                                            momentum=self.hyperparams.momentum_dataset)
        dataset_optimizer.zero_grad()
        # * loss function computed on the model
        criterion = nn.CrossEntropyLoss()

        for iteration in tqdm(range(self.hyperparams.iterations), desc="iterations", position=0):
            
            model: nn.Module = self.model_init_strategy.init().to(self.device)  # TODO: write a function to assign a model
            model_params = list(model.parameters())
            model_optimizer = torch.optim.SGD(model.parameters(), lr=self.hyperparams.lr_nn)

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
                    shuffle=True, num_workers=2
                )
                
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
    

def run(
        data_root: str
):
    Path(data_root).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    train_dataset = datasets.MNIST(
        # r'D:\USC\DeepUSC\project1\zskd baseline\Zero-shot_Knowledge_Distillation_Pytorch\data\real',
        data_root, 
        train=True, 
        download=True,
        transform=transforms.Compose([ 
                        # transforms.Grayscale(num_output_channels=3), 
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]), 
    )
    real_images = torch.stack([train_data[0] for train_data in train_dataset])
    real_labels = torch.tensor([train_data[1] for train_data in train_dataset], dtype=torch.long)

    train_dataset = TensorDataset(real_images, real_labels)
    eval_dataset = datasets.MNIST(
        # r'D:\USC\DeepUSC\project1\zskd baseline\Zero-shot_Knowledge_Distillation_Pytorch\data\real',
        data_root, 
        train=False,
        download=True,
        transform=transforms.Compose([ 
                        # transforms.Grayscale(num_output_channels=3), 
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]), 
    )

    hyperparams = DCGMHyperparameters(
        iterations=1000, 
        outer_loops=1, 
        inner_loops=1, 
        batch_size=256, 
        lr_dataset=0.1, 
        momentum_dataset=0.5, 
        lr_nn=0.01,
        ipc=10,
    )

    dataset_init_strategy = RandomStratifiedInitStrategy(
        dimensions=(1, 32, 32), 
        num_classes=10, 
        ipc=10, 
        device=device
    )

    synthesizer = DCGMSynthesizer(
        dimensions=(1, 32, 32),
        num_labels=10,
        dataset=train_dataset,
        device=device,
        dataset_init_strategy=dataset_init_strategy,
        model_init_strategy=HomogenousModelInitStrategy(LeNet5, {'channel': 1, 'num_classes': 10}),
        hyperparams=hyperparams
    )

    dataset = synthesizer.synthesize()

    dataset = TensorDataset(dataset.tensors[0].detach(), dataset.tensors[1].detach())

    train_dataloader = DataLoader(dataset, 256, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, 256)
    
    model = LeNet5(1, 10)
    model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000//2+1], gamma=0.1)
    for _ in range(1000):
        train_step(model, nn.CrossEntropyLoss(), optimizer, train_dataloader, device)
        scheduler.step()

    loss, acc = eval_step(model, nn.CrossEntropyLoss(), eval_dataloader, device)
    print(f"{loss = } {acc = }")
