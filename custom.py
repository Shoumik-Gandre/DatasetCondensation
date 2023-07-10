from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from networks import LeNet5
from dcgm.init_strategy import RandomStratifiedInitStrategy
from dcgm.model_init import HomogenousModelInitStrategy
from dcgm.synthesize import DCGMHyperparameters, DCGMSynthesizer
from dcgm.train_classifier import train_step, eval_step
import fire

from utils import epoch
from argparse import Namespace


def evaluate_synset(net, images_train, labels_train, testloader, args: Namespace):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    print(f'Evaluate: epoch = {Epoch} train time = {int(time_train)}s train loss = {loss_train :.6f} train acc = {acc_train:.4f}, test acc = {acc_test:.4f}')

    return net, acc_train, acc_test


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
        model_init_strategy=HomogenousModelInitStrategy(
            model_class=LeNet5, 
            model_args={'channel': 1, 'num_classes': 10}
        ),
        hyperparams=hyperparams
    )

    dataset = synthesizer.synthesize()

    dataset = TensorDataset(dataset.tensors[0].detach(), dataset.tensors[1].detach())

    train_dataloader = DataLoader(dataset, 256, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, 256)

    for i in range(5):
        model = LeNet5(1, 10)
        model = nn.DataParallel(model)
        model = model.to(device)

        args = Namespace()
        args.device = 'cuda'
        args.lr_net = 0.01
        args.batch_train = 256
        args.epoch_eval_train = 1000
        args.dsa = False
        args.dc_aug_param = {
            'crop': 4,
            'scale': 0.2,
            'rotate': 45,
            'noise': 0.001,
            'strategy': 'crop_scale_rotate',
        }
        evaluate_synset(
            net=model, 
            images_train=dataset.tensors[0], 
            labels_train=dataset.tensors[1], 
            testloader=eval_dataloader, 
            args=args
        )


if __name__ == '__main__':
    fire.Fire(run)
