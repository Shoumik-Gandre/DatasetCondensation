import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import augment, get_daparam


def train_step(
        model: nn.Module, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device,
        augment_flag: bool=False,
        lr_scheduler=None,
):
    model.train().to(device)
    for i, (images, labels) in enumerate(pbar := tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)

        #! REMOVE THIS CODE BEFORE PRODUCTION
        if augment_flag:
            images = augment(images, get_daparam('mnist', None, None, 1), device=device)
        
        #! END

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()
        pbar.set_description(f"Loss {loss.item():.5f}")


def eval_step(
        model: nn.Module, 
        criterion: nn.Module,
        dataloader: DataLoader,
        device: torch.device
):
    model.eval().to(device)
    total_correct = 0
    loss = 0.0
    num_items = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(pbar := tqdm(dataloader), start=1):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss += criterion(output, labels)
            pred = output.argmax(dim=1)
            total_correct += (pred == labels).sum()

            num_items += images.shape[0]
            pbar.set_description(
                f"Loss {loss.item() / (num_items):.5f}" \
                + f" | accuracy: {total_correct / (num_items):.5f}"
            )

    loss /= len(dataloader.dataset) # type: ignore
    acc = float(total_correct) / len(dataloader.dataset) # type: ignore
    pbar.set_description(f"Loss {loss.item():.5f} | accuracy: {acc}")  # type: ignore
    return float(loss), acc


def train(
        model: nn.Module, 
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epoch: int,
        device: torch.device
):
    for epoch in range(1, num_epoch+1):
        print(f"[{epoch}/{num_epoch}]")
        train_step(model, criterion, optimizer, train_dataloader, device)
        loss, acc = eval_step(model, criterion, eval_dataloader, device)
        print(f"{loss = } {acc = }")
