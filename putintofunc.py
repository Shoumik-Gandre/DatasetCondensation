from collections import defaultdict
import os
from pathlib import Path
import time
import copy
import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets
from torchvision import transforms
from utils import (
    get_loops, 
    get_dataset, 
    get_network, 
    get_eval_pool, 
    evaluate_synset, 
    get_daparam, match_loss, get_time, 
    TensorDataset, 
    epoch, 
    DiffAugment, 
    ParamDiffAug
)


def evaluate_synthetic_data(dsa, dsa_param, ipc, model_eval_pool, it, channel, num_classes, im_size, device, image_syn, label_syn, eval_real_dataloader, iterations, accs_all_exps, save_path, method, std, mean, args,):
    for model_eval in model_eval_pool:
        print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
        if dsa:
            args.epoch_eval_train = 1000
            args.dc_aug_param = None
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', dsa_param.__dict__)
        else:
            args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval, ipc) # This augmentation parameter set is only for DC method. It will be muted when dsa is True.
            print('DC augmentation parameters: \n', args.dc_aug_param)

        if dsa or args.dc_aug_param['strategy'] != 'none':
            args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
        else:
            args.epoch_eval_train = 300

        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(device) # get a random model
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, eval_real_dataloader, args)
            accs.append(acc_test)
        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

        if it == iterations: # record the final results
            accs_all_exps[model_eval] += accs

    ''' visualize and save '''
    save_name = os.path.join(save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(method, args.dataset, args.model, ipc, exp, it))
    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
    for ch in range(channel):
        image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
    image_syn_vis[image_syn_vis<0] = 0.0
    image_syn_vis[image_syn_vis>1] = 1.0
    save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.


def run(
    data_root: Path,
    save_path: Path,
    ipc: int = 10, # Images per class
    iterations: int = 1000,
    method: str = 'DC',
    eval_mode: str = 'S',
    lr_image: float = 0.1,
    lr_net: float = 0.01,
    num_exp: int = 5
):
    data_root = Path(data_root)
    save_path = Path(save_path)
    
    data_root.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    args = object()
    OUTER_LOOP, INNER_LOOP = get_loops(ipc)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DSA params
    dsa_param = ParamDiffAug()
    dsa = True if method == 'DSA' else False

    # The list of iterations when we evaluate models and record results.
    if eval_mode == 'S' or eval_mode == 'SS':
        eval_it_pool = np.arange(0, iterations+1, 500).tolist()  
    else:
        eval_it_pool = [iterations]
    print('eval_it_pool: ', eval_it_pool)

    dimensions = (3, 32, 32)
    im_size = dimensions[1]
    channel = dimensions[0]
    num_classes = 10
    mean = [0.2861]
    std = [0.3530]
    train_real_dataset = datasets.MNIST(
        data_root, 
        train=True, 
        download=True, 
        transform=transforms.Compose([ 
            transforms.Grayscale(num_output_channels=3), 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]), 
    )
    eval_real_dataset = datasets.MNIST(
        data_root, 
        train=False, 
        download=True, 
        transform=transforms.Compose([ 
            transforms.Grayscale(num_output_channels=3), 
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]), 
    )
    eval_real_dataloader = DataLoader(eval_real_dataset, batch_size=256, shuffle=False, num_workers=2)

    model_eval_pool = get_eval_pool(eval_mode, args.model, args.model)

    accs_all_exps = defaultdict(list) # record performances of all experiments

    data_save = []

    for exp in range(num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(train_real_dataset[i][0], dim=0) for i in range(len(train_real_dataset))]
        labels_all = [train_real_dataset[i][1] for i in range(len(train_real_dataset))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        syn_data = torch.randn(size=(num_classes*ipc, *dimensions), 
            dtype=torch.float, requires_grad=True, device=device)
        syn_labels = torch.arange(0, num_classes).repeat_interleave(ipc)

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                syn_data.data[c*ipc:(c+1)*ipc] = get_images(c, ipc).detach().data
        else:
            print('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([syn_data, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(device)
        print('%s training begins'%get_time())

        for it in range(iterations+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                evaluate_synthetic_data()  #! type: ignore

            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            args.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.


            for ol in range(OUTER_LOOP):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=device, dtype=torch.long) * c
                    img_syn = syn_data[c*ipc:(c+1)*ipc].reshape((ipc, *dimensions))
                    lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c

                    if dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=dsa_param)

                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == OUTER_LOOP - 1:
                    break

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(syn_data.detach()), copy.deepcopy(syn_labels.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
                for il in range(INNER_LOOP):
                    epoch('train', trainloader, net, optimizer_net, criterion, args, aug = True if dsa else False)

            loss_avg /= (num_classes*OUTER_LOOP)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == iterations: # only record the final results
                data_save.append([copy.deepcopy(syn_data.detach().cpu()), copy.deepcopy(syn_labels.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(save_path, 'res_%s_%s_%s_%dipc.pt'%(method, args.dataset, args.model, ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    fire.Fire(run)

