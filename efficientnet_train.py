import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from functools import partial
from time import time
from tqdm import tqdm
import argparse
import trainer
import make_graph
import numpy as np
import os

import timm
from timm.models import create_model

device = torch.device("cuda")

train_losses = []
train_accs = []
test_losses = []
test_accs = []
save_path = './model/'

s = time()

def main(args):
    use_amp = args.amp

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean, std),
    ])
    
    if args.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=False)

    elif args.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=False)

    class_names = train_dataset.classes        
    print(class_names)
    print('Class:', len(class_names))
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    model = create_model("efficientnet-b0", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b1", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b2", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b3", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b4", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b5", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b6", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b7", pretrained=True, num_classes=len(class_names)) 
    # model = create_model("efficientnet-b8", pretrained=True, num_classes=len(class_names)) 
    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epoch):
        train_loss, train_count = trainer.train(device, train_loader, model, criterion, optimizer, scaler, use_amp, epoch)
        # train_loss, train_count = trainer.saves_train(device, train_loader, model, criterion, optimizer, scaler, use_amp, epoch)

        test_loss, test_count = trainer.test(device, test_loader, model)

        train_loss = (train_loss/len(train_loader))
        train_acc = (train_count/len(train_loader.dataset))
        test_loss = (test_loss/len(test_loader))
        test_acc = (test_count/len(test_loader.dataset))

        print(f"epoch: {epoch+1},\
                train loss: {train_loss},\
                train accuracy: {train_acc}\
                test loss: {test_loss},\
                test accuracy: {test_acc}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        save_model_path = os.path.join(save_path + 'weights/',"{}.tar".format(epoch + 1))
        torch.save({
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch
            },save_model_path)
        
        make_graph.draw_loss_graph(train_losses, test_losses)
        make_graph.draw_acc_graph(train_accs,test_accs)

    e = time()
    print('Elapsed time is ',e-s)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument('--amp', action='store_true')
    args=parser.parse_args()
    main(args)
