import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread, BoundedSemaphore
from model import ResNet18, Artificial_Astrocyte_Network
from utils import progress_bar

torch.autograd.set_detect_anomaly(True)



parser = argparse.ArgumentParser(description='PyTorch Radiomics Training')
parser.add_argument('--lr_cnn', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lr_meta_learner', default=1e-3, type=float, help='learning rate')
parser.add_argument('--train_batch_size', default=128, type=int, help='batch size')
parser.add_argument('--val_batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=1000, type=int, help='max epoch')
args = parser.parse_args()
gpu = "5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1
best_aan_acc = 0
train_loss_cnn = 1000
train_loss_aan = 1000


def cnn_train(epoch, dataloader, cnn, aan, criterion, optimizer_cnn, optimizer_aan):
    global train_loss_cnn
    cnn.train()
    aan.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        num_id += 1
        for params_c in cnn.parameters():
            params_c.requires_grad = True
        for params_a in aan.parameters():
            params_a.requires_grad = False
        inputs, targets = inputs.to(device), targets.to(device)
        pattern = 0
        weight_mean = cnn(inputs, 0, 0, pattern)
        pr1 = aan(weight_mean)
        pattern = 1
        outputs, weights_max_1 = cnn(inputs, pr1, 0,  pattern)
        pr2 = aan(weights_max_1)
        loss = criterion(outputs, targets.long().squeeze())
        optimizer_cnn.zero_grad()
        optimizer_aan.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_cnn.step()
        pattern = 2
        outputs = cnn(inputs, pr1, pr2, pattern)
        loss = criterion(outputs, targets.long().squeeze())
        optimizer_cnn.zero_grad()
        optimizer_aan.zero_grad()
        loss.backward()
        optimizer_cnn.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('cnn:', epoch, 'Loss:', train_loss / num_id, 'Acc:', 100. * correct / total)
    train_loss_cnn = train_loss / num_id
    return train_loss / num_id, 100. * correct / total


def aan_train(epoch, dataloader, cnn, aan, criterion, optimizer_cnn, optimizer_aan):
    global train_loss_aan
    cnn.train()
    aan.train()
    num_id = 0
    train_loss = 0
    correct = 0
    total = 0
    for batch_id, (inputs, targets) in enumerate(dataloader):
        num_id += 1
        for params_c in cnn.parameters():
            params_c.requires_grad = False
        for params_a in aan.parameters():
            params_a.requires_grad = True
        inputs, targets = inputs.to(device), targets.to(device)
        pattern = 0
        weight_mean = cnn(inputs, 0, 0, pattern)
        pr1 = aan(weight_mean)
        pattern = 1
        outputs, weights_max_1 = cnn(inputs, pr1, 0,  pattern)
        pr2 = aan(weights_max_1)
        loss = criterion(outputs, targets.long().squeeze())
        optimizer_cnn.zero_grad()
        optimizer_aan.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_cnn.step()
        pattern = 2
        outputs = cnn(inputs, pr1, pr2, pattern)
        loss = criterion(outputs, targets.long().squeeze())
        optimizer_cnn.zero_grad()
        optimizer_aan.zero_grad()
        loss.backward()
        optimizer_cnn.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('aan:', epoch, 'Loss:', train_loss / num_id, 'Acc:', 100. * correct / total)
    train_loss_aan = train_loss / num_id
    return train_loss / num_id, 100. * correct / total


if __name__ == '__main__':
    print('==> Preparing data..')
    trainloader = DataLoader(torch.load('../Data/Train_CIFAR10.t7', weights_only=False), batch_size=args.train_batch_size, shuffle=True)
    valloader = DataLoader(torch.load('../Data/Val_CIFAR10.t7', weights_only=False), batch_size=args.val_batch_size, shuffle=True)
    print('==> Building model..')
    cnn = ResNet18()
    cnn.to(device)
    # cnn = torch.nn.DataParallel(cnn)
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=args.lr_cnn)
    # scheduler_cnn = torch.optim.lr_scheduler.MultiStepLR(optimizer_cnn, milestones=[30, 40], gamma=0.1, last_epoch=-1)
    scheduler_cnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    aan = Artificial_Astrocyte_Network()
    aan.to(device)
    # aan = torch.nn.DataParallel(aan)
    optimizer_aan = optim.Adam(aan.parameters(), lr=args.lr_meta_learner)
    # scheduler_aan = torch.optim.lr_scheduler.MultiStepLR(optimizer_aan, milestones=[40, 150, 175, 200], gamma=0.1, last_epoch=-1)
    scheduler_aan = optim.lr_scheduler.ReduceLROnPlateau(optimizer_aan, mode='min', factor=0.1, patience=20, verbose=True, threshold=1e-4, threshold_mode='rel')
    criterion = nn.CrossEntropyLoss()
    print('==> Training..')
    cnn_loss_lst, cnn_acc_lst, aan_loss_lst, aan_acc_lst = [], [], [], []
    for epoch in range(start_epoch, start_epoch+args.epoch):
        cnn_loss, cnn_acc = cnn_train(epoch, trainloader, cnn, aan, criterion, optimizer_cnn, optimizer_aan)
        aan_loss, aan_acc = aan_train(epoch, valloader, cnn, aan, criterion, optimizer_cnn, optimizer_aan)
        scheduler_cnn.step(train_loss_cnn)
        scheduler_aan.step(train_loss_aan)
        lr_cnn = optimizer_cnn.param_groups[0]['lr']
        lr_aan = optimizer_aan.param_groups[0]['lr']
        cnn_loss_lst.append(cnn_loss)
        cnn_acc_lst.append(cnn_acc)
        aan_loss_lst.append(aan_loss)
        aan_acc_lst.append(aan_acc)

        print('Saving:')
        plt.figure(num=1, dpi=200)
        plt.subplot(2, 2, 1)
        picture1, = plt.plot(np.arange(0, len(cnn_loss_lst)), cnn_loss_lst, color='red', linewidth=1.0, linestyle='-')
        plt.legend(handles=[picture1], labels=['cnn_loss'], loc='best')
        plt.subplot(2, 2, 2)
        picture2, = plt.plot(np.arange(0, len(cnn_acc_lst)), cnn_acc_lst, color='blue', linewidth=1.0, linestyle='-')
        plt.legend(handles=[picture2], labels=['cnn_acc'], loc='best')
        plt.subplot(2, 2, 3)
        picture3, = plt.plot(np.arange(0, len(aan_loss_lst)), aan_loss_lst, color='red', linewidth=1.0, linestyle='-')
        plt.legend(handles=[picture3], labels=['aan_loss'], loc='best')
        plt.subplot(2, 2, 4)
        picture4, = plt.plot(np.arange(0, len(aan_acc_lst)), aan_acc_lst, color='blue', linewidth=1.0, linestyle='-')
        plt.legend(handles=[picture4], labels=['aan_acc'], loc='best')
        plt.savefig('./ResNet18.jpg')
        plt.figure(num=2, dpi=200)

        if lr_cnn > 5e-5 or lr_aan > 5e-5:
            if best_aan_acc < aan_acc:
                print('Saving:')
                state1 = {
                    'net': aan.state_dict()
                }
                if not os.path.isdir('AAN_Model'):
                    os.mkdir('AAN_Model')
                torch.save(state1, './AAN_Model/aan_params''.t7')
                best_aan_acc = aan_acc
                acc = open('./Best_AAN.txt', 'w')
                acc.write(str(best_aan_acc))
                acc.close()
        else:
            break
