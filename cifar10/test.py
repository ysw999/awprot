import sys
from hyperopt import hp, fmin, tpe, space_eval, Trials
import cloudpickle as pickler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager, Lock, Pool
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import time
import os
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def train(epoch, net, trainloader, device, optimizer, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def test(epoch, net, testloader, device, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = correct / total
    return acc

def main(lr, batch, epoch, mom, dw):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_epoch = 0

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = ResNet18()
    torch.cuda.set_device(0)
    net = net.to(device)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=mom, weight_decay=dw)

    for epoch in range(epoch):
        train(epoch, net, trainloader, device, optimizer, criterion)

    final_acc = test(epoch, net, testloader, device, criterion)
    return final_acc

def target(args):
    hy_lr = args['hy_lr']
    #hy_batch = args['hy_batch']
    #hy_epoch = args['hy_epoch']
    #int_batch = int(hy_batch*10000)
    #int_epoch = int(hy_epoch*10000)
    #acc = main(hy_lr, int_batch, int_epoch)
    acc = main(hy_lr, 128, 30, 0.9, 5e-4)
    return acc * -1.0

def find_best(i, space, evals, para_best):
    trials = Trials()
    trials = pickler.load(open('trials'+ str(i), 'rb'))
    best = fmin(target, space, algo=tpe.suggest, max_evals=evals + 5, trials=trials, show_progressbar=False)
    pickler.dump(trials, open('trials' + str(i), 'wb'))
    for trial in trials:
        flag = True
        for key in best:
            if trial['misc']['vals'][key][0] != best[key]:
                flag = False
                break
        if not flag:
            continue
        para_best.put((i, best, trial['result']['loss']))
        break

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    space = {
        'hy_lr' : hp.uniform('hy_lr', 0.00001, 0.9),
        #'hy_batch' : hp.uniform('hy_batch', 0.0016,0.0256),
        #'hy_epoch' : hp.uniform('hy_epoch', 0.0050, 0.0120),
    }
    acc = 0
    evals = 0
    trials = Trials()
    s = time.time()
    while acc > -0.91:
        evals += 5
        best = fmin(target, space, algo=tpe.suggest, max_evals=evals, trials=trials, show_progressbar=False)
        for trial in trials:
            flag = True
            for key in best:
                if trial['misc']['vals'][key][0] != best[key]:
                    flag = False
                    break
            if not flag:
                continue
            acc = trial['result']['loss']
            break
    pickler.dump(trials, open('trials', 'wb'))
    e = time.time()
    print('evals: ', evals)
    print('para: ', best)
    print('acc: ', acc)
    print('time: ', e - s)
