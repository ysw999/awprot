import numpy as np
import torch
from torch import nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from mango import Tuner, scheduler
from mango.domain.distribution import loguniform
from scipy.stats import uniform
import time

class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

def main(lr, batch, epoch):
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)
    model = Model()
    torch.cuda.set_device(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    sgd = SGD(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    for current_epoch in range(epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            train_x, train_label = train_x.to(device), train_label.to(device)
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            sgd.step()

    all_correct_num = 0
    all_sample_num = 0
    model.eval()
    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x, test_label = test_x.to(device), test_label.to(device)
        predict_y = model(test_x.float())
        _, predicted = predict_y.max(1)
        all_sample_num += test_label.size(0)
        all_correct_num += predicted.eq(test_label).sum().item()
    acc = all_correct_num / all_sample_num
    return acc

@scheduler.parallel(n_jobs=10)
def target(hy_lr, hy_batch, hy_epoch):
    int_batch = int(hy_batch*10000)
    int_epoch = int(hy_epoch*10000)
    acc = main(hy_lr, int_batch, int_epoch)
    return acc * -1.0

def early_stop(results):
    acc = results['best_objective']
    return acc < -0.99

if __name__ == '__main__':
    param_space = dict(hy_lr=loguniform(-5, 5), hy_batch=uniform(0.0016, 0.0240), hy_epoch=uniform(0.0015, 0.0035))
    s = time.time()
    tuner = Tuner(param_space, target, {'num_iteration': 500, 'early_stopping': early_stop})
    results = tuner.minimize()
    e = time.time()
    print('time: ', e - s)
    print('para: ', results['best_params'])
    print('acc: ', results['best_objective'])
    print('data: ', results['params_tried'])
    print('values: ', results['objective_values'])
