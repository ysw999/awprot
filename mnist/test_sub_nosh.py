import numpy as np
import torch
from torch import nn
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from hyperopt import hp, fmin, tpe, space_eval, Trials
import cloudpickle as pickler
import multiprocessing
from multiprocessing import Process, Manager, Lock, Pool
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

def target(args):
    hy_lr = args['hy_lr']
    #hy_batch = args['hy_batch']
    #hy_epoch = args['hy_epoch']
    #int_batch = int(hy_batch*10000)
    #int_epoch = int(hy_epoch*10000)
    #acc = main(hy_lr, int_batch, int_epoch)
    acc = main(hy_lr, 256, 20)
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

    a = [([0] * 2) for i in range(10)]
    b = [0] * 10
    start = 0.00001
    end = 0.9
    interval = (end - start) / 10
    for i in range(10):
        a[i][0] = start + interval * i
        a[i][1] = start + interval * (i + 1)
    space = []
    for i in range(10):
        space.append({
            'hy_lr' : hp.uniform('hy_lr', a[i][0], a[i][1]),
            #'hy_batch' : hp.uniform('hy_batch', 0.0016,0.0256),
            #'hy_epoch' : hp.uniform('hy_epoch', 0.0050, 0.0120),
        })

    trials = Trials()
    for i in range(10):
        pickler.dump(trials, open('trials' + str(i), 'wb'))

    para_best = multiprocessing.Queue()
    s = time.time()
    evals = 0
    while True:
        evals += 1
        record = []
        for j in range(10):
            process = Process(target=find_best, args=(j, space[j], b[j], para_best))
            process.start()
            record.append(process)
        for process in record:
            process.join()
        L = sorted([para_best.get() for process in record], key = lambda x : x[2])
        print('epoch: ', evals)
        print('para: ', L[0][1])
        print('acc: ', L[0][2])
        mid = (a[L[0][0]][0] + a[L[0][0]][1]) / 2
        space[L[0][0]]['hy_lr'] = hp.uniform('hy_lr', a[L[0][0]][0], mid)
        space[L[-1][0]]['hy_lr'] = hp.uniform('hy_lr', mid, a[L[0][0]][1])
        a[L[-1][0]][0] = mid
        a[L[-1][0]][1] = a[L[0][0]][1]
        a[L[0][0]][1] = mid
        for j in range(10):
            b[j] += 5
        b[L[0][0]] = 0
        b[L[-1][0]] = 0
        trials = Trials()
        trials1 = Trials()
        trials2 = Trials()
        trials = pickler.load(open('trials'+ str(L[0][0]), 'rb'))
        for trial in trials:
            if trial['misc']['vals']['hy_lr'][0] < mid:
                tid = b[L[0][0]]
                hyperopt_trial = Trials().new_trial_docs(
                    tids=[None],
                    specs=[None],
                    results=[None],
                    miscs=[None]
                )
                hyperopt_trial[0] = trial
                hyperopt_trial[0]['tid'] = tid
                hyperopt_trial[0]['misc']['tid'] = tid
                for key in hyperopt_trial[0]['misc']['idxs'].keys():
                    hyperopt_trial[0]['misc']['idxs'][key] = [tid]
                trials1.insert_trial_docs(hyperopt_trial)
                trials1.refresh()
                b[L[0][0]] += 1
        pickler.dump(trials1, open('trials' + str(L[0][0]), 'wb'))
        pickler.dump(trials2, open('trials' + str(L[-1][0]), 'wb'))
        if L[0][2] <= -0.988:
            break
    e = time.time()
    print('time: ', e - s)
