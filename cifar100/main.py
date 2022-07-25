import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from hyperopt import hp, fmin, tpe, space_eval, Trials
import cloudpickle as pickler
import multiprocessing
from multiprocessing import Process, Manager, Lock, Pool

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))
'''
def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
'''

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main(lr, mom, batch, dw, epoch):
    best_prec = 0
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--epochs', default=epoch, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
    parser.add_argument('--lr', '--learning-rate', default=lr, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('-ct', '--cifar-type', default='10', type=int, metavar='CT', help='10 for cifar10,100 for cifar100 (default: 10)')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !

        # model = resnet20_cifar()
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()
        model = resnet164_cifar(num_classes=100)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)

        #model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/resnet20_cifar100'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
            model_type = 1
        elif isinstance(model, Wide_ResNet_Cifar):
            model_type = 2
        elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
            model_type = 3
        else:
            print('model type unrecognized...')
            return

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=mom, weight_decay=dw)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('=> loading cifar100 data...')
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)


    if args.evaluate:
        validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, model_type)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

    return best_prec

def target(args):
    hy_lr = args['hy_lr']
    hy_mom = args['hy_mom']
    hy_batch = args['hy_batch']
    dw = args['dw']
    hy_epoch = args['hy_epoch']
    int_batch = int(hy_batch*10000)
    int_epoch = int(hy_epoch*10000)
    acc = main(hy_lr, hy_mom, int_batch, dw, 1)
    return acc * -1.0

def find_best(i, space, evals, para_best):
    trials = Trials()
    trials = pickler.load(open('trials'+ str(i), 'rb'))
    best = fmin(target, space, algo=tpe.suggest, max_evals=evals + 10, trials=trials, show_progressbar=False)
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

if __name__=='__main__':
    multiprocessing.set_start_method('spawn')

    space = {
        'hy_lr' : hp.uniform('hy_lr', 0.00001, 0.9),
        'hy_mom' : hp.uniform('hy_mom', 0.001, 0.9),
        'hy_batch' : hp.uniform('hy_batch', 0.0016,0.0256),
        'dw' : hp.uniform('dw', 0.000001, 0.01),
        'hy_epoch' : hp.uniform('hy_epoch', 0.0050, 0.0120),
        'hy_drop' : hp.uniform('hy_drop', 0.1, 0.7),
    }
    s = time.time()
    trials = Trials()
    best = fmin(target, space, algo=tpe.suggest, max_evals=10, trials=trials, show_progressbar=False)
    e = time.time()
    print('para: ', best)
    for trial in trials:
        print('acc: ', trial['result']['loss'])
    print('time: ', e - s)

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
            'hy_mom' : hp.uniform('hy_mom', 0.001, 0.9),
            'hy_batch' : hp.uniform('hy_batch', 0.0016,0.0256),
            'dw' : hp.uniform('dw', 0.000001, 0.01),
            'hy_epoch' : hp.uniform('hy_epoch', 0.0050, 0.0120),
            'hy_drop' : hp.uniform('hy_drop', 0.1, 0.7),
        })

    trials = Trials()
    for i in range(10):
        pickler.dump(trials, open('trials' + str(i), 'wb'))

    para_best = multiprocessing.Queue()
    s = time.time()
    for i in range(10):
        record = []
        for j in range(10):
            process = Process(target=find_best, args=(j, space[j], b[j], para_best))
            process.start()
            record.append(process)
        for process in record:
            process.join()
        L = sorted([para_best.get() for process in record], key = lambda x : x[2])
        print('epoch: ', i)
        print('para: ', L[0][1])
        print('acc: ', L[0][2])
        mid = (a[L[0][0]][0] + a[L[0][0]][1]) / 2
        space[L[0][0]]['hy_lr'] = hp.uniform('hy_lr', a[L[0][0]][0], mid)
        space[L[-1][0]]['hy_lr'] = hp.uniform('hy_lr', mid, a[L[0][0]][1])
        a[L[-1][0]][0] = mid
        a[L[-1][0]][1] = a[L[0][0]][1]
        a[L[0][0]][1] = mid
        for j in range(10):
            b[j] += 10
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
            else:
                tid = b[L[-1][0]]
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
                trials2.insert_trial_docs(hyperopt_trial)
                trials2.refresh()
                b[L[-1][0]] += 1
        pickler.dump(trials1, open('trials' + str(L[0][0]), 'wb'))
        pickler.dump(trials2, open('trials' + str(L[-1][0]), 'wb'))
    e = time.time()
    print('time: ', e - s)
