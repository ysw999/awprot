from hyperopt import hp, fmin, tpe, space_eval, Trials
from multiprocessing import Process, Manager
import cloudpickle as pickler
import numpy as np
import time
import os

def f(args):
    x1 = args['x1']
    x2 = args['x2']
    u = 1.6 * x1 - 0.5
    v = 1.6 * x2 - 0.5
    fval = (u ** 2 + v ** 2 - 0.3 * np.cos(3 * np.pi * u) - 0.3 * np.cos(3 * np.pi * v))
    print(fval)
    return fval

def find_best(i):
    trials = Trials()
    trials = pickler.load(open('trials'+ str(i), 'rb'))
    best = fmin(f, space[i], algo=tpe.suggest, max_evals=b[i] + 5, trials=trials, show_progressbar=False)
    pickler.dump(trials, open('trials' + str(i), 'wb'))
    for trial in trials:
        flag = True
        for key in best:
            if trial['misc']['vals'][key][0] != best[key]:
                flag = False
                break
        if not flag:
            continue
        para_best[i] = (trial['result']['loss'], best)
        break

if __name__ == '__main__':
    space = {
        'x1' : hp.uniform('x1', 0.0, 1.0),
        'x2' : hp.uniform('x2', 0.0, 1.0),
    }
    s = time.time()
    trials = Trials()
    best = fmin(f, space, algo=tpe.suggest, max_evals=5, trials=trials, show_progressbar=False)
    e = time.time()
    print(best)
    print(f(best))
    print('time: ', e - s)

    a = [([0] * 2) for i in range(10)]
    b = [0] * 10
    start = 0
    end = 1.0
    interval = (end - start) / 10
    for i in range(10):
        a[i][0] = start + interval * i
        a[i][1] = start + interval * (i + 1)
    space = []
    for i in range(10):
        space.append({
            'x1' : hp.uniform('x1', a[i][0], a[i][1]),
            'x2' : hp.uniform('x2', 0.0, 1.0),
        })

    trials = Trials()
    for i in range(10):
        pickler.dump(trials, open('trials' + str(i), 'wb'))

    para_best = Manager().dict()
    s = time.time()
    for i in range(3):
        record = []
        for j in range(10):
            process = Process(target=find_best, args=(j, ))
            process.start()
            record.append(process)
        for process in record:
            process.join()
        L = sorted(para_best.items(), key = lambda kv : kv[1][0])
        print('epoch ', i)
        print(L[0][1])
        mid = (a[L[0][0]][0] + a[L[0][0]][1]) / 2
        space[L[0][0]]['x1'] = hp.uniform('x1', a[L[0][0]][0], mid)
        space[L[-1][0]]['x1'] = hp.uniform('x1', mid, a[L[0][0]][1])
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
            if trial['misc']['vals']['x1'][0] < mid:
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
