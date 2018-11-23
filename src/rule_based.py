import matplotlib.pyplot as plt
from data_util import *
import numpy as np
import time

def load_data(opt = "dev"):
    global X, Y
    if opt == "train":
        X, Y = data_train.getX(max_l), data_train.getY().asnumpy()+1
    elif opt == "dev":
        X, Y = data_dev.getX(max_l), data_dev.getY().asnumpy()+1
    elif opt == "test":
        X= data_test.getX(max_l)
        with open("human_label/2.txt", 'r') as f: # remember to update to the latest version
            Y = np.array([int(line[:-1]) for line in f.readlines()])
    print("dataset size: ", Y.shape[0])

mode = "dev"
load_data(mode)

from sequence import *

correct = 0
results = []

for i in range(X.shape[0]):
    '''
    if i != 124:
        continue
    '''
    f0 = X[i, 0].asnumpy()
    engy = X[i, 1].asnumpy() / 50
    L = f0.shape[0]
    ll, rr = findValidRange(f0, engy, cut=7, engy_th=6)
    L = rr - ll
    f0 = f0[ll:rr]
    engy = engy[ll:rr]
    f, flag = correct_jump(f0, threshold=0.1)
    result = stupidJudge(f, up_thres=0.08, down_thres=-0.08, turn_thres=0.08)
    results.append(result)
    if result == Y[i]:
        correct += 1
    continue
    print("No." + str(i+1)+ " valid from "+str(ll) + " to "+str(rr))
    plt.plot(range(L), f0, range(L), engy, range(L), f)
    plt.title("No."+str(i+1)+":  actual_tone:"+str(Y[i])+"  predict_tone:"+str(result))
    plt.show()

print("ACC: ", correct/X.shape[0])
if mode == "test":
    with open("results/rb_02.csv", 'w') as f:
        f.write("id,classes\n")
        for i in range(len(results)):
            f.write(str(i+1))
            f.write(',')
            f.write(str(results[i]))
            f.write('\n')