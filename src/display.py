import matplotlib.pyplot as plt
from data_util import *
from sequence import *
import numpy as np
import time
X, Y = data_train.getX(max_l), data_train.getY().asnumpy()+1
#X, Y = data_dev.getX(max_l), data_dev.getY().asnumpy()+1
'''
for i in range(200, 300):
    print(i+1)
    plt.plot(range(X.shape[2]), X[i, 0].asnumpy(), range(X.shape[2]), X[i, 1].asnumpy()/50 )
    plt.title(str(i+1)+":  "+str(Y[i]))
    plt.show()
    time.sleep(2)
X, Y = data_test.getX(max_l), data_train.getY().asnumpy()
with open("human_label/2.txt", 'r') as f:
    Y = [int(line[:-1]) for line in f.readlines()]
print(X, Y)
'''
print(Y.shape[0])
normal = 1
bound = 20
alert_engy = 10
alert_len1 = 6
alert_len2 = 12
dewaving = 0.15
def merge(segs, f):
    old_l = len(segs)
    for j in range(1, len(segs)-1):
        l, r, v = segs[j]
        if segs[j-1][2] == segs[j+1][2] != segs[j][2] and segs[j][2] == 0 \
            and segs[j][1] - segs[j][0] < segs[j-1][1] + segs[j+1][1] - segs[j-1][0] - segs[j+1][0]:
            segs[j][2] = segs[j-1][2]
            f[l:r] = segs[j-1][2]
    nl = [segs[0]]
    last = 0
    for i in range(1, len(segs)):
        if nl[last][2] == segs[i][2]:
            nl[last][1] = segs[i][1]
        else:
            nl.append(segs[i])
            last += 1
    if old_l == len(nl):
        return nl
    else:
        return merge(nl, f)

def group(g, engy, ori):
    f = np.zeros_like(g)
    f[:] = (g > normal) * 100 - (g < -normal) * 100
    # generate segs
    segs = []
    last = 0
    last_i = 0
    for j in range(1, f.shape[0]):
        if f[j] != last:
            segs.append([last_i, j, last])
            last = f[j]
            last_i = j
    segs.append([last_i, f.shape[0], last])
    # print(segs)
    # length limit1
    for j in range(len(segs)):
        l, r, v = segs[j]
        if r-l < alert_len1:
            segs[j][2] = 0
            f[l:r] = 0
    segs = merge(segs, f)
    print(segs)
    # energy limit
    for j in range(len(segs)):
        l, r, v = segs[j]
        if np.mean(engy[l:r]) < alert_engy:
            segs[j][2] = 0
            f[l:r] = 0
    segs = merge(segs, f)
    print(segs)
    # length limit2
    for j in range(len(segs)):
        l, r, v = segs[j]
        if r-l < alert_len2:
            # exception
            if j < len(segs) - 2 and segs[j+1][2] == 0 and segs[j+1][1] - segs[j+1][0] > alert_len2 and segs[j+2][2] > 0:
                continue
            segs[j][2] = 0
            f[l:r] = 0
    segs = merge(segs, f)
    print(segs)
    # dewaving
    for j in range(len(segs)):
        l, r, v = segs[j]
        print(np.abs(np.sum(g[l:r]))/np.mean(ori[l:r]))
        if np.abs(np.sum(g[l:r]))/np.mean(ori[l:r]) < dewaving:
            # exception
            if r-l < alert_len2:
                continue
            segs[j][2] = 0
            f[l:r] = 0
    segs = merge(segs, f)
    print(segs)
    return f, segs

def toSeq(segs):
    seq = ""
    for l, r, v in segs:
        if v != 0:
            seq = seq + ('u' if v > 0 else 'd')
    return seq

def toAns(seq):
    if seq == '':
        return 1
    elif seq == 'u':
        return 2
    elif seq == 'du':
        return 3
    elif seq == 'd':
        return 4
    else:
        return 0

for i in range(0, X.shape[0]):
    ori = X[i, 0].asnumpy()
    engy = X[i, 1].asnumpy() / 50

    g = np.zeros_like(ori)
    #g[3:-2] = (ori[3:-2] + ori[4:-1] + ori[5:] - ori[2:-3] - ori[1:-4]- ori[:-5]) / 3
    g[1:] = ori[1:] - ori[:-1]
    g[g>bound] = 0
    g[g<-bound] = 0
    f, segs = group(g, engy, ori)
    seq = toSeq(segs)
    ans = toAns(seq)
    if ans == Y[i]:
        pass
        # continue
    if Y[i] != 2:
        continue

    print("No." + str(i+1))
    print("Seq: "+seq)
    '''
    plt.plot(range(X.shape[2]), ori, range(X.shape[2]), engy, range(X.shape[2]), g, range(X.shape[2]), f, range(X.shape[2]), correct_jump(ori, 0.1)[0])
    # plt.plot(range(X.shape[2]), ori, range(X.shape[2]), engy)
    plt.legend(['f0', 'engy', 'f0\'', 'trend', 'corrected f0'])
    '''
    g = np.zeros_like(ori)    
    ll, rr = findValidRange(ori, engy, cut=7, engy_th=6)
    g[ll:rr] = -20
    plt.plot(range(X.shape[2]), ori, range(X.shape[2]), engy, range(X.shape[2]), correct_jump(ori, 0.1)[0])
    # plt.plot(range(X.shape[2]), ori, range(X.shape[2]), engy)
    plt.axvline(ll, color='r')
    plt.axvline(rr, color='r', label="Valid range")
    plt.legend(['f0', 'engy', 'corrected f0'])
    plt.title("No."+str(i+1)+" :  "+str(Y[i]))
    plt.show()