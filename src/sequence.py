import numpy as np

def findValidRange(f0, engy, f0_th=1, engy_th=10, cut=0):
    l, r = 0, f0.shape[0] - 1
    while l < r and f0[l] < f0_th:
        l += 1
    while l < r and f0[r] < f0_th:
        r -= 1
    fl, fr = l, r
    while l < r and engy[l] < engy_th:
        l += 1
    while l < r and engy[r] < engy_th:
        r -= 1
    while r-l < 2 + cut * 2 and engy_th > 1:
        engy_th //= 2
        l, r = fl, fr
        while l < r and engy[l] < engy_th:
            l += 1
        while l < r and engy[r] < engy_th:
            r -= 1
    if r-l < 2 + cut * 2:
        return (fl+cut, fr+1-cut)
    else:
        return (l+cut, r+1-cut)

def correct_jump(old_s, threshold=0.4):
    scale = 1.0
    detacted = False
    s = np.zeros_like(old_s)
    s[:] = old_s[:]
    start = 0
    while old_s[start] < 1 and start < old_s.shape[0]:
        start += 1
    for i in range(start+1, s.shape[0]):
        if s[i] < 1:
            s[i] = s[i-1]
        d = (s[i] * scale - s[i-1]) / s[i-1]
        if np.abs(d) > threshold:
            detacted = True
            scale = s[i-1] / s[i]
        s[i] *= scale
    return s, detacted

def stupidJudge(s, up_thres=0.1, down_thres=-0.1, turn_thres=0.1):
    m = np.min(s)
    if 1 - m/s[0] > turn_thres and 1 - m/s[-1] > turn_thres:
        return 3
    d = (s[-1] - s[0]) / s[0]
    if d > up_thres:
        return 2
    if d < down_thres:
        return 4
    return 1