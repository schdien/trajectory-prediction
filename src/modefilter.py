import numpy as np
from scipy import stats


def mode_filter(x,r):
    l = len(x)
    filtered_x = np.zeros(l)
    for i in range(l):
        start = max(0,i-r)
        end = min(l,i+r+1)
        filtered_x[i] = stats.mode(x[start:end],keepdims=True)[0]
    return filtered_x


if __name__ =='__main__':
    x = np.array([10,20,20,30,40,50,50,60,70,80,80,90])
    print(mode_filter(x,2))