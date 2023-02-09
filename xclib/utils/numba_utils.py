import numba as nb
import numpy as np

@nb.njit()
def in1d(a, b):
    """
    asssume a and b are unique
    """
    arr = np.concatenate((a, b))
    order = arr.argsort(kind='mergesort')
    sarr = arr[order]
    
    bool_arr = (sarr[1:] == sarr[:-1])
    flag = np.concatenate((bool_arr, np.asarray([False])))
    ret = np.empty(arr.shape, np.bool_)
    
    ret[order] = flag
    
    return ret[:len(a)]

@nb.njit()
def mean_rows(a):
    m = a.shape[0]
    ret = np.zeros((a.shape[1], ), dtype=np.float64)
    for i in range(m):
        ret += a[i]
    
    return ret / m