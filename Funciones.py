import numpy as np
import pandas as pd

def flatten_list(t):
    return [item for sublist in t for item in sublist]

def make_array(*args):
    returns = []
    for var in args:
        returns.append(np.array(var))
    return tuple(returns)

def make_df(*args):
    returns = []
    for var in args:
        returns.append(pd.DataFrame(var))
    return tuple(returns)


def igualar_largos(*args):
    minimo_largo = min([var.shape[0] for var in args])

    returns = []
    for var in args:
        if var.shape[0] > minimo_largo:
            var = var[:minimo_largo]
        returns.append(var)

    return tuple(returns)


def correlacion (a, b):
    corr=[1.]
    if (len(a)!= len(b)):
        print('Error: Vectores de diferente tamaño: {} y {}.'.format(len(a), len(b)))
    else:
        for i in range(int(len(a)/2)):
            corr.append(np.corrcoef(a[:-i-1], b[i+1:])[0,1])
    return np.array(corr)


def findFreeinterval(arr):
    # If there are no set of interval
    N = len(arr)
    if N < 1:
        return

    # To store the set of free interval
    P = []

    # Sort the given interval according
    # Starting time
    arr.sort(key=lambda a: a[0])

    # Iterate over all the interval
    for i in range(1, N):

        # Previous interval end
        prevEnd = arr[i - 1][1]

        # Current interval start
        currStart = arr[i][0]

        # If Previous Interval is less
        # than current Interval then we
        # store that answer
        if prevEnd < currStart:
            P.append([prevEnd, currStart])
    return P