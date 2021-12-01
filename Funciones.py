import numpy as np
import pandas as pd

def f_loss_Corr(x, stim, y, alpha):
    return -np.corrcoef(np.dot(stim,x), y)[0, 1] + alpha*sum(abs(x))

def f_loss_Corr_ridge(x, stim, y):
    return -np.corrcoef(np.dot(stim,x), y)[0, 1]

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def rename_paths(Stims_preprocess, EEG_preprocess, stim, Band, tmin, tmax, *paths):
    returns = []
    for path in paths:
        path += 'Stim_{}_EEG_Band_{}/'.format(stim, Band)
        returns.append(path)
    return tuple(returns)

def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


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


def correlacion(x, y, axis=0):
    if (len(x) != len(y)):
        print('Error: Vectores de diferente tamaño: {} y {}.'.format(len(x), len(y)))
    else:
        Correlaciones = []
        for j in range(x.shape[axis]):
            a, b = x[j], y[j]
            corr = [1.]
            for i in range(int(len(a) / 2)):
                corr.append(np.corrcoef(a[:-i - 1], b[i + 1:])[0, 1])
            Correlaciones.append(corr)
            print("\rProgress: {}%".format(int((j + 1) * 100 / x.shape[axis])), end='')
    return np.array(Correlaciones)


def decorrelation_time(Estimulos, sr, Autocorrelation_value = 0.1):
    Autocorrelations = correlacion(Estimulos, Estimulos)
    decorrelation_times = []

    for Autocorr in Autocorrelations:
        for i in range(len(Autocorr)):
            if Autocorr[i] < Autocorrelation_value: break
        dif_paso = Autocorr[i - 1] - Autocorr[i]
        dif_01 = Autocorr[i - 1] - Autocorrelation_value
        dif_time = dif_01 / sr / dif_paso
        decorr_time = ((i - 1) / sr + dif_time) * 1000

        if decorr_time > 0 and decorr_time < len(Autocorr)/sr*1000:
            decorrelation_times.append(decorr_time)

    return decorrelation_times

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
