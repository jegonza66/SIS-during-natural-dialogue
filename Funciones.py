import numpy as np
import pandas as pd
import scipy

def maximo_comun_divisor(a, b):
    temporal = 0
    while b != 0:
        temporal = b
        b = a % b
        a = temporal
    return a

def minimo_comun_multiplo(a, b):
    return (a * b) / maximo_comun_divisor(a, b)

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


def make_array_dict(dict):
    keys = list(dict.keys())
    for key in keys:
        dict[key] = dict[key].to_numpy()


def make_array(*args):
    returns = []
    for var in args:
        returns.append(np.array(var))
    return tuple(returns)


def make_df_dict(dict):
    keys = list(dict.keys())
    keys.remove('info')
    if 'Phonemes' in keys:
        keys.remove('Phonemes')
    for key in keys:
        dict[key] = pd.DataFrame(dict[key])


def make_df(*args):
    returns = []
    for var in args:
        returns.append(pd.DataFrame(var))
    return tuple(returns)


def igualar_largos_dict(dict, momentos):
    keys = list(dict.keys())
    keys.remove('info')

    minimo_largo = min([dict[key].shape[0] for key in keys] + [len(momentos)])

    for key in keys:
            if dict[key].shape[0] > minimo_largo:
                dict[key] = dict[key][:minimo_largo]
    if len(momentos) > minimo_largo:
        momentos = momentos[:minimo_largo]

    return momentos


def igualar_largos_dict2(dict, momentos):
    keys = list(dict.keys())
    keys.remove('info')

    minimo_largo = min([dict[key].shape[0] for key in keys] + [len(momentos)])

    for key in keys:
        if dict[key].shape[0] > minimo_largo:
                dict[key] = dict[key][:minimo_largo]
    if len(momentos) > minimo_largo:
        momentos = momentos[:minimo_largo]

    return momentos, minimo_largo


def igualar_largos_dict_sample_data(dict, momentos, minimo_largo):
    keys = list(dict.keys())
    keys.remove('info')

    for key in keys:
        if dict[key].shape[0] > minimo_largo:
            dict[key] = dict[key][:minimo_largo]
    if len(momentos) > minimo_largo:
        momentos = momentos[:minimo_largo]

    return momentos


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


def slope(x, y):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return scipy.stats.linregress(x, y)[0]


def sliding_window(df, window_size=6, func='slope', step=1, min_points=6):
    res = []
    for i in range(0, len(df), step):
        rows = df.iloc[i:i + window_size]
        if func == "mean":
            res_i = rows.apply(lambda y: np.nanmean(y) if sum(~np.isnan(y)) >= min_points else np.nan)
        elif func == "slope":
            x = rows.index
            res_i = rows.apply(lambda y: slope(x, y) if sum(~np.isnan(y)) >= min_points else np.nan)
        res.append(res_i)
    res = np.array(res)
    return res