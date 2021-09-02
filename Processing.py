# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 01:28:34 2021

@author: joaco
"""
import numpy as np
import pandas as pd
from scipy import signal
import copy

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

def matriz_shifteada(features,delays):
    features = np.array(features).reshape(len(features),1)
    nt,ndim = features.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = features[-d:,:] # The last d elements until the end
        elif d>0:
            dstim[d:,:] = features[:-d,:] # All but the last d elements
        else:
            dstim = features.copy()
        dstims.append(dstim)
    dstims = np.hstack(dstims)
    print("Stimulus matrix is now %d time points by %d features (should be # original features \
    (%d) x # delays (%d))"%(dstims.shape[0], dstims.shape[1], features.shape[1], len(delays)))   
    return dstims


def labeling(s,trial,canal_hablante, sr):
    ubi= "Datos/phrases/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal_hablante)+".phrases"
    h1t = pd.read_table(ubi,header=None,sep="\t")
    ## paso los numerales a vacío. Cambio texto por 1 y silencio por 0
    h1t.iloc[:,2] =(h1t.iloc[:,2].replace("#","").apply(len)>0).apply(int)
    ## tomo la diferencia entre el tiempo en el que empieza a hablar y el tiempo que termina de hablar
    ## lo multiplico por el sampling rate (128) y lo redondeo. Me va a quedar muy parecido al largo del envelope
    ## pero con una pequeña diferencia de una o dos samples
    veces=np.round((h1t[1] - h1t[0])*sr).astype("int")
    hablante = np.repeat(h1t.iloc[:,2],veces) 
    hablante = hablante.ravel()

    ## hago lo mismo con el oyente
    oyente = (canal_hablante - 3)*-1
    ubi= "Datos/phrases/S"+str(s)+"/s"+str(s)+".objects."+"{:02d}".format(trial)+".channel"+str(oyente)+".phrases"
    h2t = pd.read_table(ubi,header=None,sep="\t")
    ## paso los numerales a vacío. Cambio texto por 1 y silencio por 0
    h2t.iloc[:,2] =(h2t.iloc[:,2].replace("#","").apply(len)>0).apply(int)
    veces=np.round((h2t[1] - h2t[0])*sr).astype("int")
    oyente = np.repeat(h2t.iloc[:,2],veces) 
    oyente = oyente.ravel()
    
    ## hay diferencias de largos entre los hablantes? corrijo con 0-padding
    diferencia = len(hablante) - len(oyente)
    if diferencia > 0:
        oyente = np.concatenate([oyente,np.repeat(0,diferencia)])
    elif diferencia <0 : 
        hablante = np.concatenate([hablante,np.repeat(0,np.abs(diferencia))])
    
    ## sumo ambos, así tengo un vectorcito de 3(hablan ambos), 2 (solo habla oyente), 
    #1 (solo habla interlocutor) y 0 (silencio).
    hablantes = hablante + oyente*2
    
    return hablantes

def butter_filter(data, frecuencias, sampling_freq, btype, order, axis, ftype):
    
    if btype =='lowpass' or btype =='highpass':
        frecuencia = frecuencias/(sampling_freq/2) 
        b, a = signal.butter(order, frecuencia, btype = btype)
    elif btype == 'bandpass':
        frecuencias = [frecuencia/(sampling_freq/2) for frecuencia in frecuencias]
        b, a = signal.butter(order, frecuencias, btype = btype)
    
    if ftype == 'Causal':
        y = signal.lfilter(b, a, data, axis = axis)
    elif ftype == 'NonCausal':
        y = signal.filtfilt(b, a, data, axis = axis, padlen = None)
    return y

def butter_bandpass_filter(data, frecuencia, sampling_freq, order, axis):
    frecuencia /= (sampling_freq/2) 
    b, a = signal.butter(order, frecuencia, btype='lowpass')
    y = signal.filtfilt(b, a, data, axis = axis, padlen = None)
    return y

def subsamplear(x,cada_cuanto):
    x = np.array(x)
    tomar = np.arange(0,len(x),int(cada_cuanto))
    return x[tomar]  


def band_freq(band):
    if band == 'Delta':
        l_freq = 1
        h_freq = 4
    elif band == 'Theta':
        l_freq = 4
        h_freq = 8
    elif band == 'Alpha':
        l_freq = 8
        h_freq = 13
    elif band == 'Beta_1':
        l_freq = 13
        h_freq = 19
    elif band == 'Beta_2':
        l_freq = 19
        h_freq = 25
    elif band == 'All':
        l_freq = None
        h_freq = 40  
        
    return l_freq, h_freq


def igualar_largos(*args):
     
    minimo_largo = min([var.shape[0] for var in args])
    
    returns = []
    for var in args:
        if var.shape[0] > minimo_largo:   
            var = var[:minimo_largo] 
        returns.append(var)
    
    return tuple(returns)
 

def preproc(momentos_escucha, delays, situacion, *args):

    momentos_escucha_matriz = matriz_shifteada(momentos_escucha, delays).astype(float)

    if situacion == 'Todo':
        return args
    
    elif situacion == 'Silencio': situacion = 0    
    elif situacion == 'Escucha': situacion = 1
    elif situacion == 'Habla': situacion = 2
    elif situacion == 'Ambos': situacion = 3
    
    momentos_escucha_matriz[momentos_escucha_matriz == situacion] = float("nan")
    
    keep_indexes = pd.isnull(momentos_escucha_matriz).all(1).nonzero()[0]
    
    returns = []
    for var in args:
        var = var[keep_indexes,:]
        returns.append(var)

    return tuple(returns)
    
def preproc_viejo(eeg, envelope, pitch, pitch_der, momentos_escucha, delays, situacion):

    momentos_escucha_matriz = matriz_shifteada(momentos_escucha, delays).astype(float)

    if situacion == 'Todo':
        return eeg, envelope, pitch
    
    elif situacion == 'Silencio': situacion = 0    
    elif situacion == 'Escucha': situacion = 1
    elif situacion == 'Habla': situacion = 2
    elif situacion == 'Ambos': situacion = 3
    
    momentos_escucha_matriz[momentos_escucha_matriz == situacion] = float("nan")
    
    keep_indexes = pd.isnull(momentos_escucha_matriz).all(1).nonzero()[0]
    eeg = eeg[keep_indexes,:]
    envelope = envelope[keep_indexes,:]
    pitch = pitch[keep_indexes,:]

    return eeg, envelope, pitch


class estandarizar():

    def __init__(self, axis = 0):
        self.axis = axis
        
    def estandarizar_data(self, data):
        data -= np.mean(data, axis = self.axis)
        data /= np.std(data, axis = self.axis)


class normalizar():
    
    def __init__(self, axis = 0, porcent = 5):
        self.axis = axis
        self.porcent = porcent
        
    def fit_normalize_percent(self, matrix):
        # Defino el termino a agarrar
        self.n = int((self.porcent*len(matrix)-1)/100)
        
        # Busco el nesimo minimo y lo resto
        sorted_matrix = copy.deepcopy(matrix)
        sorted_matrix.sort(self.axis)
        self.minn_matrix = sorted_matrix[self.n]
        matrix -= self.minn_matrix    
        
        # Vuelvo a sortear la matriz porque cambio, y busco el nesimo maximo y lo divido
        sorted_matrix = copy.deepcopy(matrix)
        sorted_matrix.sort(self.axis)
        self.maxn_matrix = sorted_matrix[-self.n]
        matrix /= self.maxn_matrix    
        
    def normalize_stims(self, matrix):
        # Los estimulos los normalizo todos entre 0 y 1 estricto, la envolvente no tiene picos
        matrix -= np.min(matrix, axis = self.axis)   
        matrix /= np.max(matrix, axis = self.axis) 


def normalizacion(eeg, dstims, Normalizar, axis = 0, porcent = 5):   
    norm = normalizar(axis, porcent)
    if Normalizar == 'EEG':
        norm.fit_normalize_percent(eeg)
    elif Normalizar == 'Stims':
        for stim in list(dstims):
            norm.normalize_stims(stim)
    elif Normalizar == 'All':
        norm.fit_normalize_percent(eeg)
        for stim in list(dstims):
            norm.normalize_stims(stim)

def estandarizacion(eeg, dstims, Estandarizar, axis = 0):          
    estandar = estandarizar(axis)
    if Estandarizar == 'EEG':
        estandar.estandarizar_data(eeg)  
    elif Estandarizar == 'Stims':
        for stim in list(dstims):
            estandar.estandarizar_data(stim)   
    elif Estandarizar == 'All':
        estandar.estandarizar_data(eeg)
        for stim in list(dstims):
            estandar.estandarizar_data(stim)


# def normalizar_stims(dstims, axis):
#     returns = []
#     for stim in [dstims]:
#         stim -= np.min(stim, axis = axis) 
#         stim /= np.max(stim, axis = axis)
#         returns.append(stim)   
#     dstims = np.hstack([returns[i] for i in range(len(returns))])       
#     return dstims

# def normalizacion(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Normalizar, axis = 0, porcent = 5):
    
#     norm = normalizar(axis, porcent)
#     if Normalizar == 'EEG':    
#         norm.fit_normalize_percent(eeg_train_val)
#         norm.fit_normalize_percent(eeg_test)
#     elif Normalizar == 'Stims':
#         for stim in [dstims]:
#             norm.normalize_stims(dstims_train_val)
#             norm.normalize_stims(dstims_test)
#     elif Normalizar == 'All':   
#         norm.fit_normalize_percent(eeg_train_val)
#         norm.fit_normalize_percent(eeg_test)
#         norm.normalize_stims(dstims_train_val)
#         norm.normalize_stims(dstims_test)

# def estandarizacion(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Estandarizar, axis = 0):        
    
#     if Estandarizar == 'EEG':
#         eeg_train_val = estandarizar(eeg_train_val, axis)
#         eeg_test = estandarizar(eeg_test, axis)         
#     elif Estandarizar == 'Stims':
#         dstims_train_val = estandarizar(dstims_train_val, axis)
#         dstims_test = estandarizar(dstims_test, axis)   
#     elif Estandarizar == 'All':
#         eeg_train_val = estandarizar(eeg_train_val, axis)
#         eeg_test = estandarizar(eeg_test, axis)
#         dstims_train_val = estandarizar(dstims_train_val, axis)
#         dstims_test = estandarizar(dstims_test, axis)