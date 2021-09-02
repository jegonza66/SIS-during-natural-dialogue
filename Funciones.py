# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:26:19 2021

@author: joaco
"""
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io.wavfile as wavfile
from scipy import stats
from scipy import signal
import librosa
import copy
import os
import pickle
import seaborn as sn
from scipy import signal as sgn
import math
# from sklearn.utils import resample
# from scipy.linalg import toeplitz

from praatio import pitch_and_intensity
from praatio import praat_scripts
from praatio import tgio
from praatio.utilities import utils



def matriz_shifteada(features,delays):
    features = np.matrix(features).T
    nt,ndim = features.shape
    dstims = []
    for di,d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d<0: ## negative delay
            dstim[:d,:] = features[-d:,:] # The last d elements until the end
        elif d>0:
            dstim[d:,:] = features[:-d,:] # All but the last d elements
        else:
            dstim =  features.copy()
        dstims.append(dstim)
    dstims = np.hstack(dstims)
    print("Stimulus matrix is now %d time points by %d features (should be # original features \
    (%d) x # delays (%d))"%(dstims.shape[0], dstims.shape[1], features.shape[1], len(delays)))   
    return dstims


def solo_escucha(s,trial,canal_hablante, sr):
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

def butter_bandpass_filter(data, frecuencia, sampling_freq, order, axis):
    frecuencia /= (sampling_freq/2) 
    b, a = signal.butter(order, frecuencia, btype='lowpass')
    y = signal.filtfilt(b, a, data, axis = axis, padlen = None)
    return y

def f_pitch(s, trial, canal, delays):
    fname = "S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal)+".wav"
    ubi = "C:/Users/joaco/Desktop/Joac/Facultad/Tesis/Código/Datos/wavs/"+fname  
    praatEXE = r"C:\Program Files\Praat\Praat.exe"
    
    output_path = "C:/Users/joaco/Desktop/Joac/Facultad/Tesis/Código/Datos/Pitch"
    try: os.makedirs(output_path)
    except:pass
    output_path += fname[:-4]+".txt"
    sampleStep = 0.01
    minPitch = 50
    maxPitch = 350
    silenceThreshold = 0.01
    
    pitch_and_intensity.extractPI(ubi,output_path, praatEXE, minPitch, maxPitch, sampleStep, silenceThreshold)
    read_file = pd.read_csv(output_path)
    
    time = np.array(read_file['time'])
    pitch = np.array(read_file['pitch'])
    intensity = np.array(read_file['intensity'])
    pitch[pitch=='--undefined--'] = float(0)
    pitch = np.repeat(pitch,16000*sampleStep) 
    pitch = subsamplear(pitch,125)
    pitch = matriz_shifteada(pitch, delays)    
    # clean_pitch = [float(pitch[i]) for i in range(len(pitch)) if not math.isnan(pitch[i])]

    return pitch


def f_envelope(s,trial,canal, delays):
    ubi = "Datos/wavs/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal)+".wav" 
    wav1 = wavfile.read(ubi)[1]
    wav1 = wav1.astype("float")
    
    ### envelope
    envelope = np.abs(sgn.hilbert(wav1))
    envelope = butter_bandpass_filter(envelope, 25, 16000, 3, 0)
    window_size = 125
    stride = 125
    envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size <= len(envelope)])
    envelope = envelope.ravel().flatten()
        
    envelope = matriz_shifteada(envelope,delays) # armo la matriz shifteada
    
    return np.array(envelope)


def f_eeg(s,trial,l_freq,h_freq,canal,sr):
    ubi = "Datos/EEG/S"+str(s)+"/s"+str(s)+"-"+str(canal)+"-Trial"+str(trial)+"-Deci-Filter-Trim-ICA-Pruned.set"
    eeg = mne.io.read_raw_eeglab(ubi)
    eeg_freq = eeg.info.get("sfreq")
    eeg.load_data()
    
    # Hago un lowpass
    eeg = eeg.filter(l_freq = l_freq, h_freq = h_freq)
        
    # Paso a array
    eeg = eeg.to_data_frame()
    eeg = np.array(eeg)
    eeg = eeg[:,1:129] ## tomo 128 canales, tiro la primer columna de tiempos

    # Subsampleo
    eeg = subsamplear(eeg, int(eeg_freq/sr))  
    
    #Defino montage e info
    montage = mne.channels.make_standard_montage('biosemi128')
    channel_names = montage.ch_names
    info = mne.create_info(ch_names = channel_names[:], sfreq = sr, ch_types = 'eeg').set_montage(montage)

    return eeg, info


def subsamplear(x,cada_cuanto):
    x = np.array(x)
    tomar = np.arange(0,len(x),int(cada_cuanto))
    return x[tomar]  


def subsamplear_promediando(x,cada_cuanto):
    x = x.transpose()
    subsampleada = []
    for channel in range(len(x)):
        subsampleada.append(np.array([np.mean(x[channel][i:i+cada_cuanto]) for i in range(0, len(x[channel]), cada_cuanto) if i+cada_cuanto <= len(x[channel])])) 
    return np.array(subsampleada).transpose()


def band_freq(band):
    if band == 'Theta':
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
 
    
def preproc(eeg, envelope, pitch, momentos_escucha, delays, Normalizar_todo, situacion):

    momentos_escucha_matriz = matriz_shifteada(momentos_escucha, delays).astype(float)

    if situacion == 'Todo':
        if not Normalizar_todo: 
            pitch = (pitch - pitch.mean(0))/pitch.std(0)
            envelope = (envelope - envelope.mean(0))/envelope.std(0)
            eeg = (eeg - eeg.mean(0))/eeg.std(0) 
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

    # Estandarizo (si no normalizo despues)
    if not Normalizar_todo: 
        pitch = (pitch - pitch.mean(0))/pitch.std(0)
        envelope = (envelope - envelope.mean(0))/envelope.std(0)
        eeg = (eeg - eeg.mean(0))/eeg.std(0) 

    return eeg, envelope, pitch


def load_files_process_data(sesion, sr, delays, l_freq, h_freq, Save_procesed_data, procesed_data_path, Normalizar_todo, situacion):
    ###### Armo estructura de datos de sujeto ######
    eeg_sujeto_1 = pd.DataFrame()
    envelope_para_sujeto_1 = pd.DataFrame()
    pitch_para_sujeto_1 = pd.DataFrame()
    
    eeg_sujeto_2 = pd.DataFrame()
    envelope_para_sujeto_2 = pd.DataFrame()
    pitch_para_sujeto_2 = pd.DataFrame()
    
    run = True
    trial = 1
    while run:           
        try: 
            ###### Cargo data ######
            eeg_trial_sujeto_1, info = f_eeg(sesion, trial, l_freq, h_freq, canal = 1, sr = sr)
            envelope_trial_para_sujeto_1 = f_envelope(sesion, trial, canal = 2, delays = delays)
            momentos_escucha_sujeto_1 = solo_escucha(sesion, trial, canal_hablante = 2, sr = sr)
            pitch_trial_para_sujeto_1 = f_pitch(sesion, trial, canal = 2, delays = delays)
            
            eeg_trial_sujeto_2, info = f_eeg(sesion, trial, l_freq, h_freq, canal = 2, sr = sr)
            envelope_trial_para_sujeto_2 = f_envelope(sesion, trial, canal = 1, delays = delays)
            momentos_escucha_sujeto_2 = solo_escucha(sesion, trial, canal_hablante = 1, sr = sr)
            pitch_trial_para_sujeto_2 = f_pitch(sesion, trial, canal = 1, delays = delays)
        except: 
            run = False
                      
        if run: 
            ###### Igualar largos ######
            eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, momentos_escucha_sujeto_1 = igualar_largos(eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, momentos_escucha_sujeto_1)
            eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, momentos_escucha_sujeto_2 = igualar_largos(eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, momentos_escucha_sujeto_2)
            
            ###### Preprocesamiento ######
            eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1 = preproc(eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, momentos_escucha_sujeto_1, delays, Normalizar_todo, situacion)
            eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2 = preproc(eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, momentos_escucha_sujeto_2, delays, Normalizar_todo, situacion)
            
            eeg_trial_sujeto_1 = pd.DataFrame(eeg_trial_sujeto_1)
            envelope_trial_para_sujeto_1 = pd.DataFrame(envelope_trial_para_sujeto_1)
            pitch_trial_para_sujeto_1 = pd.DataFrame(pitch_trial_para_sujeto_1)
            
            eeg_trial_sujeto_2 = pd.DataFrame(eeg_trial_sujeto_2)
            envelope_trial_para_sujeto_2 = pd.DataFrame(envelope_trial_para_sujeto_2)
            pitch_trial_para_sujeto_2 = pd.DataFrame(pitch_trial_para_sujeto_2)
            
            ###### Adjunto a datos de sujeto ######
            if len(eeg_trial_sujeto_1): 
                eeg_sujeto_1 = eeg_sujeto_1.append(eeg_trial_sujeto_1)
                envelope_para_sujeto_1 = envelope_para_sujeto_1.append(envelope_trial_para_sujeto_1)
                pitch_para_sujeto_1 = pitch_para_sujeto_1.append(pitch_trial_para_sujeto_1)
            
            if len(eeg_trial_sujeto_2):
                eeg_sujeto_2 = eeg_sujeto_2.append(eeg_trial_sujeto_2)
                envelope_para_sujeto_2 = envelope_para_sujeto_2.append(envelope_trial_para_sujeto_2)
                pitch_para_sujeto_2 = pitch_para_sujeto_2.append(pitch_trial_para_sujeto_2)
                
            trial += 1
            
    eeg_sujeto_1 = np.array(eeg_sujeto_1)
    envelope_para_sujeto_1 = np.array(envelope_para_sujeto_1)
    pitch_para_sujeto_1 = np.array(pitch_para_sujeto_1)
    
    eeg_sujeto_2 = np.array(eeg_sujeto_2)
    envelope_para_sujeto_2 = np.array(envelope_para_sujeto_2)
    pitch_para_sujeto_2 = np.array(pitch_para_sujeto_2)
    
    if Save_procesed_data:
        try: os.makedirs(procesed_data_path)
        except: pass
        
        f = open(procesed_data_path + 'Sesion{}_Sujeto{}.pkl'.format(sesion, 1), 'wb')
        pickle.dump([eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1], f)
        f.close()
        
        f = open(procesed_data_path + 'Sesion{}_Sujeto{}.pkl'.format(sesion, 2), 'wb')
        pickle.dump([eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2], f)
        f.close()
        
        f = open(procesed_data_path + 'info.pkl', 'wb')
        pickle.dump(info, f)
        f.close()

    return eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1, eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2, info


class normalizar():
    
    def __init__(self, axis, porcent):
        self.axis = axis
        self.porcent = porcent
        
    def fit_normalize_eeg(self, matrix):
        # Defino el termino a agarrar
        self.n = int(self.porcent*len(matrix)/100)
        
        # De una
        # sorted_matrix = copy.deepcopy(matrix)
        # sorted_matrix.sort(self.axis)
        # self.minn_matrix = sorted_matrix[self.n]
        # matrix -= self.minn_matrix
        # self.maxn_matrix = sorted_matrix[-self.n]
        # matrix = (matrix - self.minn_matrix)/(self.maxn_matrix - self.minn_matrix)    
        
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
                   
    def normalize_eeg(self, matrix):
        #normalizo
        matrix -= self.minn_matrix
        matrix /= self.maxn_matrix 
        
    def normalize_stims(self, matrix):
        # Los estimulos los normalizo todos entre 0 y 1 estricto, la envolvente no tiene picos
        matrix -= np.min(matrix, axis = self.axis)   
        matrix /= np.max(matrix, axis = self.axis) 
        
class estandarizar():
    
    def __init__(self, axis, porcent):
        self.axis = axis
        self.porcent = porcent
        
    def fit_z_score_eeg(self, matrix):
        # Defino el termino a agarrar
        self.n = int(self.porcent*len(matrix)/100)
        
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
                   
    def z_score_eeg(self, matrix):
        #normalizo
        matrix -= self.minn_matrix
        matrix /= self.maxn_matrix 
        
    def z_score_stims(self, matrix):
        # Los estimulos los normalizo todos entre 0 y 1 estricto, la envolvente no tiene picos
        matrix -= np.min(matrix, axis = self.axis)   
        matrix /= np.max(matrix, axis = self.axis) 
    
    
def alphas_val(eeg_train_val, dstims_train_val, n_val_splits, train_percent, Normalizar_todo, n_best_channels, Plot_alphas, Prints_alphas):
    best_alphas_val = []

    all_index = np.arange(len(eeg_train_val))  
    for i in range(n_val_splits):
        
        train_index = all_index[round(i/(n_val_splits-1)*len(eeg_train_val)*(1-train_percent)):round((i/(n_val_splits-1)*(1-train_percent)+train_percent)*len(eeg_train_val))]
        val_index = np.array(list(set(all_index) - set(train_index)))
        
        eeg_train, eeg_val = eeg_train_val[train_index], eeg_train_val[val_index]
        dstims_train, dstims_val = dstims_train_val[train_index], dstims_train_val[val_index]
        
        if Normalizar_todo:
            ###### Normalizo tomando el valor del elemento al 95% mas alto de cada canal ######
            # (Deje de estandarizar el dstims en preproc) 
            axis = 0
            porcent = 5
            norm = normalizar(axis, porcent)
            
            norm.fit_normalize_eeg(eeg_train)
            norm.normalize_eeg(eeg_val)

            norm.normalize_stims(dstims_train)
            norm.normalize_stims(dstims_val)
        
        ###### Busco en grueso hiperparametro alpha ######
        n_best_channels = 16
        min_busqueda = 0
        max_busqueda = 3
        best_alpha_overall, pesos, intercepts, lista_Rmse = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = False)
        
        ###### refino busqeuda de alpha cerca del encontrado #######
        if best_alpha_overall == 1e3:
            min_busqueda,max_busqueda = 2,3
        else:
            min_busqueda,max_busqueda = np.log10(best_alpha_overall)-1,np.log10(best_alpha_overall)+1
        
        best_alpha_val, pesos, intercepts, lista_Rmse  = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = True) 
        best_alphas_val.append(best_alpha_val)
    return best_alphas_val    


def buscar_alpha(tStim, tResp, rStim, rResp, Plot_alphas, Prints, min_busqueda, max_busqueda, n_best_channels, fino):
    
    if not fino: 
        pasos = 13
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    else: 
        pasos = 50 
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    
    correlaciones = []
    pesos = []
    intercepts = []
    lista_Rmse = []
    corrmin = 0.1
    
    for alfa in alphas:
        mod = linear_model.Ridge(alpha=alfa, random_state=123)
        mod.fit(tStim,tResp) ## entreno el modelo
        predicho = mod.predict(rStim) ## testeo en ridge set
        Rcorr = np.array([np.corrcoef(rResp[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(rResp.shape[1])])
        Rmse = np.array(np.sqrt(np.power((predicho - rResp),2).mean(0)))
        lista_Rmse.append(np.array(Rmse))
        Rcorr[np.isnan(Rcorr)] = 0
        correlaciones.append(Rcorr)
        pesos.append(mod.coef_)
        intercepts.append(mod.intercept_)
        if Prints: print("Training: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f, over-under(%0.2f)=%d"%(alfa, np.mean(Rcorr), np.max(Rcorr), corrmin, (Rcorr>corrmin).sum()-( -Rcorr>corrmin).sum()))
    correlaciones_abs = np.array(np.abs(correlaciones))
    lista_Rmse = np.array(lista_Rmse)
    
    correlaciones_abs.sort()
    best_alpha_index = correlaciones_abs[:,-n_best_channels:].mean(1).argmax()
    best_alpha = alphas[best_alpha_index]
    
    if Prints:                     
        print("El mejor alfa es de: ", best_alpha)
        print("Con una correlación media de: ",correlaciones[best_alpha_index].mean())
        print("Y una correlación máxima de: ",correlaciones[best_alpha_index].max())
    
    if Plot_alphas: 
        if fino: plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 1, fino = True)
        else: plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 2, fino = False)
        
    return best_alpha, pesos, intercepts, lista_Rmse


def plot_alphas(alphas, correlaciones, best_alpha_overall, lista_Rmse, linea, fino):
    # Plot correlations vs. alpha regularization value
    ## cada linea es un canal
    fig=plt.figure(figsize=(10,5))
    fig.clf()
    plt.subplot(1,3,1)
    plt.subplots_adjust(wspace = 1 )
    plt.plot(alphas,correlaciones,'k')
    plt.gca().set_xscale('log')
    ## en rojo: el maximo de las correlaciones
    ## la linea azul marca el mejor alfa
    
    plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
    plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
    
    plt.plot(alphas,correlaciones.mean(1),'.r',linewidth=5)
    plt.xlabel('Alfa', fontsize=16)
    plt.ylabel('Correlación - Ridge set', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    
    ### como se ve sola la correlacion maxima para los distintos alfas
    plt.subplot(1,3,2)
    plt.plot(alphas,np.array(correlaciones).mean(1),'.r',linewidth=5)     
    plt.plot(alphas,np.array(correlaciones).mean(1),'-r',linewidth=linea)     
    
    if fino: 
        plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
        plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
        
    plt.xlabel('Alfa', fontsize=16)
    plt.gca().set_xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    ### el RMSE
    plt.subplot(1,3,3)
    plt.plot(alphas,np.array(lista_Rmse).min(1),'.r',linewidth=5) 
    plt.plot(alphas,np.array(lista_Rmse).min(1),'-r',linewidth=2) 
    
    if fino: 
        plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
        plt.plot([best_alpha_overall, best_alpha_overall],[plt.ylim()[0],plt.ylim()[1]])
        
    plt.xlabel('Alfa', fontsize=16)
    plt.ylabel('RMSE - Ridge set', fontsize=16)
    plt.gca().set_xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    
    titulo = "El mejor alfa es de: "+ str(best_alpha_overall)
    plt.suptitle(titulo, fontsize=18)

def simular_iteraciones(best_alpha, iteraciones, sesion, sujeto, test_round, dstims_train_val, eeg_train_val, dstims_test, eeg_test, Correlaciones_fake, Errores_fake, Save_iterations, Path_it):
    
    mod_fake = linear_model.Ridge(alpha = best_alpha, random_state=123)
    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, test_round + 1))
    for iteracion in np.arange(iteraciones):
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)
        
        mod_fake.fit(dstims_train_random, eeg_train_val) ## entreno el modelo
        
        ###### TESTEO EN VAL SET  ######
        # Predigo
        dstims_test_random = copy.deepcopy(dstims_test)
        np.random.shuffle(dstims_test_random)
        predicho_fake = mod_fake.predict(dstims_test_random)
        
        # Correlacion
        Rcorr_fake = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicho_fake[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])
        Correlaciones_fake[test_round, iteracion] = Rcorr_fake
        
        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test),2).mean(0)))
        Errores_fake[test_round, iteracion] = Rmse_fake
        
        print("\rProgress: {}%".format(int((iteracion+1)*100/iteraciones)), end='')
        
    if Save_iterations:
        try: os.makedirs(Path_it)
        except: pass                      
        f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
        pickle.dump([Correlaciones_fake, Errores_fake], f)
        f.close()

def plot_cabezas_canales(channel_names, info, sr, sesion, sujeto, Canales_sobrevivientes, Valores_promedio_abs, Display_cabezas_canales, n_canales, name, Save_cabezas_canales, Run_graficos_path):
    
    surviving_channels_names = [channel_names[j] for j in Canales_sobrevivientes]  
    mask = []
    for j in range(len(channel_names)):
        if channel_names[j] in (surviving_channels_names): mask.append(True)
        else: mask.append(False)
    
    if Display_cabezas_canales: plt.ion() 
    else: plt.ioff()
    
    ###### Grafico cabezas Correlaciones ######
    fig, axs = plt.subplots(1,2)
    plt.suptitle("Sesion{} Sujeto{}".format(sesion,sujeto))
    im = mne.viz.plot_topomap(Valores_promedio_abs, info, axes = axs[0], show = False, sphere = 0.07, 
                              cmap = 'Reds', 
                              vmin = Valores_promedio_abs.min(), vmax = Valores_promedio_abs.max())
    im2 = mne.viz.plot_topomap(np.zeros(n_canales), info, axes = axs[1], show = False, sphere = 0.07,
                               mask = np.array(mask), mask_params = dict(marker='o', markerfacecolor='g', 
                                                                         markeredgecolor='k', linewidth=0,
                                                                         markersize=4))
    # fig.tight_layout()
    plt.colorbar(im[0], ax = [axs[0], axs[1]], shrink = 0.85, label = name, orientation = 'horizontal', 
                 boundaries = np.linspace(Valores_promedio_abs.min().round(decimals = 3), 
                                          Valores_promedio_abs.max().round(decimals = 3), 100), 
                 ticks = [np.linspace(Valores_promedio_abs.min(), 
                                     Valores_promedio_abs.max(), 9).round(decimals = 3)])       
   
    if Save_cabezas_canales: 
        save_path_cabezas = Run_graficos_path + 'Cabezas_canales/'
        try: os.makedirs(save_path_cabezas)
        except: pass
        fig.savefig(save_path_cabezas + '{}_Cabeza_Sesion{}_Sujeto{}.png'.format(name,sesion,sujeto))


def plot_grafico_pesos_significativos(Display_figures_beta, sesion, sujeto, best_alpha, Pesos_promedio, 
                       Canales_sobrevivientes_corr, info, sr, 
                       Corr_promedio_abs, Rmse_promedio, Canales_sobrevivientes_rmse, 
                       Save_grafico_betas, Run_graficos_path, 
                       Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal,
                       Errores_fake, Correlaciones_fake):
    
    # Defino cosas que voy a graficar
    mejor_canal_corr = Corr_promedio_abs.argmax()
    Corr_mejor_canal = Corr_promedio_abs[mejor_canal_corr]
    # Correl_prom = np.mean(Corr_promedio_abs[Canales_sobrevivientes_corr])
    
    mejor_canal_rmse = Rmse_promedio.argmax()
    Rmse_mejor_canal = Rmse_promedio[mejor_canal_rmse]
    # Rmse_prom = np.mean(Rmse_promedio[Canales_sobrevivientes_rmse])
    
    # Errores_fake_mean = Errores_fake.mean(1).mean(0)
    Errores_fake_min = Errores_fake.min(1).min(0)
    Errores_fake_max = Errores_fake.max(1).max(0)
    
    # Correlaciones_fake_mean = Correlaciones_fake.mean(1).mean(0)
    Correlaciones_fake_min = abs(Correlaciones_fake).min(1).min(0)
    Correlaciones_fake_max = abs(Correlaciones_fake).max(1).max(0)
    
    if Display_figures_beta: plt.ion() 
    else: plt.ioff()
    
    fig, axs = plt.subplots(3,1,figsize=(10,9.5))
    fig.suptitle('Sesion {} - Sujeto {} - Corr max {:.2f} - Rmse max {:.2f}- alpha: {:.2f}'.format(sesion, sujeto, Corr_mejor_canal, Rmse_mejor_canal, best_alpha))

    if Pesos_promedio[Canales_sobrevivientes_corr].size:
        # plt.plot(np.arange(0, Pesos_promedio.shape[1]/sr, 1.0/sr), Pesos_promedio[Canales_sobrevivientes_corr].transpose(),"-")
        
        evoked = mne.EvokedArray(Pesos_promedio, info)
        evoked.plot(picks = Canales_sobrevivientes_corr, show = False, spatial_colors=True, 
                    scalings = dict(eeg=1, grad=1, mag=1), unit = False, units = dict(eeg = 'w'), 
                    time_unit = 'ms', axes = axs[0], zorder = 'std')
        axs[0].plot(np.arange(0, Pesos_promedio.shape[1]/sr*1000, 1000/sr), 
                    Pesos_promedio[Canales_sobrevivientes_corr].mean(0),'k--', 
                    label = 'Mean', zorder = 130, linewidth = 1.5)
        
        ticks =  np.array(list(evoked.times[-1]*1000 - axs[0].get_xticks()[:-1]))
        axs[0].set_xticks(list(ticks))
        axs[0].set_xticklabels(list((ticks-evoked.times[-1]*1000).astype(int)))
        axs[0].xaxis.label.set_size(13)
        axs[0].yaxis.label.set_size(13)
        axs[0].legend(fontsize = 13)
        axs[0].grid()

    else: 
        plt.text(0.5,0.5, "No surviving channels", size = 'xx-large', ha = 'center')
    
    axs[1].plot(Corr_promedio_abs, '.', color = 'C0', label = "Promedio de Correlaciones (Descartados)")
    axs[1].plot(Canales_sobrevivientes_corr, Corr_promedio_abs[Canales_sobrevivientes_corr], '*', 
                color = 'C1', label = "Promedio de Correlaciones (Pasan test)")
    # axs[1].hlines(Correl_prom, axs[1].get_xlim()[0], axs[1].get_xlim()[1], label = 'Promedio = {:0.2f}'.format(Correl_prom), color = 'C3')
    # axs[1].plot(Correlaciones_fake_mean, color = 'C3', linewidth = 0.5)
    axs[1].fill_between(np.arange(len(Corr_promedio_abs)), abs(Corr_buenas_ronda_canal).min(0), 
                                  abs(Corr_buenas_ronda_canal).max(0), alpha = 0.5, 
                                  label = 'Distribución de Corr (Stims reales)')
    axs[1].fill_between(np.arange(len(Corr_promedio_abs)), abs(Correlaciones_fake_min), 
                                  abs(Correlaciones_fake_max), alpha = 0.5, 
                                  label = 'Distribución de Corr (Stims rand)')
    axs[1].set_xlim([-1,129])
    axs[1].set_xlabel('Canales')
    axs[1].set_ylabel('Correlación')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(Rmse_promedio, '.', color = 'C0', label = "Promedio de Errores (Descartados)")
    axs[2].plot(Canales_sobrevivientes_rmse, Rmse_promedio[Canales_sobrevivientes_rmse], '*', 
                color = 'C1', label = "Promedio de Errores (Pasan test)")
    # axs[2].hlines(Rmse_prom, axs[2].get_xlim()[0],axs[2].get_xlim()[1], label = 'Promedio = {:0.2f}'.format(Rmse_prom), color = 'C3')
    # axs[2].plot(Errores_fake_mean, color = 'C3', linewidth = 0.5)
    axs[2].fill_between(np.arange(len(Rmse_promedio)), abs(Rmse_buenos_ronda_canal.min(0)), 
                                  abs(Rmse_buenos_ronda_canal.max(0)), alpha = 0.5, 
                                  label = 'Distribución de Rmse (Stims reales)', color = 'C0')
    axs[2].fill_between(np.arange(len(Rmse_promedio)), abs(Errores_fake_min), 
                                  abs(Errores_fake_max), alpha = 0.5, 
                                  label = 'Distribución de Rmse (Stims rand)', color = 'C1')

    axs[2].set_xlabel('Canales')
    axs[2].set_ylabel('Rmse')
    axs[2].legend()
    axs[2].grid()    
    
    fig.tight_layout() 
    
    if Save_grafico_betas: 
        save_path_graficos = Run_graficos_path + 'Betas_alpha_forced/'
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion, sujeto))
        
def plot_grafico_pesos_todos(Display_figures_beta, sesion, sujeto, best_alpha, Pesos_promedio, 
                            Canales_sobrevivientes_corr, info, sr, 
                            Corr_promedio_abs, Rmse_promedio, Canales_sobrevivientes_rmse, 
                            Save_grafico_betas, Run_graficos_path, 
                            Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal,
                            Errores_fake, Correlaciones_fake):
    
    # Defino cosas que voy a graficar
    mejor_canal_corr = Corr_promedio_abs.argmax()
    Corr_mejor_canal = Corr_promedio_abs[mejor_canal_corr]
    # Correl_prom = np.mean(Corr_promedio_abs)
    
    mejor_canal_rmse = Rmse_promedio.argmax()
    Rmse_mejor_canal = Rmse_promedio[mejor_canal_rmse]
    # Rmse_prom = np.mean(Rmse_promedio)
    
    # Errores_fake_mean = Errores_fake.mean(1).mean(0)
    Errores_fake_min = Errores_fake.min(1).min(0)
    Errores_fake_max = Errores_fake.max(1).max(0)
    
    # Correlaciones_fake_mean = Correlaciones_fake.mean(1).mean(0)
    Correlaciones_fake_min = abs(Correlaciones_fake).min(1).min(0)
    Correlaciones_fake_max = abs(Correlaciones_fake).max(1).max(0)
    
    if Display_figures_beta: plt.ion() 
    else: plt.ioff()
    
    fig, axs = plt.subplots(3,1,figsize=(10,9.5))
    fig.suptitle('Sesion {} - Sujeto {} - Corr max {:.2f} - Rmse max {:.2f}- alpha: {:.2f}'.format(sesion, sujeto, Corr_mejor_canal, Rmse_mejor_canal, best_alpha))

    if Pesos_promedio.size:
        # plt.plot(np.arange(0, Pesos_promedio.shape[1]/sr, 1.0/sr), Pesos_promedio[Canales_sobrevivientes_corr].transpose(),"-")
        
        evoked = mne.EvokedArray(Pesos_promedio, info)
        evoked.plot(scalings = dict(eeg=1, grad=1, mag=1), zorder = 'std', time_unit = 'ms', 
                    show = False, spatial_colors=True, unit = False, units = 'w', 
                    axes = axs[0])
        
        axs[0].plot(np.arange(0, Pesos_promedio.shape[1]/sr*1000, 1000/sr), 
                        Pesos_promedio.mean(0),'k--', label = 'Mean', zorder = 130, linewidth = 2)
        
        ticks =  np.array(list(evoked.times[-1]*1000 - axs[0].get_xticks()[:-1]))
        axs[0].set_xticks(list(ticks))
        axs[0].set_xticklabels(list((ticks-evoked.times[-1]*1000).astype(int)))
        # axs[0].set_xlabel('Time [ms]')
        axs[0].xaxis.label.set_size(13)
        axs[0].yaxis.label.set_size(13)
        axs[0].legend(fontsize = 13)
        axs[0].grid()


    else: 
        plt.text(0.5,0.5, "No surviving channels", size = 'xx-large', ha = 'center')
    
    axs[1].plot(Corr_promedio_abs, '.', color = 'C0', label = "Promedio de Correlaciones (Descartados)")
    axs[1].plot(Canales_sobrevivientes_corr, Corr_promedio_abs[Canales_sobrevivientes_corr], '*', 
                color = 'C1', label = "Promedio de Correlaciones (Pasan test)")
    # axs[1].hlines(Correl_prom, axs[1].get_xlim()[0], axs[1].get_xlim()[1], label = 'Promedio = {:0.2f}'.format(Correl_prom), color = 'C3')
    # axs[1].plot(Correlaciones_fake_mean, color = 'C3', linewidth = 0.5)
    axs[1].fill_between(np.arange(len(Corr_promedio_abs)), abs(Corr_buenas_ronda_canal).min(0), 
                                  abs(Corr_buenas_ronda_canal).max(0), alpha = 0.5, 
                                  label = 'Distribución de Corr (Stims reales)')
    axs[1].fill_between(np.arange(len(Corr_promedio_abs)), abs(Correlaciones_fake_min), 
                                  abs(Correlaciones_fake_max), alpha = 0.5, 
                                  label = 'Distribución de Corr (Stims rand)')
    axs[1].set_xlim([-1,129])
    axs[1].set_xlabel('Canales')
    axs[1].set_ylabel('Correlación')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(Rmse_promedio, '.', color = 'C0', label = "Promedio de Errores (Descartados)")
    axs[2].plot(Canales_sobrevivientes_rmse, Rmse_promedio[Canales_sobrevivientes_rmse], '*', 
                color = 'C1', label = "Promedio de Errores (Pasan test)")
    # axs[2].hlines(Rmse_prom, axs[2].get_xlim()[0],axs[2].get_xlim()[1], label = 'Promedio = {:0.2f}'.format(Rmse_prom), color = 'C3')
    # axs[2].plot(Errores_fake_mean, color = 'C3', linewidth = 0.5)
    axs[2].fill_between(np.arange(len(Rmse_promedio)), abs(Rmse_buenos_ronda_canal.min(0)), 
                                  abs(Rmse_buenos_ronda_canal.max(0)), alpha = 0.5, 
                                  label = 'Distribución de Rmse (Stims reales)', color = 'C0')
    axs[2].fill_between(np.arange(len(Rmse_promedio)), abs(Errores_fake_min), 
                                  abs(Errores_fake_max), alpha = 0.5, 
                                  label = 'Distribución de Rmse (Stims rand)', color = 'C1')

    axs[2].set_xlabel('Canales')
    axs[2].set_ylabel('Rmse')
    axs[2].legend()
    axs[2].grid()    
    
    fig.tight_layout() 
    
    if Save_grafico_betas: 
        save_path_graficos = Run_graficos_path + 'Betas_alpha_forced/'
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion, sujeto))
        
def plot_grafico_shadows_mne(channel_names, Display_figures_shadows, sesion, sujeto, best_alpha, 
                         Canales_sobrevivientes_corr, info, sr, 
                         Corr_promedio_abs, Save_grafico_shadows, Run_graficos_path, 
                         Corr_buenas_ronda_canal, Correlaciones_fake):
    
    evoked = mne.EvokedArray(np.zeros((128,67)), info)
    surviving_channels_names = [channel_names[j] for j in Canales_sobrevivientes_corr]
    mask = []
    for j in range(len(channel_names)):
        if channel_names[j] in (surviving_channels_names): mask.append(True)
        else: mask.append(False)
        
    # Defino cosas que voy a graficar
    mejor_canal_corr = Corr_promedio_abs.argmax()
    Corr_mejor_canal = Corr_promedio_abs[mejor_canal_corr]
    
    Correlaciones_fake_min = abs(Correlaciones_fake).min(1).min(0)
    Correlaciones_fake_max = abs(Correlaciones_fake).max(1).max(0)
    
    if Display_figures_shadows: plt.ion() 
    else: plt.ioff()
    
    fig, ax = plt.subplots(1,1,figsize=(10,7))
    fig.suptitle('Session {} - Subject {}'.format(sesion, sujeto))
    
    if len(Canales_sobrevivientes_corr):
        evoked.plot(picks = Canales_sobrevivientes_corr, show = False, spatial_colors=True, 
                    scalings = dict(eeg=1, grad=1, mag=1), unit = True, units = dict(eeg = '$w$'), 
                    axes = ax, zorder = 'unsorted')
        ax.plot(Corr_promedio_abs, '.', color = 'C0', label = "Mean of correlations among folds (Discarded)")
        ax.plot(Canales_sobrevivientes_corr, Corr_promedio_abs[Canales_sobrevivientes_corr], '*', 
                    color = 'C1', label = "Mean of correlations among folds (Test passed)")
        
        ax.fill_between(np.arange(len(Corr_promedio_abs)), abs(Corr_buenas_ronda_canal).min(0), 
                                      abs(Corr_buenas_ronda_canal).max(0), alpha = 0.5, 
                                      label = 'Correlation distribution (Real data)')
        ax.fill_between(np.arange(len(Corr_promedio_abs)), abs(Correlaciones_fake_min), 
                                      abs(Correlaciones_fake_max), alpha = 0.5, 
                                      label = 'Correlation distribution (Random data)')
        ax.set_xlim([-1,129])
        ax.set_xlabel('Channels', fontsize = 15)
        ax.set_ylabel('|Correlation|', fontsize = 15)
        ax.legend(fontsize = 13, loc = "upper right")
        ax.grid()
        
        ax.xaxis.set_tick_params(labelsize = 13)
        ax.yaxis.set_tick_params(labelsize = 13)
        fig.tight_layout()
    else: 
        plt.text(0.5,0.5, "No surviving channels", size = 'xx-large', ha = 'center')
    
    if Save_grafico_shadows: 
        save_path_graficos = Run_graficos_path + 'Correlation_shadows/'
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion, sujeto))
        
def plot_grafico_shadows(channel_names, Display_figures_shadows, sesion, sujeto, best_alpha, 
                         Canales_sobrevivientes_corr, info, sr, 
                         Corr_promedio_abs, Save_grafico_shadows, Run_graficos_path, 
                         Corr_buenas_ronda_canal, Correlaciones_fake):
     
    # Defino cosas que voy a graficar  
    Correlaciones_fake_min = abs(Correlaciones_fake).min(1).min(0)
    Correlaciones_fake_max = abs(Correlaciones_fake).max(1).max(0)
    
    if Display_figures_shadows: plt.ion() 
    else: plt.ioff()
    
    fig, ax = plt.subplots(1,1,figsize=(10,7))
    fig.suptitle('Session {} - Subject {}'.format(sesion, sujeto))
    
    if len(Canales_sobrevivientes_corr):
        ax.plot(Corr_promedio_abs, '.', color = 'C0', label = "Mean of correlations among folds (Discarded)")
        ax.plot(Canales_sobrevivientes_corr, Corr_promedio_abs[Canales_sobrevivientes_corr], '*', 
                    color = 'C1', label = "Mean of correlations among folds (Test passed)")
        
        ax.fill_between(np.arange(len(Corr_promedio_abs)), abs(Corr_buenas_ronda_canal).min(0), 
                                      abs(Corr_buenas_ronda_canal).max(0), alpha = 0.5, 
                                      label = 'Correlation distribution (Real data)')
        ax.fill_between(np.arange(len(Corr_promedio_abs)), abs(Correlaciones_fake_min), 
                                      abs(Correlaciones_fake_max), alpha = 0.5, 
                                      label = 'Correlation distribution (Random data)')
        ax.set_xlim([-1,129])
        ax.set_xlabel('Channels', fontsize = 15)
        ax.set_ylabel('|Correlation|', fontsize = 15)
        ax.legend(fontsize = 13, loc = "upper right")
        ax.grid()
        
        ax.xaxis.set_tick_params(labelsize = 13)
        ax.yaxis.set_tick_params(labelsize = 13)
        fig.tight_layout()
    else: 
        plt.text(0.5,0.5, "No surviving channels", size = 'xx-large', ha = 'center')
    
    if Save_grafico_shadows: 
        save_path_graficos = Run_graficos_path + 'Correlation_shadows/'
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion, sujeto))


def Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_correlacion_promedio, Save_correlacion_promedio, Run_graficos_path):
    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)
    # Correlacion_promedio_total = np.mean(Correlaciones_promedio)
    # std_corr_prom_tot = np.std(Correlaciones_promedio)
    
    if Display_correlacion_promedio: plt.ion()
    else: plt.ioff()
    
    fig = plt.figure()
    plt.suptitle("Mean correlation per channel among subjects", fontsize = 18)
    im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap = 'Reds',
                              vmin = Correlaciones_promedio.min(), vmax = Correlaciones_promedio.max(), 
                              show = False, sphere = 0.07)
    plt.colorbar(im[0], shrink = 0.85, orientation = 'vertical')
    fig.tight_layout()
    
    if Save_correlacion_promedio: 
        save_path_graficos = Run_graficos_path
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Correlaciones_promedio.png')


def Cabezas_canales_rep(Canales_repetidos_sujetos, info, Display_canales_repetidos, Save_canales_repetidos, Run_graficos_path):
    if Display_canales_repetidos: plt.ion() 
    else: plt.ioff()
    
    fig = plt.figure()
    plt.suptitle("Channels passing 5 test per subject", fontsize = 18)
    im = mne.viz.plot_topomap(Canales_repetidos_sujetos, info, cmap = 'Reds',
                              vmin=1, vmax=10, 
                              show = False, sphere = 0.07)
    cb = plt.colorbar(im[0], shrink = 0.85, orientation = 'vertical')
    cb.set_label(label = 'Number of subjects passed', size = 15)
    fig.tight_layout()
    
    if Save_canales_repetidos: 
        save_path_graficos = Run_graficos_path
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Canales_repetidos_ronda.png')


def Plot_instantes_casera(Pesos_totales_sujetos_todos_canales, info, sr, delays, Display_figure_instantes, Save_figure_instantes, Run_graficos_path):  
     # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0,2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0)
    
    instantes_de_interes = [Pesos_totales_sujetos_todos_canales_copy.mean(1).argmin()/sr, 
                            Pesos_totales_sujetos_todos_canales_copy.mean(1).argmax()/sr]
    instantes_index = [int(sr*i) for i in instantes_de_interes]
    times = [-delays[-1]/sr - i for i in instantes_de_interes]
        
    # Ploteo pesos y cabezas
    if Display_figure_instantes: plt.ion() 
    else: plt.ioff()
    
    Blues = plt.cm.get_cmap('Blues').reversed()
    cmaps = [Blues, 'Reds']
    
    fig = plt.figure(figsize = (10,5))
    for i in range(len(instantes_de_interes)):
        ax = fig.add_subplot(2, len(instantes_de_interes), i+1) 
        ax.set_title('{} ms'.format(int(-times[i]*1000)))
        im = mne.viz.plot_topomap(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]], info, axes = ax, show = False, 
                                  sphere = 0.07, cmap = cmaps[i], 
                                  vmin = Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(), 
                                  vmax = Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max())   
        plt.colorbar(im[0], ax = ax, orientation = 'vertical', 
                        boundaries = np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min().round(decimals = 3), 
                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max().round(decimals = 3), 100), 
                        ticks = [np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(), 
                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 4).round(decimals = 3)])
        
    ax = fig.add_subplot(2,1,2)
    evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy.transpose(), info)
    
    evoked.plot(show = False, spatial_colors=True, scalings = dict(eeg=1, grad=1, mag=1), 
                unit = True, units = dict(eeg = '$w$'), axes = ax, zorder = 'unsorted', selectable = False, 
                time_unit = 'ms')
    ax.plot(np.arange(0, Pesos_totales_sujetos_todos_canales_copy.transpose().shape[1]*1000/sr, 
                                1000/sr), 
                        Pesos_totales_sujetos_todos_canales_copy.transpose().mean(0),
                        'k--', label = 'Mean', zorder = 130, linewidth = 2)
    
    ticks =  np.array(list(evoked.times[-1]*1000 - ax.get_xticks()[:-1]))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(list((ticks - (evoked.times[-1] + delays[-1]/sr)*1000).astype(int)))
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.grid()
    ax.legend(fontsize = 13)

    fig.tight_layout()      
    
    if Save_figure_instantes: 
        save_path_graficos = Run_graficos_path
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Instantes_interes.png')  
        
def Plot_instantes_interes(Pesos_totales_sujetos_todos_canales, info, sr, delays, Display_figure_instantes, 
                           Save_figure_instantes, Run_graficos_path):
    
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0,2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0)
    
    # Ploteo pesos y cabezas
    if Display_figure_instantes: plt.ion() 
    else: plt.ioff()
    
    evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy.transpose(), info)
    
    instantes_de_interes = [Pesos_totales_sujetos_todos_canales_copy.mean(1).argmin()/sr, 
                            Pesos_totales_sujetos_todos_canales_copy.mean(1).argmax()/sr]
    instantes_index = [int(sr*i) for i in instantes_de_interes]
    
    times = [(delays[-1]-delays[0])/sr + i for i in instantes_de_interes]
    
    fig = evoked.plot_joint(times = instantes_de_interes, title = 'Mean of $w$ among subjects', show = False, 
                            ts_args = dict(unit = 'False', units = dict(eeg='$w$', grad='fT/cm', mag='fT'),
                                           scalings = dict(eeg=1, grad=1, mag=1), zorder = 'std', time_unit = 'ms'),                  
                            topomap_args = dict(vmin = Pesos_totales_sujetos_todos_canales_copy.min(),
                                                vmax = Pesos_totales_sujetos_todos_canales_copy.max(),
                                                time_unit = 'ms'))
    fig.set_size_inches(10,7)
    axs = fig.axes
    axs[0].plot(np.arange(0, Pesos_totales_sujetos_todos_canales_copy.shape[0]*1000/sr, 
                                1000/sr), Pesos_totales_sujetos_todos_canales_copy.mean(1),
                        'k--', label = 'Mean', zorder = 130, linewidth = 2)
    
    
    ticks =  np.array(list(evoked.times[-1]*1000 - axs[0].get_xticks()[:-1]))
    axs[0].set_xticks(list(ticks))
    axs[0].set_xticklabels(list((ticks - evoked.times[-1]*1000).astype(int)))
    axs[0].xaxis.label.set_size(13)
    axs[0].yaxis.label.set_size(13)
    axs[0].grid()
    axs[0].legend(fontsize = 13, loc = 'upper right')
    
    Blues = plt.cm.get_cmap('Blues').reversed()
    cmaps = [Blues, 'Reds']
    
    for i in range(len(instantes_de_interes)):
        axs[i+1].clear()
        axs[i+1].set_title('{} ms'.format(int(times[i]*1000)), fontsize = 13)
        im = mne.viz.plot_topomap(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]], info, axes = axs[i+1], 
                                  show = False, sphere = 0.07, cmap = cmaps[i], 
                                  vmin = Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(), 
                                  vmax = Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max())   
        plt.colorbar(im[0], ax = axs[i+1], orientation = 'vertical', 
                        boundaries = np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min().round(decimals = 3), 
                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max().round(decimals = 3), 100), 
                        ticks = [np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(), 
                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 4).round(decimals = 3)])
    
    axs[3].remove()
    axs[5].remove()
    
    if Save_figure_instantes: 
        save_path_graficos = Run_graficos_path
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Instantes_interes.png')  
        
    return Pesos_totales_sujetos_todos_canales_copy.mean(1)
        
def pearsonr_pval(x,y):
    return stats.pearsonr(x,y)[1]

def Matriz_corr(Pesos_totales_sujetos_promedio, Pesos_totales_sujetos_todos_canales, sujeto_total, Display_correlation_matrix, Save_correlation_matrix, Run_graficos_path):
    # Armo df para correlacionar
    Pesos_totales_sujetos_promedio = Pesos_totales_sujetos_promedio[:sujeto_total]
    Pesos_totales_sujetos_promedio.append(Pesos_totales_sujetos_todos_canales.transpose().mean(0).mean(1)) # agrego pesos promedio
    lista_nombres = ["1","2","3","4","5","6","7","8","9","10","Promedio"]
    Pesos_totales_sujetos_df = pd.DataFrame(Pesos_totales_sujetos_promedio).transpose()
    Pesos_totales_sujetos_df.columns = lista_nombres[:len(Pesos_totales_sujetos_df.columns)-1]+[lista_nombres[-1]]
    
    pvals_matrix = Pesos_totales_sujetos_df.corr(method=pearsonr_pval)
    Correlation_matrix = np.array(Pesos_totales_sujetos_df.corr(method = 'pearson'))
    for i in range(len(Correlation_matrix)):
        Correlation_matrix[i,i] = Correlation_matrix[-1,i]
        
    Correlation_matrix = pd.DataFrame(Correlation_matrix[:-1,:-1])
    Correlation_matrix.columns = lista_nombres[:len(Correlation_matrix)-1]+[lista_nombres[-1]]
    
    if Display_correlation_matrix: plt.ion()
    else: plt.ioff()
    
    mask = np.ones_like(Correlation_matrix)
    mask[np.tril_indices_from(mask)] = False
    Reds = plt.cm.get_cmap('Reds').reversed()
    
    fig, (ax,cax) = plt.subplots(ncols = 2,figsize = (15,9), gridspec_kw={"width_ratios":[1, 0.05]})
    fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize = 26)
    sn.heatmap(abs(Correlation_matrix), mask = mask, cmap = "coolwarm", fmt='.3', ax = ax, 
               annot=True, center = 0, xticklabels = True, annot_kws={"size": 19},
               cbar = False)
    
    ax.set_yticklabels(['Mean of subjects']+lista_nombres[1:len(Correlation_matrix)], rotation = 'horizontal', fontsize = 19)
    ax.set_xticklabels(lista_nombres[:len(Correlation_matrix)-1]+['Mean of subjects'], rotation = 'horizontal', ha = 'left', fontsize = 19)
    
    sn.despine(right=True, left=True, bottom=True, top = True)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
    cax.yaxis.set_tick_params(labelsize = 20)
    
    fig.tight_layout()
        
    if Save_correlation_matrix: 
        save_path_graficos = Run_graficos_path
        try: os.makedirs(save_path_graficos)
        except: pass
        fig.savefig(save_path_graficos + 'Correlation_matrix.png')


#########   FIN FUNCIONES   ############
def load_eeg_data(s, trial, canal_oyente, sr):
    ubi = "Datos/EEG/S"+str(s)+"/s"+str(s)+"-"+str(canal_oyente)+"-Trial"+str(trial)+"-Deci-Filter-Trim-ICA-Pruned.set"
    eeg = mne.io.read_raw_eeglab(ubi)
    # eeg_freq = eeg.info.get("sfreq")
    eeg.load_data()
    
    eeg = eeg.filter(l_freq = None, h_freq = 40)
    return eeg

def juntar_trials_cortos(eeg_sujeto, dstims_sujeto, n):    
    largo_minimo = sum(len(eeg_trial) for eeg_trial in eeg_sujeto)/len(eeg_sujeto)/n
    eeg_sujeto.sort(key = len)
    dstims_sujeto.sort(key = len)
    
    correct_length = True
    while correct_length:
        if len(eeg_sujeto[0]) < largo_minimo:
            eeg_sujeto[1] = np.vstack((eeg_sujeto[0], eeg_sujeto[1]))
            dstims_sujeto[1] = np.vstack((dstims_sujeto[0], dstims_sujeto[1]))
            del eeg_sujeto[0]
            del dstims_sujeto[0]
            eeg_sujeto.sort(key = len)
            dstims_sujeto.sort(key = len)
        else: correct_length = False

def separar_datos(dstims, eeg, porcentaje_train, porcentaje_ridge, porcentaje_val):
    #DIVIDO TRAIN RIDGE Y TEST 
    nt = dstims.shape[0]
    
    # Training indices
    train_inds = np.arange(int(nt*porcentaje_train))    
    # Ridge indices
    ridge_inds = np.arange(int(nt*porcentaje_train),int(nt*(porcentaje_train + porcentaje_ridge)))   
    # Validation indices
    val_inds = np.arange(int(nt*(porcentaje_train + porcentaje_ridge)),nt)
    
    print("Delayed stimulus matrix has dimensions", dstims.shape)      
    # Training
    tStim = dstims[train_inds]
    tResp = eeg[train_inds]  
    # Ridge
    rStim = dstims[ridge_inds]
    rResp = eeg[ridge_inds]    
    # Validation
    vStim = dstims[val_inds]
    vResp = eeg[val_inds]
    
    return tStim, tResp, rStim, rResp, vStim, vResp

def comparar_preproc_trial(eeg, momentos_escucha, Diferencia, delay_time, sr):
    ## Comparo == 2 vs fila ==2
    eeg_Mauro = eeg[momentos_escucha == 2,:]  
    delays = np.arange(np.floor(delay_time*sr), dtype=int) 
    momentos_escucha_matriz = matriz_shifteada(momentos_escucha,delays)
    keep_indexes = []
    for j in range(len(momentos_escucha_matriz)):
        x = momentos_escucha_matriz[j] == 2 
        if x.all() == True:
            keep_indexes.append(j)           
    eeg_joaco = eeg[keep_indexes,:]   
    Diferencia.append(len(eeg_joaco)*100/len(eeg_Mauro))
        
def plot_envs(momentos_escucha, trial, enve):
    momentos_escucha_1 = momentos_escucha == 1    
    momentos_escucha_2 = momentos_escucha == 2
    
    fig = plt.figure()
    fig.clf()

    plt.subplot(3,1,1)
    plt.title('Trial {}'.format(trial))
    plt.plot(enve[:,0], label = 'Envelope')
    plt.legend()
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.title('Momentos escucha = 2')
    plt.plot(momentos_escucha_2, color = 'C1', label = 'interlocutor')
    plt.legend()
    plt.grid()

    plt.subplot(3,1,3)
    plt.title('Momentos escucha = 1')
    plt.plot(momentos_escucha_1, color = 'C2', label = 'sujeto')
    plt.legend()
    plt.grid()    
    
    fig.tight_layout()
