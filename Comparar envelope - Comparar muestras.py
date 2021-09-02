# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:37:11 2021

@author: joaco
"""
#%% COMPARAR MUESTRAS PREPROCESAMIENTO

import numpy as np
import matplotlib.pyplot as plt
import pickle

import Funciones as func

Save_comparar = False
Diferencia = []

sr = 128
delay_time = 0.3

sesiones = np.arange(21,22)
for sesion in sesiones:    
    run = True
    i=0

    while run:
        trial = i+1
        
        if trial % 2: 
            canal_hablante = 1
        else:
            canal_hablante = 2      
        canal_oyente = (canal_hablante - 3)*(-1)
                
        try: 
            eeg_trial = func.f_eeg(sesion,trial, canal_oyente, sr, delay_time)
            dstims_trial = func.f_envelope(sesion,trial, canal_hablante, sr, delay_time)
            momentos_escucha = func.solo_escucha(sesion,trial, canal_hablante, sr, delay_time)
        except: 
            run = False
            
        if run: 
            eeg_trial, dstims_trial, momentos_escucha = func.igualar_largos(eeg = eeg_trial, enve = dstims_trial, momentos_escucha = momentos_escucha, delay_time = delay_time, sr = sr)
            func.comparar_preproc_trial(eeg = eeg_trial, momentos_escucha = momentos_escucha, Diferencia = Diferencia, delay_time = delay_time, sr = sr)
            i += 1

            
Promedio = np.mean(Diferencia)
plt.figure()
plt.grid()
plt.plot(Diferencia, label = '% de muestras')
plt.hlines(Promedio, plt.xlim()[0],plt.xlim()[1], label = 'Promedio = {:0.2f}'.format(Promedio), color = 'C1')
plt.legend()

if Save_comparar: 
    plt.savefig('gráficos/Comparacion_Preproc.png')

    f = open('Dif_Prom.pkl', 'wb')
    pickle.dump([Diferencia, Promedio], f)
    f.close()
    
    # f = open('Dif_Prom.pkl', 'rb')
    # Diferencia, Promedio = pickle.load(f)
    # f.close()
    
    
#%%% PLOTEAR ENVELOPES Y VECTOR HABLANTES

import numpy as np
import matplotlib.pyplot as plt
import Funciones as func

sr = 128
delay_time = 0.3

sesiones = np.arange(21,22)
for sesion in sesiones:    
    run = True
    i=0

    for i in np.arange(2):
    # while run:
        trial = i+1   
        # if False: 
        if trial % 2:
            canal_hablante = 1
        else:
            canal_hablante = 2      
        canal_oyente = (canal_hablante - 3)*(-1)
                
        try: 
            eeg_trial = func.f_eeg(sesion,trial, canal_oyente, sr, delay_time)
            dstims_trial = func.f_envelope(sesion,trial, canal_hablante, sr, delay_time)
            momentos_escucha = func.solo_escucha(sesion,trial, canal_hablante, sr, delay_time)
        except: 
            run = False
            
        if run: 
            ###### LIMPIAR DATOS ######
            eeg_trial, dstims_trial, momentos_escucha = func.igualar_largos(eeg = eeg_trial, enve = dstims_trial, momentos_escucha = momentos_escucha, delay_time = delay_time, sr = sr)     
            # eeg_trial, dstims_trial = func.preproc(preproc_mauro_mal = False, preproc_fila = True, eeg = eeg_trial, enve = dstims_trial, momentos_escucha = momentos_escucha, delay_time = delay_time, sr = sr)
            
            #### PLOT ENVOLVENTE Y HABLANTES
            func.plot_envs(momentos_escucha = momentos_escucha, trial = trial, enve = dstims_trial)
         
        i += 1
       