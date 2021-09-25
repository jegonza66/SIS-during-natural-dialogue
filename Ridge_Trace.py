# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:46:03 2021

@author: joaco
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
import os
import pickle
import Processing
import Plot
import Load

individual_trace = False
Display_figures_Trace = True
Save_figures_Trace = False

Display_figure_instantes, Save_figure_instantes = True, False

# Stims = ['Envelope', 'Pitch', 'Pitch_der', 'Envelope_Pitch_Pitch_der']
# Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
###### DEFINO PARAMETROS ######
stim = 'Envelope'
Stims_Order = ['Envelope','Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
###### Defino banda de eeg ######
Band = 'Delta'
###### Defino situacion de interes ######
situacion = 'Escucha'
###### Defino estandarizacion
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
# Defino tiempos
sr = 128
n_canales = 128
tmin, tmax = -0.53, 0.1
delays = - np.arange(np.floor(tmin*sr), np.ceil(tmax*sr), dtype=int)
times = np.linspace(delays[0]*np.sign(tmin)*1/sr, np.abs(delays[-1])*np.sign(tmax)*1/sr, len(delays))

###### Paths ######
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin,tmax)
Run_graficos_path = 'gráficos/Ridge_Trace/'

min_busqueda, max_busqueda = -2, 6
pasos = 9
alphas = np.logspace(min_busqueda, max_busqueda, pasos)

Standarized_Betas = np.zeros((len(alphas),len(times)))
Pesos_promedio_saves = np.zeros((len(alphas),len(times)))

# Alphas = []
sesiones = np.arange(21, 26)
###### Empiezo corrida ######
for alpha_num, alpha_forzado in enumerate(alphas):
    print('Alpha: {}'.format(alpha_forzado))
    sujeto_total = 0
    for sesion in sesiones:
        print('Sesion {}'.format(sesion))
        ##### LOAD DATA BY SUBJECT #####
        Sujeto_1, Sujeto_2 = Load.Load_Data(sesion, Band, sr, tmin, tmax, situacion, procesed_data_path, sujeto_total)
        
        ##### LOAD EEG BY SUBJECT #####
        eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']
        
        ##### LOAD STIMULUS BY SUBJECT #####
        dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
        Cant_Estimulos = len(dstims_para_sujeto_1)
        
        for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
        # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):    
            print('Sujeto {}'.format(sujeto))
            
            ###### Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)######
            Predicciones = {}
            n_splits = 5
            iteraciones = 3000
              
            ###### Defino variables donde voy a guardar mil cosas ######
            Pesos_ronda_canales = np.zeros((n_splits, n_canales, len(delays)*Cant_Estimulos))
                               
            ###### Empiezo el KFold de test ######
            kf_test = KFold(n_splits, shuffle = False)
            for test_round, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]
                
                dstims_train_val = list()
                dstims_test = list()
    
                for stimulus in list(dstims):
                    dstims_train_val.append(stimulus[train_val_index])
                    dstims_test.append(stimulus[test_index])
                    
                axis = 0
                porcent = 5
                eeg, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg, dstims_train_val, dstims_test, Stims_preprocess, EEG_preprocess, axis = 0, porcent = 5)
                
                ###### Entreno el modelo con el mejor alpha ######
                mod = linear_model.Ridge(alpha = alpha_forzado, random_state=123)
                mod.fit(dstims_train_val, eeg_train_val) ## entreno el modelo
                ###### Guardo los pesos de esta ronda ######
                Pesos_ronda_canales[test_round] = mod.coef_
                
                ###### Predigo en val set  ######
                predicho = mod.predict(dstims_test)
                Predicciones[test_round] = predicho
    
            Pesos_promedio_canales = Pesos_ronda_canales.mean(0)  
            Pesos_promedio = Pesos_promedio_canales.mean(0)
            Pesos_promedio_saves[alpha_num] = Pesos_promedio
        
            # if individual_trace:
            #     ### BUSCO ALPHA ###
            #     df = pd.DataFrame(Pesos_promedio_saves)
            #     rolling = df.rolling(window = 5, min_periods = 1).mean()
                
            #     Pendientes = np.diff(rolling, axis = 0)
            #     Pendientes_2 = np.diff(rolling, axis = 0)
                
            #     Threshold = 0.1
            #     Cambio = abs(Threshold*rolling[:-1][:]) - abs(Pendientes)
                
            #     Derivada = 'Cambio'
                
            #     if Derivada == 1:
            #         for i in range(len(Pendientes)):
            #             if all(Pendientes[i] < Threshold): break
            #         Alpha = alphas[i+1]
            #         Alphas.append(Alpha)  
                    
            #     elif Derivada == 2:
            #         for i in range(len(Pendientes_2)):
            #             if all(Pendientes_2[i] < Threshold): break
            #         Alpha = alphas[i+1]
            #         Alphas.append(Alpha)  
                
            #     elif Derivada == 'Cambio':
            #         for i in range(len(Cambio)):
            #             if all(Cambio[i] > 0): break
            #         Alpha = alphas[i+1]
            #         Alphas.append(Alpha) 
                    
                
            #     if Display_figures_Trace: plt.ion() 
            #     else: plt.ioff()
                
            #     plt.figure()
            #     plt.suptitle('Ridge Trace')
            #     plt.title('Sesion {} - Sujeto {} - Band {} - Stim {}'.format(sesion, sujeto, Band, stim))
            #     plt.xlabel('Ridge Parameter')
            #     plt.ylabel('Standarized Coefficents')
            #     plt.xscale('log')
            #     plt.plot(alphas, rolling[:][:], 'o--', label = '')
            #     plt.vlines(Alpha, Pesos_promedio_saves.min(), Pesos_promedio_saves.max(), color = 'black', linestyle = 'dashed', linewidth = 2)
            #     plt.grid()
            
            # if Save_figures_Trace: 
            #     save_path_graficos = Run_graficos_path+ '{}/{}/'.format(stim,Band)
            #     try: os.makedirs(save_path_graficos)
            #     except: pass
            #     plt.savefig(save_path_graficos+ 'Sesion_{}_Sujeto{}.png'.format(sesion,sujeto))
        
            # Guardo las correlaciones y los pesos promediados entre test de cada canal del sujeto y lo adjunto a lista para promediar entre canales de sujetos
            if not sujeto_total:
                # if len(Canales_sobrevivientes_corr):
                Pesos_totales_sujetos_todos_canales = Pesos_promedio_canales     
            else:
                # if len(Canales_sobrevivientes_corr):
                Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, Pesos_promedio_canales))
            sujeto_total += 1  
     
    curva_pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_figure_instantes, Save_figure_instantes, Run_graficos_path, Cant_Estimulos, Stims_Order, stim)
    Standarized_Betas[alpha_num] = curva_pesos_totales[0]

plt.ion()
plt.figure()
plt.title('Ridge Trace - {} - {}'.format(Band, stim))
plt.xlabel('Ridge Parameter')
plt.ylabel('Standarized Coefficents')
plt.xscale('log')
plt.plot(alphas, Standarized_Betas[:,:10], 'o--', label = '')
plt.plot(alphas, Standarized_Betas[:,-10:], 'o--', label = '')
plt.vlines(1e3, Standarized_Betas.min(), Standarized_Betas.max(), color = 'black', linestyle = 'dashed', linewidth = 2)
plt.grid()
# plt.savefig('gráficos/Ridge_Trace/{}_{}.png'.format(Band, stim))


