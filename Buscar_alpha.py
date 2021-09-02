# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 16:16:18 2021

@author: joaco
"""
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
from scipy import stats
import os
import pickle
from numpy.fft import fft, fftfreq    
import Processing
import Plot
import Load, Validation, Simulation

###### Defino parametros ######
Load_procesed_data = True
Save_procesed_data = False

Causal_filter = False

Calculate_pitch = False # Solo si cambias threshold, sampleStep, min o max
valores_faltantes_pitch = 0

Buscar_alpha = True
if not Buscar_alpha: alpha_forzado = 1000
Plot_alphas = False

Fit_Model = False # If false it loads the correlations and weights of every channel for every test round
Save_Pesos_Predicciones_Corr_Rmse_buenas = False

Simulate_random_data = False

Statistical_test = False
Save_iterations = False

Prints_alphas = False

# FIGURAS IND
Display_Ind = False
if Display_Ind: Display_figures_beta, Display_figures_shadows, Display_cabezas_canales, Display_PSD = True, True, True, True
else: Display_figures_beta, Display_figures_shadows, Display_cabezas_canales, Display_PSD = False, False, False, False
Save_Ind = False
if Save_Ind: Save_grafico_betas, Save_grafico_shadows, Save_cabezas_canales, Save_PSD = True, True, True, True
else: Save_grafico_betas, Save_grafico_shadows, Save_cabezas_canales, Save_PSD = False, False, False, False
  
# FIGURAS TODOS
Display_Total = False
if Display_Total: Display_correlacion_promedio, Display_canales_repetidos, Display_figure_instantes, Display_correlation_matrix, Display_channel_correlation_topo, Display_PSD_boxplot = True, True, True, True, True, True
else: Display_correlacion_promedio, Display_canales_repetidos, Display_figure_instantes, Display_correlation_matrix, Display_channel_correlation_topo, Display_PSD_boxplot = False, False, False,  False,  False,  False
Save_Total = False
if Save_Total: Save_correlacion_promedio, Save_canales_repetidos, Save_figure_instantes, Save_correlation_matrix, Save_channel_correlation_topo, Save_PSD_boxplot = True, True, True, True, True, True
else: Save_correlacion_promedio, Save_canales_repetidos, Save_figure_instantes, Save_correlation_matrix, Save_channel_correlation_topo, Save_PSD_boxplot = False, False, False, False,  False,  False

###### DEFINO PARAMETROS ######
stim = 'Envelope'
###### Defino banda de eeg ######
Band = 'Theta'
###### Defino situacion de interes ######
situacion = 'Escucha'
###### Defino estandarizacion
Normalizar = 'Stims'
Estandarizar = 'EEG'
# Defino tiempos
sr = 128
n_canales = 128
tmin, tmax = -0.53, -0.003
delays = - np.arange(np.floor(tmin*sr), np.floor(tmax*sr)+1, dtype=int)

###### Paths ######
procesed_data_path = 'saves/Preprocesed_Data/'

Run_graficos_path = 'gráficos/Ridge/'
Path_it = 'saves/Ridge/Fake_it/'
Path_Pesos_Predicciones_Corr_Rmse = 'saves/Ridge/Corr_Rmse_Pesos_Predicciones/'

Path_it, Run_graficos_path, Path_Pesos_Predicciones_Corr_Rmse = Load.rename_paths(Estandarizar, Normalizar, stim, valores_faltantes_pitch, Band, tmin, tmax, Causal_filter, Path_it, Run_graficos_path, Path_Pesos_Predicciones_Corr_Rmse)

# lista para guardar los mejores alphas elegidos y los pesos totales de los sujetos
best_alphas_total = []
Pesos_totales_sujetos_significativos = []
Pesos_totales_sujetos_promedio = []
Canales_repetidos_sujetos = np.zeros(128)
Canales_repetidos_ronda = np.zeros(128)
psd_pred_correlations = []
psd_rand_correlations = []

###### Empiezo corrida ######
sesiones = np.arange(21, 26)
sujeto_total = 0
for sesion in sesiones:
    print('Sesion {}'.format(sesion))
    Sesion_obj = Load.Sesion(sesion, Band, sr, tmin, tmax, valores_faltantes_pitch, Causal_filter, situacion, Calculate_pitch, procesed_data_path, Save_procesed_data)
    if Load_procesed_data:
        # Intento cargar preprocesados
        try: 
            Sesion = Sesion_obj.load_procesed()
            Sujeto_1, Sujeto_2 = Sesion['Sujeto_1'], Sesion['Sujeto_2']

            eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1 = Sujeto_1['EEG'], Sujeto_2['Envelope'], Sujeto_2['Pitch']
            eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2 = Sujeto_2['EEG'], Sujeto_1['Envelope'], Sujeto_1['Pitch']
            info = Sujeto_1['info']       
        # Si falla cargo de raw y guardo en Auto_save    
        except:  
            if not sujeto_total: procesed_data_path += 'Auto_save/'
            Save_procesed_data = True
            Sesion_obj = Load.Sesion(sesion, Band, sr, tmin, tmax, valores_faltantes_pitch, Causal_filter, situacion, Calculate_pitch, procesed_data_path, Save_procesed_data)

            Sesion = Sesion_obj.load_from_raw() 
            Sujeto_1, Sujeto_2  = Sesion['Sujeto_1'], Sesion['Sujeto_2']
            
            eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1 = Sujeto_1['EEG'], Sujeto_2['Envelope'], Sujeto_2['Pitch']
            eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2 = Sujeto_2['EEG'], Sujeto_1['Envelope'], Sujeto_1['Pitch']
            info = Sujeto_1['info']
    # Cargo directo de raw
    else: 
        Sesion = Sesion_obj.load_from_raw() 
        Sujeto_1, Sujeto_2 = Sesion['Sujeto_1'], Sesion['Sujeto_2']
        
        eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1 = Sujeto_1['EEG'], Sujeto_2['Envelope'], Sujeto_2['Pitch']
        eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2 = Sujeto_2['EEG'], Sujeto_1['Envelope'], Sujeto_1['Pitch']
        info = Sujeto_1['info']
    
    if stim == 'Envelope': dstims_para_sujeto_1, dstims_para_sujeto_2 = envelope_para_sujeto_1, envelope_para_sujeto_2
    elif stim == 'Pitch': 
        dstims_para_sujeto_1, dstims_para_sujeto_2 = pitch_para_sujeto_1[pd.DataFrame(pitch_para_sujeto_1).notna().all(1)], pitch_para_sujeto_2[pd.DataFrame(pitch_para_sujeto_2).notna().all(1)]
        eeg_sujeto_1, eeg_sujeto_2 = eeg_sujeto_1[pd.DataFrame(pitch_para_sujeto_1).notna().all(1)], eeg_sujeto_2[pd.DataFrame(pitch_para_sujeto_2).notna().all(1)]
    elif stim == 'Full': dstims_para_sujeto_1, dstims_para_sujeto_2 = np.hstack(envelope_para_sujeto_1,pitch_para_sujeto_1), np.hstack(envelope_para_sujeto_2,pitch_para_sujeto_2)
    else:
        print('Invalid sitmulus')
        break
    
    for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
    # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):    
        print('Sujeto {}'.format(sujeto))
        ###### Separo los datos en 5 y tomo test set de ~15% de datos con kfold (5 iteraciones)######
        Predicciones = {}
        n_splits = 5
        iteraciones = 3000
    
        ###### Defino variables donde voy a guardar mil cosas ######
        Pesos_ronda_canales = np.zeros((n_splits, n_canales, len(delays)))
        
        Prob_Corr_ronda_canales = np.ones((n_splits, n_canales))
        Prob_Rmse_ronda_canales = np.ones((n_splits, n_canales))
        
        Correlaciones_fake = np.zeros((n_splits, iteraciones, n_canales))
        Errores_fake = np.zeros((n_splits, iteraciones, n_canales))
        
        Corr_buenas_ronda_canal = np.zeros((n_splits, n_canales))
        Rmse_buenos_ronda_canal = np.zeros((n_splits, n_canales))
        
        ###### Empiezo el KFold de test ######
        kf_test = KFold(n_splits, shuffle = False)
        for test_round, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
            eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]
            dstims_train_val, dstims_test = dstims[train_val_index], dstims[test_index]
            
            if Buscar_alpha:

                train_percent = 0.8
                n_val_splits = 32
                n_best_channels = 32 # mejores canales por alpha
                min_busqueda = -1
                max_busqueda = 6
                
                # Obtengo lista de mejores alphas de cada corrida de val
                best_alphas_val = Validation.alphas_val(eeg_train_val, dstims_train_val, n_val_splits, train_percent, Normalizar, Estandarizar, n_best_channels, min_busqueda, max_busqueda, Plot_alphas, Prints_alphas)
                best_alpha = np.mean(best_alphas_val)
                best_alphas_total.append(best_alphas_val)

plt.ion()
plt.figure()
plt.plot(best_alphas_total, '.')
plt.hlines(np.mean(best_alphas_total), plt.xlim()[0], plt.xlim()[1], 'k', linestyles = 'dashed', label = 'Mean = {}'.format(np.mean(best_alphas_total)))
plt.legend()
f = open('saves/Alphas/Best_Alphas_Total_Theta.pkl', 'wb')
pickle.dump(best_alphas_total, f)
f.close()  
