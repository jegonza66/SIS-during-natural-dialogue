# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 14:38:57 2021

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

Causal_filter = True

Calculate_pitch = False # Solo si cambias threshold, sampleStep, min o max
valores_faltantes_pitch = 0

Buscar_alpha = False
if not Buscar_alpha: alpha_forzado = 1000
Plot_alphas = False

Fit_Model = True # If false it loads the correlations and weights of every channel for every test round
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
Display_Total = True
if Display_Total: Display_correlacion_promedio, Display_canales_repetidos, Display_figure_instantes, Display_correlation_matrix, Display_channel_correlation_topo, Display_PSD_boxplot = True, True, True, True, True, True
else: Display_correlacion_promedio, Display_canales_repetidos, Display_figure_instantes, Display_correlation_matrix, Display_channel_correlation_topo, Display_PSD_boxplot = False, False, False,  False,  False,  False
Save_Total = False
if Save_Total: Save_correlacion_promedio, Save_canales_repetidos, Save_figure_instantes, Save_correlation_matrix, Save_channel_correlation_topo, Save_PSD_boxplot = True, True, True, True, True, True
else: Save_correlacion_promedio, Save_canales_repetidos, Save_figure_instantes, Save_correlation_matrix, Save_channel_correlation_topo, Save_PSD_boxplot = False, False, False, False,  False,  False

###### DEFINO PARAMETROS ######
stim = 'Envelope'
###### Defino banda de eeg ######
Band = 'All'
###### Defino situacion de interes ######
situacion = 'Escucha'
###### Defino estandarizacion
Normalizar = 'All'
Estandarizar = None
# Defino tiempos
sr = 128
n_canales = 128
tmin, tmax = -0.53, -0.003
delays = - np.arange(np.floor(tmin*sr), np.ceil(tmax*sr), dtype=int)
times = np.linspace(delays[0]*np.sign(tmin)*1/sr, np.abs(delays[-1])*np.sign(tmax)*1/sr, len(delays))

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
        ###### Separo los datos en 5 y tomo test set de ~20% de datos con kfold (5 iteraciones)######
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

            else: best_alpha = alpha_forzado
             
            if Normalizar: 
                axis = 0
                porcent = 5
                Processing.normalizacion(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Normalizar, axis, porcent)
                          
            elif Estandarizar:
                axis = 0
                Processing.estandarizacion(eeg_train_val, eeg_test, dstims_train_val, dstims_test, Estandarizar, axis)
                
            if Fit_Model:
                ###### Entreno el modelo con el mejor alpha ######
                mod = linear_model.Ridge(alpha = best_alpha, random_state=123)
                mod.fit(dstims_train_val, eeg_train_val) ## entreno el modelo
                ###### Guardo los pesos de esta ronda ######
                Pesos_ronda_canales[test_round] = mod.coef_
                
                ###### Predigo en val set  ######
                predicho = mod.predict(dstims_test)
                Predicciones[test_round] = predicho
                
               
                ###### Calculo Correlacion ######
                Rcorr = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])
                mejor_canal_corr = Rcorr.argmax()
                Corr_mejor_canal = Rcorr[mejor_canal_corr]
                Correl_prom = np.mean(Rcorr)
                ###### Guardo las correlaciones de esta ronda ######
                Corr_buenas_ronda_canal[test_round] = Rcorr

                
                ###### Calculo Error ######
                Rmse = np.array(np.sqrt(np.power((predicho - eeg_test),2).mean(0)))
                mejor_canal_rmse = Rmse.argmax()
                Rmse_mejor_canal = Rmse[mejor_canal_rmse]
                Rmse_prom = np.mean(Rmse)
                ###### Guardo los errores de esta ronda ######
                Rmse_buenos_ronda_canal[test_round] = Rmse
            
            else: # Load Correlations and Rmse buenas         
                f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Corr_Rmse_buenas_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'rb')
                Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal = pickle.load(f)
                f.close()
            
                Rcorr = Corr_buenas_ronda_canal[test_round]
                Rmse = Rmse_buenos_ronda_canal[test_round]
            
            if Simulate_random_data: 
                ###### SIMULACIONES PERMUTADAS PARA COMPARAR ######
                toy_iterations = 10
                psd_rand_correlation = Simulation.simular_iteraciones_Ridge_plot(info,sr, situacion,best_alpha, toy_iterations, sesion, sujeto, test_round, dstims_train_val, eeg_train_val, dstims_test, eeg_test)
                psd_rand_correlations.append(psd_rand_correlation)
            
            if Statistical_test:
                Simulation.simular_iteraciones_Ridge(best_alpha, iteraciones, sesion, sujeto, test_round, dstims_train_val, eeg_train_val, dstims_test, eeg_test, Correlaciones_fake, Errores_fake, Save_iterations, Path_it)
                
            else: # Load data from iterations     
                f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'rb')
                Correlaciones_fake, Errores_fake = pickle.load(f)
                f.close() 
                
            for canal in range(n_canales): # Numero de canales
                Rcorr_fake_canal = Correlaciones_fake[test_round, :, canal]
                Rmse_fake_canal = Errores_fake[test_round, :,canal]  
            
                p_corr = (sum(abs(j) > abs(Rcorr[canal]) for j in Rcorr_fake_canal)+1)/(iteraciones+1)
                p_rmse = (sum(j < Rmse[canal] for j in Rmse_fake_canal)+1)/(iteraciones+1)
                
                # Umbral de 1% y aplico Bonferroni (/128 Divido el umbral por el numero de intentos)
                umbral = 0.05/128             
                if p_corr < umbral:
                    Prob_Corr_ronda_canales[test_round, canal] = p_corr
                if p_rmse < umbral:
                    Prob_Rmse_ronda_canales[test_round, canal] = p_rmse
        
        ###### Guardo Correlaciones y Rmse buenas de todos las test_round ######
        if Save_Pesos_Predicciones_Corr_Rmse_buenas:
            try: os.makedirs(Path_Pesos_Predicciones_Corr_Rmse)
            except: pass
        
            f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Corr_Rmse_buenas_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
            pickle.dump([Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal], f)
            f.close()
            
            f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Pesos_Predicciones_buenas_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
            pickle.dump([Pesos_ronda_canales, Predicciones], f)
            f.close()
        
        ###### Si no ajusto modelo cargo los pesos del modelo ######
        if not Fit_Model:
            f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Pesos_Predicciones_buenas_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'rb')
            Pesos_ronda_canales, Predicciones = pickle.load(f)
            f.close()
            
        ###### Guardo los canales que pasaron las pruebas en todos los test_rounds y los p valores ######
        Canales_sobrevivientes_corr = []
        Canales_sobrevivientes_rmse = []
        
        Canales_Corr_prob = np.zeros(n_canales)
        Canales_Corr_std = np.zeros(n_canales)
        Canales_Rmse_prob = np.zeros(n_canales)
        Canales_Rmse_std = np.zeros(n_canales)
        
        # Armo vector con cantidad de test que pasa cada canal        
        for canal in range(n_canales): 
            Canales_repetidos_ronda[canal] += sum(Prob_Corr_ronda_canales[:,canal] < 1)
            if all(Prob_Corr_ronda_canales[:,canal] < 1): 
                Canales_sobrevivientes_corr.append(canal)
            if all (Prob_Rmse_ronda_canales[:,canal] < 1): 
                Canales_sobrevivientes_rmse.append(canal)
                           
        ###### Tomo promedio de pesos entre los folds para todos los canales ######
        Pesos_promedio = Pesos_ronda_canales.mean(0)
        
        ###### Tomo promedio del modulo de Corr y Rmse entre las rondas de test para todos los canales (para no desvirtuar las cabezas) ######
        Corr_promedio_abs = abs(Corr_buenas_ronda_canal).mean(0)
        Rmse_promedio = abs(Rmse_buenos_ronda_canal).mean(0)  
               
        ###### Grafico cabezas y canales ######       
        Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Canales_sobrevivientes_corr, 
                                  Corr_promedio_abs, Display_cabezas_canales, n_canales, 'Correlación', 
                                  Save_cabezas_canales, Run_graficos_path)
        Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Canales_sobrevivientes_rmse, Rmse_promedio, Display_cabezas_canales, n_canales, 'Rmse', Save_cabezas_canales, Run_graficos_path)
        
        ###### Grafico Pesos, Corr, Rmse ######
        Plot.plot_grafico_pesos_todos(Display_figures_beta, sesion, sujeto, best_alpha, Pesos_promedio, 
                                Canales_sobrevivientes_corr, info, times, sr, 
                                Corr_promedio_abs, Rmse_promedio, Canales_sobrevivientes_rmse, 
                                Save_grafico_betas, Run_graficos_path, 
                                Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal,
                                Errores_fake, Correlaciones_fake)
         
        ###### Grafico Shadows ######
        # Plot.plot_grafico_shadows(channel_names, Display_figures_shadows, sesion, sujeto, best_alpha,
        #                           Canales_sobrevivientes_corr, info, sr,
        #                           Corr_promedio_abs, Save_grafico_shadows, Run_graficos_path, 
        #                           Corr_buenas_ronda_canal, Correlaciones_fake)
        
        # Guardo pesos promediados entre todos los canales (buenos) del sujeto y lo adjunto a lista para correlacionar entre sujetos
        Pesos_totales_sujetos_significativos.append(Pesos_promedio[Canales_sobrevivientes_corr].mean(0))
        Pesos_totales_sujetos_promedio.append(Pesos_promedio.mean(0))
        
        # Guardo los canales sobrevivientes de cada sujeto
        for i in Canales_sobrevivientes_corr:
            Canales_repetidos_sujetos[i] += 1 
        
        # Guardo las correlaciones y los pesos promediados entre test de cada canal del sujeto y lo adjunto a lista para promediar entre canales de sujetos
        if not sujeto_total:
            # if len(Canales_sobrevivientes_corr):
            Pesos_totales_sujetos_todos_canales = Pesos_promedio
            Correlaciones_totales_sujetos = Corr_promedio_abs
            sujeto_total += 1              
        else:
            # if len(Canales_sobrevivientes_corr):
            Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, 
                                                             Pesos_promedio))
            Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio_abs))
            sujeto_total += 1  
    
# f = open('saves/Alphas/Best_Alphas_Total.pkl', 'wb')
# pickle.dump(best_alphas_total, f)
# f.close()

###### Armo cabecita con correlaciones promedio entre sujetos######
Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_correlacion_promedio, Save_correlacion_promedio, Run_graficos_path)

###### Armo cabecita con canales repetidos ######
Plot.Cabezas_canales_rep(Canales_repetidos_sujetos, info, Display_canales_repetidos, Save_canales_repetidos, Run_graficos_path)    

###### Instantes de interés ######
curva_pesos_totales = Plot.Plot_instantes_interes(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, delays, Display_figure_instantes, Save_figure_instantes, Run_graficos_path)

# Matriz de Correlacion
Plot. Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, Display_correlation_matrix, Save_correlation_matrix, Run_graficos_path)

# Cabezas de correlacion de pesos por canal
Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_channel_correlation_topo, Save_channel_correlation_topo, Run_graficos_path)

# PSD Boxplot
Plot.PSD_boxplot(psd_pred_correlations, psd_rand_correlations, Display_PSD_boxplot, Save_PSD_boxplot, Run_graficos_path)
