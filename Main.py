# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:23:54 2021

@author: joaco
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import KFold
import os
import pickle
import Processing
import Plot
import Load, Simulation, Models

###### DEFINe PARAMETERS ######
# Stimuli and EEG
Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
stim = 'Envelope'
Band = 'Theta'
situacion = 'Escucha'
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
# Times
sr = 128
tmin, tmax = -0.53, 0.1
delays = - np.arange(np.floor(tmin*sr), np.ceil(tmax*sr), dtype=int)
times = np.linspace(delays[0]*np.sign(tmin)*1/sr, np.abs(delays[-1])*np.sign(tmax)*1/sr, len(delays))
# Model
# Model = 'Ridge'
alpha = 100
# Random permutations
Simulate_random_data = False
Statistical_test = False
# Display
Display_Ind_Figures = False
Display_Total_Figures = True
# Save
Save_Ind_Figures = False
Save_Total_Figures = False
# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin,tmax)
Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/Alpha_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(Stims_preprocess,EEG_preprocess,alpha,tmin,tmax, stim, Band)
Path_it = 'saves/Ridge/Fake_it/Stims_{}_EEG_{}/Alpha_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(Stims_preprocess, EEG_preprocess,alpha,tmin,tmax, stim, Band)
Path_Pesos_Predicciones_Corr_Rmse = 'saves/Ridge/Corr_Rmse_Pesos_Predicciones/Stims_{}_EEG_{}/Alpha_{}/tmin{}_tmax{}/Stim_{}__EEG_Band_{}/'.format(Stims_preprocess, EEG_preprocess,alpha,tmin,tmax, stim, Band)

# Save variables
Variables = {}
psd_pred_correlations = []
psd_rand_correlations = []

###### Start Run ######
sesiones = np.arange(21, 26)
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
        Pesos_ronda_canales = np.zeros((n_splits, info['nchan'], len(delays)*Cant_Estimulos))
        
        Prob_Corr_ronda_canales = np.ones((n_splits, info['nchan']))
        Prob_Rmse_ronda_canales = np.ones((n_splits, info['nchan']))
        
        Correlaciones_fake = np.zeros((n_splits, iteraciones, info['nchan']))
        Errores_fake = np.zeros((n_splits, iteraciones, info['nchan']))
        
        Corr_buenas_ronda_canal = np.zeros((n_splits, info['nchan']))
        Rmse_buenos_ronda_canal = np.zeros((n_splits, info['nchan']))
        
        Canales_repetidos_corr_sujeto = np.zeros(info['nchan'])
        Canales_repetidos_rmse_sujeto = np.zeros(info['nchan'])
      
        ###### Empiezo el KFold de test ######
        kf_test = KFold(n_splits, shuffle = False)
        for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
            eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]
            
            dstims_train_val = list()
            dstims_test = list()

            for stimulus in list(dstims):
                dstims_train_val.append(stimulus[train_val_index])
                dstims_test.append(stimulus[test_index])
                     
            axis = 0
            porcent = 5
            eeg, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg, dstims_train_val, dstims_test, Stims_preprocess, EEG_preprocess, axis = 0, porcent = 5)

            ###### Ajusto el modelo y guardo ######
            Model = Models.Ridge(alpha)
            Model.fit(dstims_train_val, eeg_train_val)
            Pesos_ronda_canales[fold] = Model.coefs
            
            ###### Predigo en test set y guardo ######
            predicted = Model.predict(dstims_test)
            Predicciones[fold] = predicted
            
            ###### Calculo Correlacion y guardo ######
            Rcorr = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicted[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])
            Corr_buenas_ronda_canal[fold] = Rcorr
            
            ###### Calculo Error y guardo ######
            Rmse = np.array(np.sqrt(np.power((predicted - eeg_test),2).mean(0)))
            Rmse_buenos_ronda_canal[fold] = Rmse
            
            # Calculo psd de pred y señal
            # fmin, fmax = 0, 40
            # psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'], fmin, fmax)
            # psds_pred, freqs_mean = mne.time_frequency.psd_array_welch(predicho.transpose(), info['sfreq'], fmin, fmax)
            
            # psds_channel_corr = np.array([np.corrcoef(psds_test[ii].ravel(), np.array(psds_pred[ii]).ravel())[0,1] for ii in range(len(psds_test))])
            # psd_pred_correlations.append(np.mean(psds_channel_corr))
            
            # Ploteo PSD
            # Plot.Plot_PSD(sesion, sujeto, fold, situacion, Display_PSD, Save_PSD, 'Prediccion', info, predicho.transpose())           
            # Plot.Plot_PSD(sesion, sujeto, fold, situacion, Display_PSD, Save_PSD, 'Test', info, eeg_test.transpose())                
            
            # Matriz de Covarianza
            # raw = mne.io.RawArray(predicho.transpose(), info)
            # cov_mat = mne.compute_raw_covariance(raw)
            # plt.ion()
            # ax1, ax2 = cov_mat.plot(info)
            # try: os.makedirs('gráficos/Covariance/Cov_prediccion')
            # except: pass
            # ax1.savefig('gráficos/Covariance/Cov_prediccion/Sesion{} - Sujeto{} - {}'.format(sesion,sujeto,situacion))
    
            # raw = mne.io.RawArray(eeg_test.transpose(), info)
            # cov_mat = mne.compute_raw_covariance(raw)
            # plt.ion()
            # ax1, ax2 = cov_mat.plot(info)
            # try: os.makedirs('gráficos/Covariance/Cov_test')
            # except: pass
            # ax1.savefig('gráficos/Covariance/Cov_test/Sesion{} - Sujeto{} - {}'.format(sesion,sujeto,situacion))
            
            
            ##### SINGAL AND PREDICTION PLOT #####
            # plt.ion()
            # fig = plt.figure()
            # fig.suptitle('Pearson Correlation = {}'.format(Rcorr[0]))
            # plt.plot(eeg_test[:,0], label = 'Signal')
            # plt.plot(predicho[:,0], label = 'Prediction')
            # plt.title('Original Signals - alpha: {}'.format(alpha))
            # plt.xlim([2000,3000])
            # plt.xlabel('Samples')
            # plt.ylabel('Amplitude')
            # plt.grid()
            # plt.legend()
            # plt.savefig('gráficos/Ridge/Alpha/Signals_alpha{}.png'.format(alpha))
            # plt.tight_layout()
            # break

            if Simulate_random_data: 
                ###### SIMULACIONES PERMUTADAS PARA COMPARAR ######
                toy_iterations = 10
                psd_rand_correlation = Simulation.simular_iteraciones_Ridge_plot(info,sr, situacion, alpha, toy_iterations, sesion, sujeto, fold, dstims_train_val, eeg_train_val, dstims_test, eeg_test)
                psd_rand_correlations.append(psd_rand_correlation)
            
            if Statistical_test:
                Simulation.simular_iteraciones_Ridge(alpha, iteraciones, sesion, sujeto, fold, dstims_train_val, eeg_train_val, dstims_test, eeg_test, Correlaciones_fake, Errores_fake, Path_it)
                
            else: # Load data from iterations     
                f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'rb')
                Correlaciones_fake, Errores_fake = pickle.load(f)
                f.close() 
            
            ##### TEST ESTADISTICO #####
            Rcorr_fake = Correlaciones_fake[fold]
            Rmse_fake = Errores_fake[fold] 
            
            p_corr = ((Rcorr_fake > Rcorr).sum(0)+1)/(iteraciones+1)
            p_rmse = ((Rmse_fake < Rmse).sum(0)+1)/(iteraciones+1)
            
            # Umbral de 5% y aplico Bonferroni (/128 Divido el umbral por el numero de intentos)
            umbral = 0.05/128 
            Prob_Corr_ronda_canales[fold][p_corr < umbral] = p_corr[p_corr < umbral]
            Prob_Rmse_ronda_canales[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]
        
        ###### Guardo Correlaciones y Rmse buenas de todos las fold ######
        try: os.makedirs(Path_Pesos_Predicciones_Corr_Rmse)
        except: pass

        f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Corr_Rmse_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
        pickle.dump([Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal], f)
        f.close()
        
        f = open(Path_Pesos_Predicciones_Corr_Rmse + 'Pesos_Pred_ronda_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
        pickle.dump([Pesos_ronda_canales, Predicciones], f)
        f.close()
            
        ###### Guardo los canales que pasaron las pruebas en todos los folds y los p valores ######
        
        Canales_Corr_prob = np.zeros(info['nchan'])
        Canales_Corr_std = np.zeros(info['nchan'])
        Canales_Rmse_prob = np.zeros(info['nchan'])
        Canales_Rmse_std = np.zeros(info['nchan'])
        
        # Armo lista con canales que pasan el test        
        Canales_sobrevivientes_corr, = np.where(np.all((Prob_Corr_ronda_canales < 1), axis = 0))
        Canales_sobrevivientes_rmse, = np.where(np.all((Prob_Rmse_ronda_canales < 1), axis = 0))

        # Guardo los canales sobrevivientes de cada sujeto     
        Canales_repetidos_corr_sujeto[Canales_sobrevivientes_corr] += 1
        Canales_repetidos_rmse_sujeto[Canales_sobrevivientes_rmse] += 1

        ###### Tomo promedio de pesos entre los folds para todos los canales ######
        Pesos_promedio = Pesos_ronda_canales.mean(0)
        
        ###### Tomo promedio de Corr y Rmse entre las rondas de test para todos los canales (para no desvirtuar las cabezas) ######
        Corr_promedio = Corr_buenas_ronda_canal.mean(0)
        Rmse_promedio = Rmse_buenos_ronda_canal.mean(0)  
               
        ###### Grafico cabezas y canales ######       
        Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Canales_sobrevivientes_corr, 
                                  Corr_promedio, Display_Ind_Figures, info['nchan'], 'Correlación', 
                                  Save_Ind_Figures, Run_graficos_path)
        Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Canales_sobrevivientes_rmse, 
                                  Rmse_promedio, Display_Ind_Figures, info['nchan'], 'Rmse', 
                                  Save_Ind_Figures, Run_graficos_path)
        
        ###### Grafico Pesos ######
        Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Pesos_promedio, 
                                info, times, sr, Corr_promedio, Rmse_promedio, Save_Ind_Figures,
                                Run_graficos_path, Cant_Estimulos, Stims_Order)
         
        ###### Grafico Shadows ######
        Plot.plot_grafico_shadows(Display_Ind_Figures, sesion, sujeto, alpha,
                                  Canales_sobrevivientes_corr, info, sr,
                                  Corr_promedio, Save_Ind_Figures, Run_graficos_path, 
                                  Corr_buenas_ronda_canal, Correlaciones_fake)
        
        # Guardo las correlaciones y los pesos promediados entre test de cada canal del sujeto y lo adjunto a lista para promediar entre canales de sujetos
        if not sujeto_total:
            # if len(Canales_sobrevivientes_corr):
            Pesos_totales_sujetos_todos_canales = Pesos_promedio
            
            Correlaciones_totales_sujetos = Corr_promedio
            Rmse_totales_sujetos = Rmse_promedio
            
            Canales_repetidos_corr_sujetos = Canales_repetidos_corr_sujeto
            Canales_repetidos_rmse_sujetos = Canales_repetidos_rmse_sujeto
        else:
            # if len(Canales_sobrevivientes_corr):
            Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, 
                                                              Pesos_promedio))
            
            Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
            Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))

            
            Canales_repetidos_corr_sujetos = np.vstack((Canales_repetidos_corr_sujetos,Canales_repetidos_corr_sujeto))
            Canales_repetidos_rmse_sujetos = np.vstack((Canales_repetidos_rmse_sujetos,Canales_repetidos_rmse_sujeto))
        sujeto_total += 1  
       
###### Armo cabecita con correlaciones promedio entre sujetos######
Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_Total_Figures, Save_Total_Figures, Run_graficos_path, title = 'Correlation')
Plot.Cabezas_corr_promedio(Rmse_totales_sujetos, info, Display_Total_Figures, Save_Total_Figures, Run_graficos_path, title = 'Rmse')

###### Armo cabecita con canales repetidos ######
Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures, Save_Total_Figures, Run_graficos_path, title = 'Correlation')    
Plot.Cabezas_canales_rep(Canales_repetidos_rmse_sujetos.sum(0), info, Display_Total_Figures, Save_Total_Figures, Run_graficos_path, title = 'Rmse')    

###### Instantes de interés ######
curva_pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_Total_Figures, Save_Total_Figures, Run_graficos_path, Cant_Estimulos, Stims_Order, stim)

# Matriz de Correlacion
Plot. Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, Display_Total_Figures, Save_Total_Figures, Run_graficos_path)

# Cabezas de correlacion de pesos por canal
Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info,Display_Total_Figures, Save_Total_Figures, Run_graficos_path)

# PSD Boxplot
Plot.PSD_boxplot(psd_pred_correlations, psd_rand_correlations, Display_Total_Figures, Save_Total_Figures, Run_graficos_path)

##### SAVE FINAL CORRELATION #####
save_path = 'saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/'.format(alpha,tmin,tmax)
try: os.makedirs(save_path)
except: pass
f = open(save_path + '{}_EEG_{}.pkl'.format(stim,Band), 'wb')
pickle.dump([Correlaciones_totales_sujetos, Canales_repetidos_corr_sujetos], f)
f.close()
