# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:26:37 2021

@author: joaco
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
import pickle
from scipy import stats
import Funciones as func

###### Defino parametros ######
Load_procesed_data = True
Save_procesed_data = False
procesed_data_path = 'Preprocesed_data/'

Normalizar_todo = True

Buscar_alpha = True
if not Buscar_alpha: alpha_forzado = 100
Plot_alphas = False

Prints_alphas = False

Save_best_alphas_val = True 
Save_figure_alphas_val = True

Run_graficos_path = 'gráficos/Kfold/'
Run_saves_path = 'saves/Kfold/'

sr = 128
delay_time = 0.53
n_canales = 128

###### Empiezo corrida ######
sesiones = np.arange(21,26)
sujeto_total = 0
for sesion in sesiones:
    print('Sesion {}'.format(sesion))    
    if not Load_procesed_data:  
        eeg_sujeto_1, dstims_para_sujeto_1, eeg_sujeto_2, dstims_para_sujeto_2 = func.load_files_process_data(sesion, sr, delay_time, procesed_data_path)
       
    elif Load_procesed_data:
        f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 1), 'rb')
        eeg_sujeto_1, dstims_para_sujeto_1 = pickle.load(f)
        f.close()
        
        f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 2), 'rb')
        eeg_sujeto_2, dstims_para_sujeto_2 = pickle.load(f)
        f.close()

    for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
    # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):    
        print('Sujeto {}'.format(sujeto))
        ###### Separo los datos en 5 y tomo test set de ~20% de datos con kfold (5iteraciones)######
        n_splits = 5
        iteraciones = 3000
        best_alphas_val_total = pd.DataFrame()
        
        ###### Defino variables donde voy a guardar mil cosas ######
        Pesos_ronda_canales = np.zeros((n_splits, n_canales, int(sr*delay_time)))
        
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
                n_best_channels = 16 # mejores canales por alpha
                
                # Obtengo lista de mejores alphas de cada corrida de val
                best_alphas_val = func.alphas_val(eeg_train_val, dstims_train_val, n_val_splits, train_percent, Normalizar_todo, n_best_channels, Plot_alphas, Prints_alphas)
                best_alpha_test = np.mean(best_alphas_val)
                best_alphas_val_total['test_round_{}'.format(test_round)] = best_alphas_val
                                
                # # Testeo normalidad de alphas de val
                # k2, p = stats.normaltest(best_alphas_val)     
                # if p > 0.05/n_val_splits:
                #     print('\nSesion {} - Sujeto {}\nAlphas de val siguen dist normal\n'.format(sesion, sujeto))
                #     print(best_alphas_val)
                # else: 
                #     print('\nSesion {} - Sujeto {}\nAlphas de val NO siguen dist normal\n'.format(sesion, sujeto))
                #     print(best_alphas_val)
                
        # Guardo alphas            
        if Save_best_alphas_val:      
            best_alphas_val_total.to_csv(Run_saves_path + 'Alphas/best_alphas_val_Sesion{}_Sujeto{}.csv'.format(sesion,sujeto))        
        
        
        # Grafico alphas val ESCALA LOG
        fig, axs = plt.subplots(1,n_splits, sharey=True, figsize = (19,5))
        plt.suptitle('Sesion {}, Sujeto {}'.format(sesion,sujeto))
        for i in range(best_alphas_val_total.shape[1]):
            axs[i].plot(best_alphas_val_total.to_numpy()[:,i], '.')
            axs[i].hlines(best_alphas_val_total.to_numpy()[:,i].mean(), axs[i].get_xlim()[0],
                          axs[i].get_xlim()[1], 
                          label = 'Promedio = {:0.2f}'.format(best_alphas_val_total.to_numpy()[:,i].mean()),
                          color = 'C1',linestyles = 'dashed', linewidth = 1.3)
            axs[i].hlines(stats.gmean(best_alphas_val_total.to_numpy()[:,i]), -10, 40,
                          label = 'Media Geo = {:0.2f}'.format(stats.gmean(best_alphas_val_total.to_numpy()[:,i])), 
                          color = 'C2', linestyles = 'dashed', linewidth = 1.3)
            axs[i].hlines(stats.hmean(best_alphas_val_total.to_numpy()[:,i]), -10, 40,
                          label = 'Media armonica = {:0.2f}'.format(stats.hmean(best_alphas_val_total.to_numpy()[:,i])), 
                          color = 'C3', linestyles = 'dashed', linewidth = 1.3)
            axs[i].hlines(np.median(best_alphas_val_total.to_numpy()[:,i]), -10, 40,
                          label = 'Mediana = {:0.2f}'.format(np.median(best_alphas_val_total.to_numpy()[:,i])), 
                          color = 'C4', linestyles = 'dashed', linewidth = 1.3)
            axs[i].hlines(stats.trim_mean(best_alphas_val_total.to_numpy()[:,i], 0.05),-10,40,
                          label = 'Media Truncada (5%)= {:0.2f}'.format(stats.trim_mean(best_alphas_val_total.to_numpy()[:,i], 0.05)), 
                          color = 'C5', linestyles = 'dashed', linewidth = 1.3)
            axs[i].hlines(10**(np.log10(best_alphas_val_total.to_numpy()[:,i]).mean()),-10,40,
                          label = 'Media de log10  = {:0.2f}'.format(stats.trim_mean(best_alphas_val_total.to_numpy()[:,i], 0.05)), 
                          color = 'C6', linestyles = 'dashed', linewidth = 1.3)
            axs[i].set_title('test round {}'.format(i+1))               
            axs[i].grid()
            axs[i].legend(loc = 'best', prop = {'size':7})
            
        if Save_figure_alphas_val: 
            save_path_graficos = Run_graficos_path + 'Alphas_Val/'
            try: os.makedirs(save_path_graficos)
            except: pass
            fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion,sujeto))       
    
    
    
    
    
    
