# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:28:05 2021

@author: joaco
"""
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io.wavfile as wavfile
import librosa as librosa
from sklearn.utils import resample
from scipy.linalg import toeplitz
import os
import pickle
import mne
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy import stats

import Funciones as func

###### Defino parametros ######
Load_procesed_data = False
Save_procesed_data = False
procesed_data_path = 'Preprocesed_data/'

Normalizar_todo = True

Plot_alphas = False
Prints_alphas = False

# FIGURAS IND
Display_figures_beta = True
Save_grafico_betas = False

sr = 128
delay_time = 0.53
###### Empiezo corrida ######
sesiones = np.arange(21, 26)
sujeto_total = 0
best_alphas = []

for sesion in sesiones:    
    print('Sesion {}'.format(sesion))
    if not Load_procesed_data:  
        eeg_sujeto_1, dstims_para_sujeto_1, eeg_sujeto_2, dstims_para_sujeto_2, info = func.load_files_process_data(sesion, sr, delay_time, Save_procesed_data, procesed_data_path, Normalizar_todo)
       
    elif Load_procesed_data:
        f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 1), 'rb')
        eeg_sujeto_1, dstims_para_sujeto_1 = pickle.load(f)
        f.close()
        
        f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 2), 'rb')
        eeg_sujeto_2, dstims_para_sujeto_2 = pickle.load(f)
        f.close()
        
        f = open(procesed_data_path + 'info.pkl', 'rb')
        info = pickle.load(f)
        f.close()
        
    montage = mne.channels.make_standard_montage('biosemi128')
    channel_names = montage.ch_names
    info = mne.create_info(ch_names = channel_names[:], sfreq = sr, ch_types = 'eeg').set_montage(montage)
    
    for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):      
        print('Sujeto {}'.format(sujeto))   
        ###### DIVIDO TRAIN RIDGE Y VAL ###### 
        tStim, tResp, rStim, rResp, vStim, vResp = func.separar_datos(dstims = dstims, eeg = eeg, porcentaje_train = 0.8, porcentaje_ridge = 0.0, porcentaje_val = 0.2)
             
        axis = 0
        porcent = 5
        norm = func.normalizar(axis, porcent)
        
        norm.fit_normalize_eeg(tResp)
        norm.normalize_eeg(vResp)

        norm.normalize_stims(tStim)
        norm.normalize_stims(vStim)
        
        n_best_channels = 16       
        min_busqueda = 0
        max_busqueda = 5
        pasos = 6
        
        ###### BUSCO ALPHA ######
        # best_alpha_overall, pesos, intercepts, lista_Rmse = func.buscar_alpha(tStim, tResp, vStim, vResp, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = False)     
        # ###### refino busqeuda de alpha cerca del encontrado #######
        # if best_alpha_overall == 1e5:
        #     min_busqueda,max_busqueda = 4,6
        # else:
        #     min_busqueda,max_busqueda = np.log10(best_alpha_overall)-1,np.log10(best_alpha_overall)+1
        
        # best_alpha_val, pesos, intercepts, lista_Rmse  = func.buscar_alpha(tStim, tResp, vStim, vResp, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = True)  
        # best_alphas.append(best_alpha_val)
        
        ###### BARRIDO EN ALPHAS ######
        alphas = np.hstack([0,np.logspace(min_busqueda, max_busqueda, pasos)])       
        for alpha in alphas:
            mod = linear_model.Ridge(alpha=alpha, random_state=123)
            mod.fit(tStim, tResp) ## entreno el modelo
            
            ###### TESTEO EN VAL SET  ######
            # Predigo
            predicho = mod.predict(vStim)
            
            # Correlacion
            Rcorr = np.array([np.corrcoef(vResp[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(vResp.shape[1])])
            mejor_canal_corr = Rcorr.argmax()
            Corr_mejor_canal = Rcorr[mejor_canal_corr]
            
            # Error
            Rmse = np.array(np.sqrt(np.power((predicho - vResp),2).mean(0)))
            mejor_canal_rmse = Rmse.argmax()
            Rmse_mejor_canal = Rmse[mejor_canal_rmse]
            
            Correl_prom = np.mean(Rcorr)
            Rmse_prom = np.mean(Rmse)
                      
            #GRAFICO BETAS, CORRELACION Y ERROR
            if Save_grafico_betas: 
                save_path_graficos = 'gráficos/Prueba_envelope/Sesion{}/Sujeto{}'.format(sesion,sujeto)
                try: os.makedirs(save_path_graficos)
                except: pass
        
            if Display_figures_beta: plt.ion() 
            else: plt.ioff()
            
            fig, axs = plt.subplots(3,1,figsize=(10,9.5))
            fig.suptitle('Sesion:{} - Sujeto:{} - alpha:{:.2f} - Corr max:{:.2f} - Rmse max:{:.2f}'.format(sesion, sujeto, alpha, Corr_mejor_canal, Rmse_mejor_canal))
            
            evoked = mne.EvokedArray(mod.coef_, info)
            evoked.plot(show = False, spatial_colors=True, 
                        scalings = dict(eeg=1, grad=1, mag=1), unit = True, units = dict(eeg = 'Pesos'), 
                        axes = axs[0], zorder = 'unsorted')
            axs[0].plot(np.arange(0, mod.coef_.shape[1]/sr, 1.0/sr), 
                    mod.coef_.mean(0),'k--', 
                    label = 'Promedio', zorder = 130, linewidth = 1.5)
            axs[0].set_xlabel('Delay [s]')
            axs[0].grid()
            axs[0].set_title('')
            axs[0].legend()

            axs[1].plot(Rcorr, '.', color = 'C0')
            axs[1].hlines(Correl_prom, plt.xlim()[0],plt.xlim()[1], label = 'Promedio = {:0.2f}'.format(Correl_prom), color = 'C1')
            axs[1].set_xlabel('Canales')
            axs[1].set_ylabel('Correlación')
            axs[1].legend()
            axs[1].grid()
        
            axs[2].plot(Rmse, '.', color = 'C0')
            axs[2].hlines(Rmse_prom, plt.xlim()[0],plt.xlim()[1], label = 'Promedio = {:0.2f}'.format(Rmse_prom), color = 'C1')
            axs[2].set_xlabel('Canales')
            axs[2].set_ylabel('Rmse')
            axs[2].legend()
            axs[2].grid()
            
            fig.tight_layout() 
        
            if Save_grafico_betas: plt.savefig(save_path_graficos + '/alpha{}.png'.format(alpha))
#%% FINO
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.io.wavfile as wavfile
import librosa as librosa
from sklearn.utils import resample
from scipy.linalg import toeplitz
import os
import pickle
import mne

import Funciones as func

Plot_alphas = False
Save_grafico_betas = True
Prints = False
Save_text = False
Display_figures_beta = False
CSD = True
Use_PCA = True

sr = 128
delay_time = 0.53

sesiones = np.arange(21,26)
for sesion in sesiones:    
    run = True
    i=0
    
    eeg_sujeto_1 = pd.DataFrame()
    eeg_sujeto_2 = pd.DataFrame()
    
    dstims_para_sujeto_1 = pd.DataFrame()
    dstims_para_sujeto_2 = pd.DataFrame()

    while run:
    # for i in range(10):
        trial = i+1
        
        if trial % 2: 
            canal_hablante = 1
        else:
            canal_hablante = 2      
        canal_oyente = (canal_hablante - 3)*(-1)
                
        try: 
            eeg_trial = func.f_eeg(sesion,trial, canal_oyente, CSD, sr, delay_time)
            dstims_trial = func.f_envelope(sesion,trial, canal_hablante, sr, delay_time)
            momentos_escucha = func.solo_escucha(sesion,trial, canal_hablante, sr, delay_time)
        except: 
            run = False
            
        if run: 
            ###### Igualar largos ######
            eeg_trial, dstims_trial, momentos_escucha = func.igualar_largos(eeg = eeg_trial, enve = dstims_trial, momentos_escucha = momentos_escucha, sr = sr, delay_time = delay_time)
            
            ###### PREPROCESAMIENTO ######
            eeg_trial, dstims_trial = func.preproc(preproc_mauro_mal = False, preproc_fila = True, eeg = eeg_trial, enve = dstims_trial, momentos_escucha = momentos_escucha, sr = sr, delay_time = delay_time)
            
            eeg_trial = pd.DataFrame(eeg_trial)
            dstims_trial = pd.DataFrame((dstims_trial))
            
            if trial % 2: 
                eeg_sujeto_2 = eeg_sujeto_2.append(eeg_trial)
                dstims_para_sujeto_2 = dstims_para_sujeto_2.append(dstims_trial)
            else:
                eeg_sujeto_1 = eeg_sujeto_1.append(eeg_trial)
                dstims_para_sujeto_1 = dstims_para_sujeto_1.append(dstims_trial)
            
            i += 1
    
    for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
        eeg = np.array(eeg)
        dstims = np.array(dstims)
        
        if Use_PCA:
            pca = PCA()
            eeg_pca = pca.fit_transform(eeg)
            
            porcentaje_var = 0.03
            indice_comp = sum((pca.explained_variance_ratio_) > porcentaje_var)
            eeg_pca = eeg_pca[:,:indice_comp]
            
            eeg = eeg_pca
            
        ###### DIVIDO TRAIN RIDGE Y VAL ###### 
        tStim, tResp, rStim, rResp, vStim, vResp = func.separar_datos(dstims = dstims, eeg = eeg, porcentaje_train = 0.6, porcentaje_ridge = 0.2, porcentaje_val = 0.2)
        
        #BUSCO EL HIPERPARÁMETRO ALPHA
        #entreno variando el alpha entre exp(-/+6) y me fijo cual tiene mejor correlación
        #con el Ridge set para elegir el mejor alpha (Creo qeu eso es el criterio de Liberty)
        best_alpha_overall, correlaciones_grueso, best_channels, pesos, intercepts, lista_Rmse = func.buscar_alpha(tStim, tResp, rStim, rResp, Plot_alphas, Prints, min_busqueda = -6, max_busqueda = 6, pasos = 13, fino = False)
        
        min_busqueda,max_busqueda = np.log10(best_alpha_overall)-1,np.log10(best_alpha_overall)+1

        # GRAF BETAS VS ALPHAS
        pasos = 50
        
        alphas = np.hstack([0,np.logspace(min_busqueda, max_busqueda, pasos)])
        for alpha in alphas:
            mod = linear_model.Ridge(alpha=alpha, random_state=123)
            mod.fit(tStim,tResp) ## entreno el modelo
           
            ###### TESTEO EN RIDGEN SET (DEBERIA SER VAL SET) ######
            
            predicho = mod.predict(rStim)
            Rcorr = np.array([np.corrcoef(rResp[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(rResp.shape[1])])
            mejor_canal = Rcorr.argmax()
            
            #GRAFICO BETAS (PESOS)
            if Save_grafico_betas: 
                save_path_graficos = 'gráficos/Betas_vs_alphas_CSD_PCA/Fino'
                try: os.makedirs(save_path_graficos)
                except: pass
            
            if Display_figures_beta: plt.ion() 
            else: plt.ioff()
            
            plt.figure()
            plt.plot(np.arange(0, mod.coef_.shape[1]/sr, 1.0/sr),mod.coef_.transpose(),"-" )
            plt.title('alpha {}'.format(alpha))
            plt.grid()
            plt.ylabel('Betas')
            plt.xlabel('Delay [ms]')
            if Save_grafico_betas: plt.savefig(save_path_graficos + '/Sesion{}_Sujeto{}_alpha{}.png'.format(sesion, sujeto, alpha))
            plt.legend()    