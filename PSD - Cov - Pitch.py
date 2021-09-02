# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 17:06:40 2021

@author: joaco
"""
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sn
import os
import pickle

import Load
import Funciones as func

###### Defino parametros ######
Load_procesed_data = True
Save_procesed_data = False

Causal_filter = False

PSD = True
Cov = False

Pitch = False
Calculate_pitch = False
valores_faltantes_pitch = None

Display_fig = True
Save_fig = False
Save_pitch = False


###### DEFINO PARAMETROS ######
stim = 'Envelope'
###### Defino banda de eeg ######
Band = 'All'
###### Defino situacion de interes ######
situacion = 'Todo'
###### Defino estandarizacion
Normalizar = 'Stims'
Estandarizar = 'EEG'
# Defino tiempos
sr = 128
n_canales = 128
tmin, tmax = -0.53, -0.003
delays = - np.arange(np.floor(tmin*sr), np.ceil(tmax*sr), dtype=int)
times = np.linspace(delays[0]*np.sign(tmin)*1/sr, np.abs(delays[-1])*np.sign(tmax)*1/sr, len(delays))

###### Paths ######
procesed_data_path = 'saves/Preprocesed_Data/'
Run_saves_path = 'saves/'
###### Empiezo corrida ######
sesiones = []
sujeto = []
pitch_mean = []
pitch_std = []

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
    
    if Pitch: 
        for sujeto, dstims in zip((1,2), (dstims_para_sujeto_2, dstims_para_sujeto_1)):
            print('Sujeto {}'.format(sujeto))

            pitch = dstims[:,-1]
            pitch_mean.append(np.mean(pitch))
            pitch_std.append(np.std(pitch))
    else:
        for sujeto, eeg, dstims in zip((1,2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
    # for sujeto, eeg, in zip([1], [eeg_sujeto_1]):    
            print('Sujeto {}'.format(sujeto))
            
            if PSD: 
                psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(eeg.transpose(), sfreq = info['sfreq'], fmin = 4, fmax = 60)
                
                if Display_fig:plt.ion()
                else: plt.ioff()
                
                fig, ax = plt.subplots()
                fig.suptitle('Sesion {} - Sujeto {} - Situacion {}'.format(sesion, sujeto, situacion))
    
                evoked = mne.EvokedArray(psds_welch_mean, info)
                evoked.times = freqs_mean
                evoked.plot(scalings = dict(eeg=1, grad=1, mag=1), zorder = 'std', time_unit = 's', 
                show = False, spatial_colors=True, unit = False, units = 'w', axes = ax)
                ax.set_xlabel('Frequency [Hz]')
                ax.grid()
                
                if Save_fig:
                    Run_graficos_path = 'gráficos/PSD/PSD {}/'.format(situacion)
                    try: os.makedirs(Run_graficos_path)
                    except: pass
                    plt.savefig(Run_graficos_path + 'Sesion{} - Sujeto{} - Situacion {}'.format(sesion,sujeto, situacion))
                
            if Cov:    
                raw = mne.io.RawArray(eeg.transpose(), info)
                cov_mat = mne.compute_raw_covariance(raw)
                ax1, ax2 = cov_mat.plot(info)
                
                if Save_fig:
                    Run_graficos_path = 'gráficos/Covariance/Cov_{}/'.format(situacion)
                    try: os.makedirs(Run_graficos_path)
                    except: pass
                    ax1.savefig(Run_graficos_path + 'Sesion{} - Sujeto{} - {}'.format(sesion,sujeto,situacion))
                
                cov_mat_num = np.cov(eeg.transpose())
                fig = sn.heatmap(cov_mat_num, cmap = "coolwarm", center = 0)

if Save_pitch:
    f = open(Run_saves_path + 'Pitch.pkl', 'wb')
    pickle.dump([pitch_mean,pitch_std], f)
    f.close()
