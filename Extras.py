# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:38:25 2021

@author: joaco
"""
import os
import pickle
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.model_selection import KFold
from scipy import signal as sgn

import Load
import Models
import Plot
import Processing
import Simulation

# WHAT TO DO
Plot_EEG_PSD = True
Simulate_random_data = False
Cov_Matrix = False
Signal_vs_Pred = False
Phase_Align = False
Pitch = False

# Figures
Display = False
Save = True

if Display:
    plt.ion()
else:
    plt.ioff()

# Define Parameters
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Model
# Model = Ridge
alpha = 100

# Stimuli and EEG
Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Stims = ['Envelope', 'Pitch', 'Pitch_der', 'Envelope_Pitch_Pitch_der']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']

stim = 'Envelope'
Band = None
situacion = 'Todo'
tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'saves/'

# Save Variables
if Simulate_random_data:
    psd_pred_correlations, psd_rand_correlations = [], []
if Pitch:
    pitch_mean, pitch_std = [], []

# Start Run
sesiones = np.arange(21, 26)
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
sujeto_total = 0
N_Samples = []
for sesion in sesiones:
    print('Sesion {}'.format(sesion))

    # LOAD DATA BY SUBJECT
    Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                        procesed_data_path=procesed_data_path)

    # LOAD EEG BY SUBJECT
    eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']
    N_Samples.append(len(eeg_sujeto_1))
    N_Samples.append(len(eeg_sujeto_2))
    # LOAD STIMULUS BY SUBJECT
    dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
    Cant_Estimulos = len(dstims_para_sujeto_1)

    if Pitch:
        for sujeto, dstims in zip((1, 2), (dstims_para_sujeto_2, dstims_para_sujeto_1)):
            print('Sujeto {}'.format(sujeto))

            pitch = dstims[:, -1]
            pitch_mean.append(np.mean(pitch))
            pitch_std.append(np.std(pitch))
    else:
        for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                       (dstims_para_sujeto_1, dstims_para_sujeto_2)):
            # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):
            print('Sujeto {}'.format(sujeto))
            # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
            n_splits = 5

            # Ploteo PSD de señal de EEG
            if Plot_EEG_PSD: Plot.Plot_PSD(sesion, sujeto, Band, situacion, Display, Save, situacion, info,
                                           eeg.transpose())

            if Simulate_random_data or Signal_vs_Pred:
                # Empiezo el KFold de test
                kf_test = KFold(n_splits, shuffle=False)
                for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                    eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]

                    dstims_train_val = list()
                    dstims_test = list()

                    for stimulus in list(dstims):
                        dstims_train_val.append(stimulus[train_val_index])
                        dstims_test.append(stimulus[test_index])

                    axis = 0
                    porcent = 5
                    eeg, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg, dstims_train_val, dstims_test,
                                                                                         Stims_preprocess, EEG_preprocess,
                                                                                         axis=0, porcent=5)

                    # Ajusto el modelo y guardo
                    Model = Models.Ridge(alpha)
                    Model.fit(dstims_train_val, eeg_train_val)

                    # Predigo en test set y guardo
                    predicted = Model.predict(dstims_test)

                    # Calculo Correlacion y guardo
                    Rcorr = np.array(
                        [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicted[:, ii]).ravel())[0, 1] for ii in
                         range(eeg_test.shape[1])])

                    # Calculo Error y guardo
                    Rmse = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))

                    if Signal_vs_Pred:
                        # SINGAL AND PREDICTION PLOT
                        plt.ion()
                        fig = plt.figure()
                        fig.suptitle('Pearson Correlation = {}'.format(Rcorr[0]))
                        plt.plot(eeg_test[:, 0], label='Signal')
                        plt.plot(predicted[:, 0], label='Prediction')
                        plt.title('Original Signals - alpha: {}'.format(alpha))
                        plt.xlim([2000, 3000])
                        plt.xlabel('Samples')
                        plt.ylabel('Amplitude')
                        plt.grid()
                        plt.legend()
                        plt.savefig('gráficos/Ridge/Alpha/Signals_alpha{}.png'.format(alpha))
                        plt.tight_layout()
                        break

                    if Simulate_random_data:
                        fmin, fmax = 0, 40
                        # SIMULACIONES PERMUTADAS PARA COMPARAR
                        toy_iterations = 10
                        psd_rand_correlation = Simulation.simular_iteraciones_Ridge_plot(info, times, situacion, alpha,
                                                                                         toy_iterations, sesion, sujeto,
                                                                                         fold, dstims_train_val,
                                                                                         eeg_train_val, dstims_test,
                                                                                         eeg_test, fmin, fmax,
                                                                                         Display=False)
                        psd_rand_correlations.append(psd_rand_correlation)

                        # PSD of predicted EEG vs Signal
                        psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'],
                                                                                   fmin, fmax)
                        psds_pred, freqs_mean = mne.time_frequency.psd_array_welch(predicted.transpose(), info['sfreq'],
                                                                                   fmin, fmax)
                        # Ploteo PSD
                        Plot.Plot_PSD(sesion, sujeto, fold, Band, situacion, Display, Save, 'Prediccion', info,
                                      predicted.transpose())
                        Plot.Plot_PSD(sesion, sujeto, fold, Band, situacion, Display, Save, 'Test', info,
                                      eeg_test.transpose())

                        psds_channel_corr = np.array(
                            [np.corrcoef(psds_test[ii].ravel(), np.array(psds_pred[ii]).ravel())[0, 1]
                             for ii in range(len(psds_test))])
                        psd_pred_correlations.append(np.mean(psds_channel_corr))

            if Cov_Matrix:
                # Matriz de Covarianza
                raw = mne.io.RawArray(predicted.transpose(), info)
                cov_mat = mne.compute_raw_covariance(raw)
                ax1, ax2 = cov_mat.plot(info)

                try:
                    os.makedirs('gráficos/Covariance/Cov_prediccion')
                except:
                    pass
                ax1.savefig(
                    'gráficos/Covariance/Cov_prediccion/Sesion{} - Sujeto{} - {}'.format(sesion, sujeto, situacion))

                raw = mne.io.RawArray(eeg_test.transpose(), info)
                cov_mat = mne.compute_raw_covariance(raw)
                plt.ion()
                ax1, ax2 = cov_mat.plot(info)
                try:
                    os.makedirs('gráficos/Covariance/Cov_test')
                except:
                    pass
                ax1.savefig('gráficos/Covariance/Cov_test/Sesion{} - Sujeto{} - {}'.format(sesion, sujeto, situacion))

            if Phase_Align:
                analytic_signal = sgn.hilbert(eeg, axis=0)
                phase = np.angle(analytic_signal).transpose()
                average_phase_diff = np.zeros((len(phase), len(phase)))

                for channel in range(len(phase)):
                    print(channel)
                    phase_diff = np.zeros(phase.shape)
                    vector_x_diff = np.zeros(phase.shape)
                    vector_y_diff = np.zeros(phase.shape)
                    for channel_2 in range(len(phase)):
                        phase_diff[channel_2] = phase[channel] - phase[channel_2]

                    vector_x_diff = np.cos(phase_diff)
                    vector_y_diff = np.sin(phase_diff)
                    vector_diff = vector_x_diff + vector_y_diff
                    average_phase_diff[channel] = np.abs(np.mean(vector_diff, axis=-1))

                plt.figure()
                plt.imshow(average_phase_diff)
                plt.colorbar()
                plt.title('Phase sincronization {}'.format(situacion))

            sujeto_total += 1

if Pitch:
    f = open(Run_saves_path + 'Subjects_Pitch.pkl', 'wb')
    pickle.dump([pitch_mean, pitch_std], f)
    f.close()

# PSD Boxplot
if Simulate_random_data:
    Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/Alpha_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
        Stims_preprocess, EEG_preprocess, alpha, tmin, tmax, stim, Band)
    Plot.PSD_boxplot(psd_pred_correlations, psd_rand_correlations, Display, Save, Run_graficos_path)
