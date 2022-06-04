import os
import pickle
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.model_selection import KFold

import Load_Envelope as Load
import Models
import Plot
import Processing
import Permutations

# WHAT TO DO
Plot_EEG_PSD = False
Simulate_random_data = False
Signal_vs_Pred = False
Pitch = False

# Figures
Display = False
Save = False
if Display:
    plt.ion()
else:
    plt.ioff()

# Define Parameters
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Model
set_alpha = None
Corr_limit = 0.01
alphas_fname = 'saves/Alphas/Alphas_Corr{}.pkl'.format(Corr_limit)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

# Stimuli and EEG
Stims = ['Envelope', 'Pitch', 'Spectrogram', 'Envelope_Pitch_Spectrogram']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']

stim = 'Envelope'
Band = 'Theta'
situacion = 'Escucha'
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'saves/'
graficos_save_path = 'gráficos/Signals_vs_Pred/tmin{}_tmax{}/'.format(tmin, tmax, stim, Band)


if Simulate_random_data:
    psd_pred_correlations, psd_rand_correlations = [], []
if Pitch:
    pitch_mean, pitch_std = [], []

sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
sujeto_total = 0
N_Samples = []
for sesion in sesiones:
    print('\nSesion {}'.format(sesion))

    if Pitch:
        # LOAD DATA BY SUBJECT
        Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                            procesed_data_path=procesed_data_path, situacion=situacion)
        dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos('Pitch', Sujeto_1, Sujeto_2)
        for sujeto, dstims in zip((1, 2), (dstims_para_sujeto_2, dstims_para_sujeto_1)):
            print('Sujeto {}'.format(sujeto))

            pitch = dstims[0][1:, -1]
            pitch_mean.append(np.mean(pitch))
            pitch_std.append(np.std(pitch))

    else:
        # LOAD DATA BY SUBJECT
        Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                            procesed_data_path=procesed_data_path, situacion=situacion)
        # LOAD EEG BY SUBJECT
        eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']
        N_Samples.append(len(eeg_sujeto_1))
        N_Samples.append(len(eeg_sujeto_2))
        # LOAD STIMULUS BY SUBJECT
        dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
        Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

        for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                       (dstims_para_sujeto_1, dstims_para_sujeto_2)):
            # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):
            print('\nSujeto {}'.format(sujeto))
            # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
            n_splits = 5

            # Ploteo PSD de señal de EEG
            if Plot_EEG_PSD:
                fmin, fmax = 0, 40
                Plot.Plot_PSD(sesion, sujeto, Band, situacion, Display, Save, situacion, info,
                              eeg.transpose(), fmin, fmax)

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
                    eeg_train_val, eeg_test, dstims_train_val, dstims_test = \
                        Processing.standarize_normalize(eeg_train_val, eeg_test, dstims_train_val, dstims_test,
                                                        Stims_preprocess, EEG_preprocess, axis, porcent)

                    # Ajusto el modelo y guardo
                    if set_alpha == None:
                        try:
                            alpha = Alphas[Band][stim][sesion][sujeto]
                        except:
                            alpha = 1000
                            print('Alpha missing. Ussing default value: {}'.format(alpha))
                    else:
                        alpha = set_alpha

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
                        eeg_x = np.linspace(0, len(eeg_test) / sr, len(eeg_test))
                        # SINGAL AND PREDICTION PLOT
                        fig = plt.figure()
                        fig.suptitle('Original prediction')
                        plt.title('Pearson Correlation = {}'.format(Rcorr[0]))
                        plt.plot(eeg_x, eeg_test[:, 0], label='Signal')
                        plt.plot(eeg_x, predicted[:, 0], label='Prediction')
                        plt.xlim([18, 26])
                        plt.ylim([-3, 3])
                        plt.xlabel('Time [ms]')
                        plt.ylabel('Amplitude')
                        plt.grid()
                        plt.legend()
                        plt.tight_layout()

                        os.makedirs(graficos_save_path + 'Original/Stim_{}_EEG_Band{}/'.format(stim, Band),
                                    exist_ok=True)
                        plt.savefig(
                            graficos_save_path + 'Original/Stim_{}_EEG_Band{}/Sesion{}_Sujeto{}.png'.format(stim, Band,
                                                                                                            sesion,
                                                                                                            sujeto))
                        plt.savefig(
                            graficos_save_path + 'Original/Stim_{}_EEG_Band{}/Sesion{}_Sujeto{}.svg'.format(stim, Band,
                                                                                                            sesion,
                                                                                                            sujeto))

                    if Simulate_random_data:
                        fmin, fmax = 3, 25
                        # SIMULACIONES PERMUTADAS PARA COMPARAR
                        toy_iterations = 1
                        psd_rand_correlation = Permutations.simular_iteraciones_Ridge_plot(info, times, situacion,
                                                                                           alpha,
                                                                                           toy_iterations, sesion,
                                                                                           sujeto,
                                                                                           fold, dstims_train_val,
                                                                                           eeg_train_val, dstims_test,
                                                                                           eeg_test, fmin, fmax, stim,
                                                                                           Band,
                                                                                           save_path=False,
                                                                                           Display=False)
                        psd_rand_correlations.append(psd_rand_correlation)

                        # PSD of predicted EEG vs Signal
                        psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'],
                                                                                   fmin, fmax)
                        psds_pred, freqs_mean = mne.time_frequency.psd_array_welch(predicted.transpose(), info['sfreq'],
                                                                                   fmin, fmax)
                        # Ploteo PSD
                        Plot.Plot_PSD(sesion, sujeto, Band, situacion, Display, Save, 'Prediccion', info,
                                      predicted.transpose(), fmin, fmax)
                        Plot.Plot_PSD(sesion, sujeto, Band, situacion, Display, Save, 'Test', info,
                                      eeg_test.transpose(), fmin, fmax)

                        # Calculate channelwise correlation between prediction and signal psd
                        psds_channel_corr = np.array(
                            [np.corrcoef(psds_test[ii].ravel(), np.array(psds_pred[ii]).ravel())[0, 1]
                             for ii in range(len(psds_test))])
                        # save average of channels psd correlation
                        psd_pred_correlations.append(np.mean(psds_channel_corr))

if Pitch:
    f = open(Run_saves_path + 'Subjects_Pitch.pkl', 'wb')
    pickle.dump([pitch_mean, pitch_std], f)
    f.close()

# PSD Boxplot
if Simulate_random_data:
    Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
        Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
    Plot.PSD_boxplot(psd_pred_correlations, psd_rand_correlations, Display, Save, Run_graficos_path)
