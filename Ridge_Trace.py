import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
import Load
import Models
import Processing
import Funciones

Display_figures_Trace = False
Save_figures_Trace = True

Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Stims = ['Envelope', 'Pitch', 'Pitch_der', 'Envelope_Pitch_Pitch_der']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', (1,15), 'All']

Trace_interval = 2/3
min_trace_derivate = 0
Corr_limit = 0.025

alphas_fname = 'saves/Alphas/Alphas_Trace{}_Corr{}.pkl'.format(Trace_interval, Corr_limit)
failed_fname = 'saves/Alphas/Failed_Trace{}_Corr{}.pkl'.format(Trace_interval, Corr_limit)

try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    Alphas = {}

failed_subjects = []

# DEFINO PARAMETROS
for Band in Bands:
    print('\n{}\n'.format(Band))
    Alphas_Band = {}
    for stim in Stims:
        print(stim + '\n')
        Alphas_Stim = {}
        # Defino situacion de interes
        situacion = 'Escucha'
        # Defino estandarizacion
        Stims_preprocess = 'Normalize'
        EEG_preprocess = 'Standarize'
        # Defino tiempos
        sr = 128
        n_canales = 128
        tmin, tmax = -0.6, -0.003
        delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
        times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr,
                            len(delays))

        # Paths
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/Ridge_Trace/Stims_{}_EEG_{}/tmin{}_tmax{}/'.format(Stims_preprocess,
                                                                                         EEG_preprocess, tmin, tmax)

        min_busqueda, max_busqueda = -1, 6
        pasos = 32
        alphas_swept = np.logspace(min_busqueda, max_busqueda, pasos)
        alpha_step = np.diff(np.log(alphas_swept))[0]

        sesiones = np.arange(21, 26)
        # Empiezo corrida

        sujeto_total = 0
        for sesion in sesiones:
            print('Sesion {}'.format(sesion))
            Alphas_Sesion = {}
            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion, Band, sr, tmin, tmax, situacion, procesed_data_path,
                                                sujeto_total)

            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
            Cant_Estimulos = len(dstims_para_sujeto_1)

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('\nSujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                n_splits = 5

                Standarized_Betas = np.zeros(len(alphas_swept))
                Correlaciones = np.zeros(len(alphas_swept))
                Std_Corr = np.zeros(len(alphas_swept))
                Errores = np.zeros(len(alphas_swept))

                for alpha_num, alpha in enumerate(alphas_swept):
                    # print('Alpha: {}'.format(alpha))

                    # Defino variables donde voy a guardar cosas para el alpha
                    Pesos_ronda_canales = np.zeros((n_splits, info['nchan'], len(delays) * Cant_Estimulos))
                    Corr_buenas_ronda_canal = np.zeros((n_splits, info['nchan']))
                    Rmse_buenos_ronda_canal = np.zeros((n_splits, info['nchan']))

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
                        eeg, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg, dstims_train_val,
                                                                                             dstims_test,
                                                                                             Stims_preprocess,
                                                                                             EEG_preprocess,
                                                                                             axis=0, porcent=5)

                        # Ajusto el modelo y guardo
                        Model = Models.Ridge(alpha)
                        Model.fit(dstims_train_val, eeg_train_val)
                        Pesos_ronda_canales[fold] = Model.coefs

                        # Predigo en test set y guardo
                        predicted = Model.predict(dstims_test)

                        # Calculo Correlacion y guardo
                        Rcorr = np.array(
                            [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicted[:, ii]).ravel())[0, 1] for ii in
                             range(eeg_test.shape[1])])
                        Corr_buenas_ronda_canal[fold] = Rcorr

                        # Calculo Error y guardo
                        Rmse = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                        Rmse_buenos_ronda_canal[fold] = Rmse

                    Correlaciones[alpha_num] = Corr_buenas_ronda_canal.mean()
                    Std_Corr[alpha_num] = Corr_buenas_ronda_canal.std()
                    Standarized_Betas[alpha_num] = np.sum(abs(Pesos_ronda_canales).mean(0).mean(0))
                    Trace_derivate = np.diff(Standarized_Betas) / alpha_step
                    Trace_derivate_2 = np.diff(Trace_derivate) / alpha_step

                    print("\rProgress: {}%".format(int((alpha_num + 1)*100/pasos)), end='')
                # Individual Ridge Trace
                Trace_range = np.where(Trace_derivate_2 < min_trace_derivate)[0] + 1  # +1 because 2nd derivate is defined in intervales
                Corr_range = np.where(Correlaciones.max() - Correlaciones < Correlaciones.max() * Corr_limit)[0]

                Overlap = sorted(set(Trace_range).intersection(set(Corr_range)), key=list(Trace_range).index)
                Trace_range_med = None
                try:
                    Trace_range_med = int(len(Trace_range) * Trace_interval + Trace_range[0])
                except:
                    pass

                try:
                    if Overlap and Trace_range_med in Corr_range:
                        Alpha_Sujeto = alphas_swept[Trace_range_med]

                    else:
                        intervals = [[Corr_range.min(), Corr_range.max()], [Trace_range_med, Trace_range_med]]
                        gap = Funciones.findFreeinterval(intervals)
                        values = set(gap[0])

                        # Interpolo (alphas vs indice)
                        alpha_index = np.mean(list(values))
                        decimal_part = alpha_index % 1
                        alpha_extra = decimal_part * alpha_step
                        Alpha_Sujeto = alphas_swept[int(alpha_index)] + alpha_extra

                except:
                    Alpha_Sujeto = 'FAILED'

                if Display_figures_Trace:
                    plt.ion()
                else:
                    plt.ioff()

                fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 13))
                fig.suptitle('Ridge Trace - {} - {}'.format(Band, stim))
                plt.xlabel('Ridge Parameter')
                plt.xscale('log')

                axs[0].set_ylabel('Standarized Coefficents')
                axs[0].plot(alphas_swept, Standarized_Betas, 'o--')
                if Alpha_Sujeto != 'FAILED': axs[0].vlines(Alpha_Sujeto, axs[0].get_ylim()[0], axs[0].get_ylim()[1], linestyle='dashed',
                              color='black', linewidth=1.5)
                if Trace_range.size: axs[0].axvspan(alphas_swept[Trace_range[0]], alphas_swept[Trace_range[-1]], alpha=0.4, color='grey',
                               label='Trace range')
                if Overlap:
                    axs[0].axvspan(alphas_swept[Overlap[0]], alphas_swept[Overlap[-1]], alpha=0.4, color='green',
                                   label='Overlap')
                axs[0].grid()
                axs[0].legend()

                axs[1].set_ylabel('Mean Correlation')
                axs[1].plot(alphas_swept, Correlaciones, 'o--')
                axs[1].errorbar(alphas_swept, Correlaciones, yerr=Std_Corr, fmt='none', ecolor='black',
                                elinewidth=0.5, capsize=0.5)
                if Alpha_Sujeto != 'FAILED': axs[1].vlines(Alpha_Sujeto, axs[1].get_ylim()[0], axs[1].get_ylim()[1], linestyle='dashed',
                              color='black', linewidth=1.5)
                if Corr_range.size: axs[1].axvspan(alphas_swept[Corr_range[0]], alphas_swept[Corr_range[-1]], alpha=0.4, color='grey',
                               label='Corr range')
                if Overlap:
                    axs[1].axvspan(alphas_swept[Overlap[0]], alphas_swept[Overlap[-1]], alpha=0.4, color='green',
                                   label='Overlap')
                axs[1].grid()
                axs[1].legend()

                fig.tight_layout()

                if Save_figures_Trace:
                    save_path = Run_graficos_path + 'Band_{}/Stim_{}/'.format(Band, stim, )
                    try:
                        os.makedirs(save_path)
                    except:
                        pass
                    plt.savefig(save_path + 'Sesion_{}_Sujeto_{}.png'.format(sesion, sujeto))

                Alphas_Sesion[sujeto] = Alpha_Sujeto
            Alphas_Stim[sesion] = Alphas_Sesion
        Alphas_Band[stim] = Alphas_Stim
    Alphas[Band] = Alphas_Band

# Save Alphas
f = open(alphas_fname, 'wb')
pickle.dump(Alphas, f)
f.close()

