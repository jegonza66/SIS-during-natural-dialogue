# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:46:03 2021

@author: joaco
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import KFold
import Load
import Models
import Plot
import Processing

Display_figures_Trace = False
Save_figures_Trace = True
Display_figure_instantes, Save_figure_instantes = False, False

Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Stims = ['Envelope_Pitch_Pitch_der']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']

# DEFINO PARAMETROS
for Band in Bands:
    print(Band + '\n')
    # stim = 'Envelope'
    # Defino banda de eeg
    for stim in Stims:
        print(stim + '\n')
        # Band = 'Theta'
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

        min_busqueda, max_busqueda = -2, 6
        pasos = 16
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)

        Standarized_Betas = []
        Correlaciones = np.zeros(len(alphas))
        Errores = np.zeros(len(alphas))

        sesiones = np.arange(21, 26)
        # Empiezo corrida
        for alpha_num, alpha in enumerate(alphas):
            print('Alpha: {}'.format(alpha))
            sujeto_total = 0
            for sesion in sesiones:
                print('Sesion {}'.format(sesion))
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
                    # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):
                    print('Sujeto {}'.format(sujeto))

                    # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                    Predicciones = {}
                    n_splits = 5
                    iteraciones = 3000

                    # Defino variables donde voy a guardar mil cosas
                    Pesos_ronda_canales = np.zeros((n_splits, n_canales, len(delays) * Cant_Estimulos))
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
                        Predicciones[fold] = predicted

                        # Calculo Correlacion y guardo
                        Rcorr = np.array(
                            [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicted[:, ii]).ravel())[0, 1] for ii in
                             range(eeg_test.shape[1])])
                        Corr_buenas_ronda_canal[fold] = Rcorr

                        # Calculo Error y guardo
                        Rmse = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                        Rmse_buenos_ronda_canal[fold] = Rmse

                    # Tomo promedio de Corr y Rmse entre las rondas de test para todos los canales (para no
                    # desvirtuar las cabezas)
                    Corr_promedio = Corr_buenas_ronda_canal.mean(0)
                    Rmse_promedio = Rmse_buenos_ronda_canal.mean(0)
                    Pesos_promedio = Pesos_ronda_canales.mean(0)

                    # Guardo las correlaciones y los pesos promediados entre test de cada canal del sujeto y lo
                    # adjunto a lista para promediar entre canales de sujetos
                    if not sujeto_total:
                        Pesos_totales_sujetos_todos_canales = Pesos_promedio
                        Correlaciones_totales_sujetos = Corr_promedio
                        Rmse_totales_sujetos = Rmse_promedio
                    else:
                        # if len(Canales_sobrevivientes_corr):
                        Pesos_totales_sujetos_todos_canales = np.dstack(
                            (Pesos_totales_sujetos_todos_canales, Pesos_promedio))
                        Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
                        Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))
                    sujeto_total += 1

            curva_pesos_totales = np.array(Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, Band,
                                                                   times, sr, Display_figure_instantes,
                                                                   Save_figure_instantes, Run_graficos_path,
                                                                   Cant_Estimulos, Stims_Order, stim))

            Standarized_Betas.append(curva_pesos_totales.ravel())
            Correlaciones[alpha_num] = Correlaciones_totales_sujetos.mean()
            Errores[alpha_num] = Rmse_totales_sujetos.mean()

        Standarized_Betas = np.array(Standarized_Betas)
        if Display_figures_Trace:
            plt.ion()
        else:
            plt.ioff()

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 13))
        fig.suptitle('Ridge Trace - {} - {}'.format(Band, stim))
        plt.xlabel('Ridge Parameter')
        plt.xscale('log')

        axs[0].set_ylabel('Standarized Coefficents')
        axs[0].plot(alphas, Standarized_Betas[:, :20], 'o--')
        axs[0].plot(alphas, Standarized_Betas[:, -20:], 'o--')
        axs[0].grid()

        axs[1].set_ylabel('Mean Correlation')
        axs[1].plot(alphas, Correlaciones, 'o--')
        axs[1].errorbar(alphas, Correlaciones, yerr=np.std(Correlaciones), fmt='none', ecolor='black', elinewidth=0.5,
                        capsize=0.5)
        axs[1].grid()

        axs[2].set_ylabel('Mean Error')
        axs[2].plot(alphas, Errores, 'o--')
        axs[2].errorbar(alphas, Errores, yerr=np.std(Errores), fmt='none', ecolor='black', elinewidth=0.5, capsize=0.5)
        axs[2].grid()

        fig.tight_layout()

        if Save_figures_Trace:
            save_path = Run_graficos_path + '{}/'.format(Band)
            try:
                os.makedirs(save_path)
            except:
                pass
            plt.savefig(save_path + '{}.png'.format(stim))
