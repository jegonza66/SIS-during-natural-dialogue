# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:23:54 2021

@author: joaco
"""
import numpy as np
from sklearn.model_selection import KFold

import Load
import Models
import Processing

def run_pipeline(Stims_preprocess='Normalize', EEG_preprocess='Standarize', stim='Envelope', Band='Theta',
                 situacion='Escucha', tmin=-0.6, tmax=-0.003, sr=128,
                 procesed_data_path='saves/Preprocesed_Data/tmin{}_tmax{}/'.format(-0.6, -0.003), alpha=100
                 ):
    sesiones = np.arange(21, 26)
    sujeto_total = 0
    for sesion in sesiones:
        # print('Sesion {}'.format(sesion))

        # LOAD DATA BY SUBJECT
        Sujeto_1, Sujeto_2 = Load.Load_Data(sesion, Band, sr, tmin, tmax, situacion, procesed_data_path)

        # LOAD EEG BY SUBJECT
        eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']

        # LOAD STIMULUS BY SUBJECT
        dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
        Cant_Estimulos = len(dstims_para_sujeto_1)

        for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                       (dstims_para_sujeto_1, dstims_para_sujeto_2)):
            # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
            # print('Sujeto {}'.format(sujeto))
            # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
            n_splits = 5

            # Defino variables donde voy a guardar mil cosas
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
                eeg, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg, dstims_train_val, dstims_test,
                                                                                     Stims_preprocess, EEG_preprocess,
                                                                                     axis, porcent)

                # Ajusto el modelo y guardo
                Model = Models.Ridge(alpha)
                Model.fit(dstims_train_val, eeg_train_val)

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

            # Tomo promedio de pesos Corr y Rmse entre los folds para todos los canales
            Corr_promedio = Corr_buenas_ronda_canal.mean(0)
            Rmse_promedio = Rmse_buenos_ronda_canal.mean(0)

            # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
            # para promediar entre canales de sujetos
            if not sujeto_total:
                Correlaciones_totales_sujetos = Corr_promedio
                Rmse_totales_sujetos = Rmse_promedio
            else:
                Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
                Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))

            sujeto_total += 1

    return np.mean(Correlaciones_totales_sujetos), np.mean(Rmse_totales_sujetos)