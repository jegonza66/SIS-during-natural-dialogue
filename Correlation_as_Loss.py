import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize as minimizador
import matplotlib.pyplot as plt

import Funciones
import Load
import Models
import Plot
import Processing
import Simulation

from datetime import datetime

# Random permutations
Statistical_test = False
Run_permutations = False

# Figures
Display_Ind_Figures = False
Display_Total_Figures = True

Save_Ind_Figures = False
Save_Total_Figures = False

Save_Final_Correlation = False

# Define Parameters
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Stimuli and EEG
Stims_Order = ['Envelope', 'Pitch', 'Spectrogram', 'Phonemes']
Stims = ['Envelope', 'Pitch', 'Envelope_Pitch']
Stims = ['Envelope']
Bands = ['Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']

# Model parameters
alphas_fname = 'saves/Alphas/Alphas_Trace{:.1f}_Corr0.025.pkl'.format(2 / 3)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

f = open('saves/Subjects_Pitch.pkl', 'rb')
subjects_pitch = pickle.load(f)
f.close()

tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

for Band in Bands:
    print('\n{}\n'.format(Band))
    for stim in Stims:
        print('\n' + stim + '\n')
        # Paths
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_origial = 'saves/Ridge/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_it = 'saves/Ridge/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

        # Start Run
        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
        sesiones = [21]
        sujeto_total = 0
        for sesion in sesiones:
            print('Sesion {}'.format(sesion))

            startTime = datetime.now()
            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                procesed_data_path=procesed_data_path)
            print(datetime.now() - startTime)

            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1, Sujeto_2=Sujeto_2)
            Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Sujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                Predicciones = {}
                n_folds = 5
                iteraciones = 1000

                # Defino variables donde voy a guardar mil cosas
                Pesos_ronda_canales = np.zeros((n_folds, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Corr_buenas_ronda_canal = np.zeros((n_folds, info['nchan']))
                Rmse_buenos_ronda_canal = np.zeros((n_folds, info['nchan']))

                Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))

                Prob_Corr_ronda_canales = np.ones((n_folds, info['nchan']))
                Prob_Rmse_ronda_canales = np.ones((n_folds, info['nchan']))

                Canales_repetidos_corr_sujeto = np.zeros(info['nchan'])
                Canales_repetidos_rmse_sujeto = np.zeros(info['nchan'])

                # Empiezo el KFold de test
                kf_test = KFold(n_folds, shuffle=False)
                for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                    eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]

                    dstims_train_val = list()
                    dstims_test = list()

                    for stimulus in list(dstims):
                        dstims_train_val.append(stimulus[train_val_index])
                        dstims_test.append(stimulus[test_index])

                    axis = 0
                    porcent = 5
                    eeg_train_val, eeg_test, dstims_train_val, dstims_test = Processing.standarize_normalize(eeg_train_val, eeg_test, dstims_train_val,
                                                                                         dstims_test,
                                                                                         Stims_preprocess,
                                                                                         EEG_preprocess,
                                                                                         axis, porcent)
                    # alpha = Alphas[Band][stim][sesion][sujeto]
                    # if alpha == 'FAILED':
                    #     alpha = np.mean([value for sesion_dict in Alphas[Band][stim].keys() for value in list(Alphas[Band][stim][sesion_dict].values()) if type(value) != str])
                    alpha = 0
                    xo = np.random.rand(len(times))
                    res = minimizador(Funciones.f_loss_Corr, xo, args=(dstims_train_val, eeg_train_val[:,0], alpha))

                    plt.ion()
                    plt.figure()
                    plt.plot(res.x)
                    plt.title('New model Weigths')

                    # Predigo en test set y guardo
                    predicted_new_model_corr = np.dot(dstims_test, res.x)
                    Rcorr_new = np.array(
                        np.corrcoef(eeg_test[:, 0].ravel(), np.array(predicted_new_model_corr).ravel())[0, 1])
                    Rmse_new = np.array(np.sqrt(np.power((predicted_new_model_corr - eeg_test[:, 0]), 2).mean(0)))

                    Model = Models.Ridge(alpha)
                    Model.fit(dstims_train_val, eeg_train_val[:, 0])
                    predicted = Model.predict(dstims_test)

                    Predicciones[fold] = predicted

                    # Calculo Correlacion y guardo
                    Rcorr = np.array(
                        np.corrcoef(eeg_test[:, 0].ravel(), np.array(predicted).ravel())[0, 1])
                    Corr_buenas_ronda_canal[fold] = Rcorr

                    # Calculo Error y guardo
                    Rmse = np.array(np.sqrt(np.power((predicted - eeg_test[:,0]), 2).mean(0)))
                    Rmse_buenos_ronda_canal[fold] = Rmse

print(datetime.now() - startTime)