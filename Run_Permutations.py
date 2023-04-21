import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

import New_load as Load
import Models
import Processing
import Permutations


startTime = datetime.now()

# Define Parameters
tmin, tmax = -0.4, 0.2
sr = 128
n_folds = 5
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
situacion = 'Escucha'
# Model parameters
model = 'Decoding'

if model == 'Ridge':
    iteraciones = 3000

elif model == 'Decoding':
    iteraciones = 200
    Max_t_lags_fname = 'Saves/Decoding_t_lag/{}/Final_Correlation/tmin{}_tmax{}/Max_t_lags.pkl'.format(situacion, tmin,
                                                                                                       tmax)
    try:
        f = open(Max_t_lags_fname, 'rb')
        Max_t_lags = pickle.load(f)
        f.close()
    except:
        print('\n\nMean_Correlations file not found\n\n')
        Max_t_lags = {}

# Stimuli and EEG
Stims = ['Envelope']
Bands = ['Alpha', 'Beta_1', 'All']

# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Model
Corr_limit = 0.01
alphas_fname = 'Saves/Alphas/Alphas_Corr{}.pkl'.format(Corr_limit)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

for Band in Bands:
    for stim in Stims:
        print('\nModel: ' + model)
        print('Band: ' + Band)
        print('Stimulus: ' + stim)
        print('Status: ' + situacion)
        print('tmin: {} - tmax: {}'.format(tmin, tmax))
        # Paths
        procesed_data_path = 'Saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Path_it = 'Saves/{}/{}/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            model, situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

        # Start Run
        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
        sujeto_total = 0
        for sesion in sesiones:
            print('\nSesion {}'.format(sesion))
            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                    procesed_data_path=procesed_data_path, situacion=situacion)
            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1, Sujeto_2=Sujeto_2)
            Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

            # Defino variables donde voy a guardar mil cosas
            if model == 'Ridge':
                Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)),
                                      dtype=np.float16)
                Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))

            elif model == 'Decoding':
                Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)),
                                      dtype=np.float16)
                Patterns_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)),
                                         dtype=np.float16)
                Correlaciones_fake = np.zeros((n_folds, iteraciones))
                Errores_fake = np.zeros((n_folds, iteraciones))

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Sujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                n_folds = 5

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
                    eeg_train_val, eeg_test, dstims_train_val, dstims_test = Processing.standarize_normalize(
                        eeg_train_val, eeg_test, dstims_train_val,
                        dstims_test,
                        Stims_preprocess,
                        EEG_preprocess,
                        axis, porcent)
                    try:
                        alpha = Alphas[Band][stim][sesion][sujeto]
                    except:
                        alpha = 1000
                        print('Alpha missing. Ussing default value: {}'.format(alpha))

                    # Run_permutations:
                    if model == 'Ridge':
                        Fake_Model = Models.Ridge(alpha)
                        Pesos_fake, Correlaciones_fake, Errores_fake = \
                            Permutations.simular_iteraciones_Ridge(Fake_Model, iteraciones, sesion, sujeto, fold,
                                                                   dstims_train_val, eeg_train_val, dstims_test,
                                                                   eeg_test, Pesos_fake, Correlaciones_fake,
                                                                   Errores_fake)
                    elif model == 'Decoding':
                        t_lag = np.where(times == Max_t_lags[Band])[0][0]
                        Fake_Model = Models.mne_mtrf_decoding(tmin, tmax, sr, info, alpha, t_lag)
                        Pesos_fake, Patterns_fake, Correlaciones_fake, Errores_fake = \
                            Permutations.simular_iteraciones_decoding(Fake_Model, iteraciones, sesion, sujeto, fold,
                                                                   dstims_train_val, eeg_train_val, dstims_test,
                                                                   eeg_test, Pesos_fake, Patterns_fake, Correlaciones_fake,
                                                                   Errores_fake)

                # Save permutations
                os.makedirs(Path_it, exist_ok=True)
                f = open(Path_it + 'Corr_Rmse_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Correlaciones_fake, Errores_fake], f)
                f.close()

                f = open(Path_it + 'Pesos_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump(Pesos_fake.mean(0), f)
                f.close()

                if model == 'Decoding':
                    f = open(Path_it + 'Patterns_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                    pickle.dump(Patterns_fake.mean(0), f)
                    f.close()

print('\n')
print(datetime.now() - startTime)
