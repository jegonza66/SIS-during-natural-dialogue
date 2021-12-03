import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

import Load
import Models
import Plot
import Processing
import Permutations

from datetime import datetime
startTime = datetime.now()

# Define Parameters
# Model parameters
tmin, tmax = -0.6, -0.003
sr = 128

# Stimuli and EEG
Stims = ['Envelope', 'Pitch', 'Spectrogram', 'Envelope_Pitch', 'Envelope_Spectrogram', 'Pitch_Spectrogram', 'Envelope_Pitch_Spectrogram']
# Stims = ['Envelope']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', (4,6), (1,15)]
Bands = ['Theta']

# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Files
alphas_fname = 'saves/Alphas/Alphas_Trace{:.1f}_Corr0.025.pkl'.format(2 / 3)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

for Band in Bands:
    print('\n{}\n'.format(Band))
    for stim in Stims:
        print('\n' + stim + '\n')
        # Paths
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Path_it = 'saves/Ridge/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

        # Start Run
        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
        # sesiones = [21]
        sujeto_total = 0
        for sesion in sesiones:
            print('Sesion {}'.format(sesion))
            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                procesed_data_path=procesed_data_path)
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
                Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))


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
                    try:
                        alpha = Alphas[Band][stim][sesion][sujeto]
                        if alpha == 'FAILED':
                            alpha = np.mean([value for sesion_dict in Alphas[Band][stim].keys() for value in list(Alphas[Band][stim][sesion_dict].values()) if type(value) != str])
                    except:
                        alpha = 100
                        print('Alpha missing. Ussing default value: {}'.format(alpha))
                        break
                        break

                    # Run_permutations:
                    Fake_Model = Models.Ridge(alpha)
                    Pesos_fake, Correlaciones_fake, Errores_fake = \
                        Permutations.simular_iteraciones_Ridge(Fake_Model, alpha, iteraciones, sesion, sujeto, fold,
                        dstims_train_val, eeg_train_val, dstims_test, eeg_test,
                        Pesos_fake, Correlaciones_fake, Errores_fake)

                # Save permutations
                if Run_permutations:
                    try:
                        os.makedirs(Path_it)
                    except:
                        pass
                    f = open(Path_it + 'Corr_Rmse_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                    pickle.dump([Correlaciones_fake, Errores_fake], f)
                    f.close()

                    f = open(Path_it + 'Pesos_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                    pickle.dump(Pesos_fake.mean(0), f)
                    f.close()

print(datetime.now() - startTime)