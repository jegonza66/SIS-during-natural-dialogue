import pickle
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import Load
import Models

import Processing


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

        # Start Run
        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
        sesiones = [21]
        sujeto_total = 0
        for sesion in sesiones:
            print('Sesion {}'.format(sesion))

            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                procesed_data_path=procesed_data_path)

            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1,
                                                                              Sujeto_2=Sujeto_2)
            Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Sujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                Predicciones = {}
                n_splits = 5
                iteraciones = 3000

                # Defino variables donde voy a guardar mil cosas
                Pesos_ronda_canales = np.zeros((len(dstims_para_sujeto_1), n_splits, info['nchan'], len(times)))
                Intercept_ronda_canales = np.zeros((len(dstims_para_sujeto_1), n_splits, info['nchan']))

                # Empiezo el KFold de test
                kf_test = KFold(n_splits, shuffle=False)
                for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                    print('\n\nFOLD {}'.format(fold))
                    eeg_train_val, eeg_test = eeg[train_val_index], eeg[test_index]

                    # ENTRENO
                    for i, stim in enumerate(list(dstims)):
                        stim_train = [stim[train_val_index]]
                        stim_test = [stim[test_index]]

                        eeg_train_val = eeg_train_val

                        axis = 0
                        porcent = 5
                        eeg_train_val, eeg_test, stim_train, stim_test = Processing.standarize_normalize(eeg_train_val,
                                                                                                         eeg_test,
                                                                                                         stim_train,
                                                                                                         stim_test,
                                                                                                         Stims_preprocess,
                                                                                                         EEG_preprocess,
                                                                                                         axis, porcent)
                        # alpha = Alphas[Band][stim][sesion][sujeto]
                        # if alpha == 'FAILED':
                        #     alpha = np.mean([value for sesion_dict in Alphas[Band][stim].keys() for value in list(Alphas[Band][stim][sesion_dict].values()) if type(value) != str])

                        alpha = 100

                        # Ajusto el modelo y guardo
                        Model = Models.Ridge(alpha)
                        Model.fit(stim_train, eeg_train_val)
                        Pesos_ronda_canales[i, fold] = Model.model.coef_
                        Intercept_ronda_canales[i, fold] = Model.model.intercept_

                        # Predigo en test set y guardo
                        predicted = Model.predict(stim_train)

                        Predicciones[fold] = predicted

                    #     plt.ion()
                    #     plt.figure()
                    #     plt.plot(eeg_train_val[:,0])
                    #     plt.plot(predicted[:,0])
                    #
                    # plt.figure()
                    # plt.plot(eeg_train_val[:,0] - predicted[:,0])

                    # TESTEO
                    predicted_final = np.zeros(eeg_test.shape)
                    for i, stim in enumerate(list(dstims)):
                        stim_train = [stim[train_val_index]]
                        stim_test = [stim[test_index]]

                        eeg_train_val, eeg_test, stim_train, stim_test = Processing.standarize_normalize(eeg_train_val,
                                                                                                         eeg_test,
                                                                                                         stim_train,
                                                                                                         stim_test,
                                                                                                         Stims_preprocess,
                                                                                                         EEG_preprocess,
                                                                                                         axis, porcent)

                        Model.model.coef_ = Pesos_ronda_canales[i, fold]
                        Model.model.intercept_ = Intercept_ronda_canales[i, fold]

                        predicted_final += Model.predict(stim_test)

                        # plt.ion()
                        # plt.figure()
                        # plt.plot(eeg_test[:,0])
                        # plt.plot(predicted_final[:,0])

                    predicted_final /= len(dstims_para_sujeto_1)

                    Rcorr = np.array(
                        [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicted_final[:, ii]).ravel())[0, 1] for ii
                         in range(eeg_test.shape[1])])
                    # Corr_buenas_ronda_canal[fold] = Rcorr

                    # Calculo Error y guardo
                    Rmse = np.array(np.sqrt(np.power((predicted_final - eeg_test), 2).mean(0)))
                    # Rmse_buenos_ronda_canal[fold] = Rmse

                    print('\n\nCorrelacion')
                    print(np.mean(Rcorr))