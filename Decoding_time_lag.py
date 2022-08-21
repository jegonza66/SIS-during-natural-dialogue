import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

import Load_light as Load
import Models
import Plot
import Processing

from datetime import datetime
startTime = datetime.now()

# Define Parameters
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = list(np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays)))
# take lower time rolution to run faster
skip = 2
times = times[::skip]
t_lags = np.arange(len(times))*skip
situacion = 'Escucha'

# Model parameters
set_alpha = None
Alpha_Corr_limit = 0.01
alphas_fname = 'saves/Alphas/Alphas_Corr{}.pkl'.format(Alpha_Corr_limit)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

# Stimuli and EEG
Stims = ['Envelope']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'All']

# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Random permutations
Statistical_test = False

# Save / Display Figures
Display_Ind_Figures = False
Display_Total_Figures = False
Save_Ind_Figures = True
Save_Total_Figures = True
Save_Final_Correlation = True

# Save mean correlations
Mean_Correlations_fname = 'saves/Decoding_t_lag/{}/Final_Correlation/tmin{}_tmax{}/Mean_Correlations.pkl'.format(situacion, tmin, tmax)
try:
    f = open(Mean_Correlations_fname, 'rb')
    Mean_Correlations = pickle.load(f)
    f.close()
except:
    print('\n\nMean_Correlations file not found\n\n')
    Mean_Correlations = {}

f = open('saves/Subjects_Pitch.pkl', 'rb')
subjects_pitch = pickle.load(f)
f.close()

for Band in Bands:
    try:
        Mean_Correlations_Band = Mean_Correlations[Band]
    except:
        Mean_Correlations_Band = {}
    for stim in Stims:
        print('\nBand: ' + Band)
        print('Stimulus: ' + stim)
        print('Status: ' + situacion)
        print('tmin: {} - tmax: {}'.format(tmin, tmax))
        # Paths
        save_path = 'saves/Decoding_t_lag/{}/Final_Correlation/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/Decoding_t_lag/{}/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_origial = 'saves/Decoding_t_lag/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

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

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Sujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                n_folds = 5
                iteraciones = 100

                # Defino variables donde voy a guardar cosas
                Corr_buenas_ronda = np.zeros((n_folds, len(t_lags)))

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
                    eeg_train_val, eeg_test, dstims_train_val, dstims_test = \
                        Processing.standarize_normalize(eeg_train_val, eeg_test, dstims_train_val, dstims_test,
                                                        Stims_preprocess, EEG_preprocess, axis, porcent)
                    if set_alpha == None:
                        try:
                            alpha = Alphas[Band][stim][sesion][sujeto]
                        except:
                            alpha = 1000
                            print('Alpha missing. Ussing default value: {}'.format(alpha))
                    else:
                        alpha = set_alpha

                    for i, t_lag in enumerate(t_lags):
                        # Ajusto el modelo y guardo
                        Model = Models.mne_mtrf_decoding(tmin, tmax, sr, info, alpha, t_lag)
                        Model.fit(eeg_train_val, dstims_train_val)

                        # Predigo en test set y guardo
                        predicted = Model.predict(eeg_test)

                        # Calculo Correlacion y guardo
                        Rcorr = np.array(
                            [np.corrcoef(dstims_test[:, t_lag].ravel(), np.array(predicted).ravel())[0, 1]])
                        Corr_buenas_ronda[fold, i] = Rcorr

                # Save Model Weights and Correlations
                os.makedirs(Path_origial, exist_ok=True)
                f = open(Path_origial + 'Corr_Rmse_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Corr_buenas_ronda], f)
                f.close()

                # Grafico cabezas y canales
                Plot.corr_sujeto_decoding(sesion, sujeto, Corr_buenas_ronda.mean(0), Display_Ind_Figures, 'Correlation', Save_Ind_Figures, Run_graficos_path)

                # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
                # para promediar entre canales de sujetos
                if not sujeto_total:
                    Correlaciones_totales_sujetos = Corr_buenas_ronda # Save correlation per timelag
                else:
                    Correlaciones_totales_sujetos = np.dstack((Correlaciones_totales_sujetos, Corr_buenas_ronda))

                sujeto_total += 1

        # Plots
        Plot.decoding_t_lags(Correlaciones_totales_sujetos, times, Band, Display_Total_Figures, Save_Total_Figures,
                             Run_graficos_path)

        # SAVE FINAL CORRELATION
        Mean_Correlations[Band] = Mean_Correlations_Band
        if Save_Final_Correlation and sujeto_total == 18:
            os.makedirs(save_path, exist_ok=True)
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump(Correlaciones_totales_sujetos, f)
            f.close()

            f = open(Mean_Correlations_fname, 'wb')
            pickle.dump(Mean_Correlations, f)
            f.close()

print(datetime.now() - startTime)

## Run from load
# Define Parameters
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
# take lower time rolution to run faster
skip = 2
times = times[::skip]
t_lags = np.arange(len(times))*skip
situacion = 'Escucha'

Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

stim = 'Envelope'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

Display_Total_Figures = True
Save_Total_Figures = True

max_t_lags = {}

for Band in Bands:
    print('\n' + Band)
    save_path = 'saves/Decoding_t_lag/{}/Final_Correlation/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
    Run_graficos_path = 'gráficos/Decoding_t_lag/{}/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
        situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

    f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'rb')
    Correlaciones_totales_sujetos = pickle.load(f)
    f.close()

    Corr_time_sub = Correlaciones_totales_sujetos.mean(0)
    mean_time_corr = Corr_time_sub.mean(1)

    # get max correlation t_lag
    max_t_lag = np.argmax(mean_time_corr)
    print(times[max_t_lag])
    print(max_t_lag)

    max_t_lags[Band] = times[max_t_lag]

    # Plots
    Plot.decoding_t_lags(Correlaciones_totales_sujetos, times, Band, Display_Total_Figures, Save_Total_Figures,
                         Run_graficos_path)

f = open(save_path + 'Max_t_lags.pkl', 'wb')
pickle.dump(max_t_lags, f)
f.close()
