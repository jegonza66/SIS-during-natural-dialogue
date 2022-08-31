import pickle
import numpy as np
from sklearn.model_selection import KFold

import Load
import Funciones
import Models
import Plot
import Processing

# Figures
Display = True
Save = True

# Define Parameters
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Stimuli and EEG
stim = 'Spectrogram'
Band = 'Theta'
situacion = 'Escucha'
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
times = np.flip(-times)

# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_graficos_path = 'gráficos/Decorrelation time/tmin{}_tmax{}/Stim_{}_EEG_Band_{}_Causal/'.format(tmin, tmax, stim, Band)

Alpha_Corr_limit = 0.01
alphas_fname = 'saves/Alphas/Alphas_Corr{}.pkl'.format(Alpha_Corr_limit)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

# Save variables
N_samples = []
# Start Run
decorrelation_times = []

sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
sujeto_total = 0

print('\nModel: Ridge')
print('Band: ' + Band)
print('Stimulus: ' + stim)
print('Status: ' + situacion)
print('tmin: {} - tmax: {}'.format(tmin, tmax))

for sesion in sesiones:
    print('\nSession {}'.format(sesion))

    # LOAD DATA BY SUBJECT
    Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                            procesed_data_path=procesed_data_path)
    # LOAD EEG BY SUBJECT
    eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

    # LOAD STIMULUS BY SUBJECT
    dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1, Sujeto_2=Sujeto_2)
    Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

    for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
        # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
        print('Subject {}'.format(sujeto))
        N_samples.append(len(eeg))

        # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
        Predicciones = {}
        n_splits = 5

        # Defino variables donde voy a guardar mil cosas
        Pesos_ronda_canales = np.zeros((n_splits, info['nchan'], sum(Len_Estimulos)))
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
            alpha = Alphas[Band][stim][sesion][sujeto]
            alpha = 1000

            # Ajusto el modelo y guardo
            Model = Models.Ridge(alpha)
            Model.fit(dstims_train_val, eeg_train_val)
            Pesos_ronda_canales[fold] = Model.coefs

        # Tomo promedio de pesos Corr y Rmse entre los folds para todos los canales
        Pesos_promedio = Pesos_ronda_canales.mean(0)

        # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
        # para promediar entre canales de sujetos
        if not sujeto_total:
            Pesos_totales_sujetos_todos_canales = Pesos_promedio
            Estimulos = dstims[0]

        else:
            Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, Pesos_promedio))
            Estimulos = np.vstack((Estimulos, dstims[0]))
        sujeto_total += 1

try:
    f = open('saves/Decorrelation_times_Envelope_Causal_tmin{}_tmax{}.pkl'.format(tmin, tmax), 'rb')
    decorrelation_times = pickle.load(f)
    f.close()
except:
    decorrelation_times = Funciones.decorrelation_time(Estimulos, sr)
    f = open('saves/Decorrelation_times_{}_Causal_tmin{}_tmax{}.pkl'.format(stim, tmin, tmax), 'wb')
    pickle.dump(decorrelation_times, f)
    f.close()


curva_pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display, Save,
                                              Run_graficos_path, Len_Estimulos, stim,
                                              decorrelation_times=decorrelation_times)

