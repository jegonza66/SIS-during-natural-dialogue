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
Save = False

# Define Parameters
# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Stimuli and EEG
Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Stims = ['Envelope', 'Pitch', 'Pitch_der', 'Envelope_Pitch_Pitch_der']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']

stim = 'Envelope'
Band = 'Theta'
situacion = 'Escucha'
tmin, tmax = -0.6, 0.3
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
    Stims_preprocess, EEG_preprocess,  tmin, tmax, stim, Band)

alphas_fname = 'saves/Alphas/Alphas_Trace{:.1f}_Corr0.025.pkl'.format(2/3)
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

sesiones = np.arange(21, 26)
sujeto_total = 0
for sesion in sesiones:
    print('Sesion {}'.format(sesion))

    # LOAD DATA BY SUBJECT
    Sujeto_1, Sujeto_2 = Load.Load_Data(sesion, Band, sr, tmin, tmax, procesed_data_path)

    # LOAD EEG BY SUBJECT
    eeg_sujeto_1, eeg_sujeto_2 = Sujeto_1['EEG'], Sujeto_2['EEG']

    # LOAD STIMULUS BY SUBJECT
    dstims_para_sujeto_1, dstims_para_sujeto_2, info = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
    Cant_Estimulos = len(dstims_para_sujeto_1)

    for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (dstims_para_sujeto_1, dstims_para_sujeto_2)):
        # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):

        print('\nSujeto {}'.format(sujeto))
        N_samples.append(len(eeg))

        # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
        Predicciones = {}
        n_splits = 5
        iteraciones = 3000

        # Defino variables donde voy a guardar mil cosas
        Pesos_ronda_canales = np.zeros((n_splits, info['nchan'], len(delays) * Cant_Estimulos))
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
            alpha = Alphas[Band][stim][sesion][sujeto]
            if alpha == 'FAILED':
                alpha = np.mean([value for sesion_dict in Alphas[Band][stim].keys() for value in
                                 list(Alphas[Band][stim][sesion_dict].values()) if type(value) != 'FAILED'])

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
    f = open('saves/Decorrelation_times_{}_NoFilt_tmin{}_tmax{}.pkl'.format(stim, tmin, tmax), 'rb')
    decorrelation_times = pickle.load(f)
    f.close()
except:
    decorrelation_times = Funciones.decorrelation_time(Estimulos, sr)
    f = open('saves/Decorrelation_times_{}_NoFilt_tmin{}_tmax{}.pkl'.format(stim, tmin, tmax), 'wb')
    pickle.dump(decorrelation_times, f)
    f.close()


curva_pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display, Save,
                                              Run_graficos_path, Cant_Estimulos, Stims_Order, stim, decorrelation_times)




