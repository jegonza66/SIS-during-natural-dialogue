import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne

import Funciones
import Load
import Processing

stim = 'Envelope'
Band = 'Theta'
situacion = 'Escucha'
tmin, tmax = -0.5, 0.1
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))


# Paths
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)

# Start Run
sesiones = np.arange(21, 26)
sujeto_total = 0

EEG = pd.DataFrame()
for sesion in sesiones:
    print('Sesion {}'.format(sesion))
    # LOAD DATA BY SUBJECT
    Sujeto_1, Sujeto_2 = Load.Load_Data(sesion, Band, sr, tmin, tmax, situacion, procesed_data_path)
    # LOAD EEG BY SUBJECT
    eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']
    eeg_sujeto_1, eeg_sujeto_2 = Funciones.make_df(eeg_sujeto_1, eeg_sujeto_2)

    EEG = EEG.append(eeg_sujeto_1)
    EEG = EEG.append(eeg_sujeto_2)

N_samples = len(EEG)
EEG = Funciones.make_array(EEG)[0].reshape((N_samples, info["nchan"]))
AEP = np.zeros((info["nchan"], len(delays)))

for channel in range(info["nchan"]):
    EEG_delays = np.zeros((len(EEG), len(delays)))
    for i in range(len(EEG)):
        EEG_delays = Processing.matriz_shifteada(EEG[i], -delays)
    AEP[channel] = EEG_delays.mean(0)
    print("\rProgress: {}%".format(int((channel + 1) * 100 / info["nchan"])), end='')

##
plt.ion()
fig, ax = plt.subplots(figsize=(15,5))
evoked = mne.EvokedArray(AEP, info)
evoked.times = - times
evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                    show=False, spatial_colors=True, unit=True, units='W', axes=ax)