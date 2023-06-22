import os
import pickle
import numpy as np
from scipy import signal as sgn
import Plot
import Load as Load

# Figures
Display = False
Save = True

# Define Parameters
# Stimuli and EEG
Stims = ['Envelope']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'All', (1, 12)]
Bands = ['Theta']
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
total_subjects = len(sesiones)*2

situacion = 'Silencio'
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

# Paths
procesed_data_path = 'Saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'Saves/'

for Band in Bands:
    for stim in Stims:
        print('\nBand: ' + Band)
        print('Stimulus: ' + stim)
        print('Status: ' + situacion)
        print('tmin: {} - tmax: {}'.format(tmin, tmax))
        # Save Variables

        try:
           graficos_save_path = 'Plots/PLV/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax, Band)
           # Save Cortical entrainment
           save_path = Run_saves_path + '/PLV/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)

           f = open(save_path + '{}.pkl'.format(Band), 'rb')
           total_phase_consistency = pickle.load(f)
           f.close()

           # Get info
           info_path = 'Saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
           f = open(info_path, 'rb')
           info = pickle.load(f)
           f.close()

           Plot.ch_heatmap_topo(total_data=total_phase_consistency, info=info,
                                delays=delays, times=times, Display=Display, Save=Save,
                                graficos_save_path=graficos_save_path, title='PLV',
                                total_subjects=total_subjects, fontsize=18)

        except:
            total_phase_consistency = np.zeros((total_subjects, 128, len(delays)))

            sujeto_total = 0
            for sesion in sesiones:
                print('\nSesion {}'.format(sesion))

                # LOAD DATA BY SUBJECT
                Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                    procesed_data_path=procesed_data_path, situacion=situacion)
                # LOAD EEG BY SUBJECT
                eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']
                # LOAD STIMULUS BY SUBJECT
                dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim, Sujeto_1, Sujeto_2)
                Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

                for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                               (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                    # for sujeto, eeg, dstims in zip([1], [eeg_sujeto_1], [dstims_para_sujeto_1]):
                    print('\nSujeto {}'.format(sujeto))
                    # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                    n_splits = 5

                    graficos_save_path = 'Plots/PLV/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax,
                                                                                          Band)
                    print('Runing Cortical entrainment...')
                    for t_lag in range(len(delays)):
                        # env phase
                        analytic_envelope_signal = sgn.hilbert(dstims[0][:, t_lag], axis=0)
                        b, a = sgn.butter(3, (4 / (128 / 2), 8 / (128 / 2)), 'band')
                        analytic_envelope_signal_filt = sgn.filtfilt(b, a, analytic_envelope_signal)
                        env_phase = np.angle(analytic_envelope_signal_filt).transpose()

                        # eeg phase
                        analytic_signal = sgn.hilbert(eeg, axis=0)
                        eeg_phase = np.angle(analytic_signal).transpose()

                        phase_diff = eeg_phase - env_phase

                        # Inter-Site Phase Clustering:
                        total_phase_consistency[sujeto_total, :, t_lag] = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))

                        print("\rProgress: {}%".format(int((t_lag + 1) * 100 / len(delays))), end='')
                    print()
                    # Graficos save path
                    graficos_save_path_subj = graficos_save_path + 'Subjects/'

                    Plot.ch_heatmap_topo(total_data=total_phase_consistency[sujeto_total], info=info,
                                         delays=delays, times=times, Display=Display, Save=Save,
                                         graficos_save_path=graficos_save_path_subj,  title='Phase Sync',
                                         total_subjects=total_subjects, sesion=sesion, sujeto=sujeto)

                    sujeto_total += 1

            # Save Cortical entrainment
            save_path = Run_saves_path + '/PLV/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
            os.makedirs(save_path, exist_ok=True)

            f = open(save_path + '{}.pkl'.format(Band), 'wb')
            pickle.dump(total_phase_consistency, f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_phase_consistency, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='PLV',
                                 total_subjects=total_subjects)


##
import numpy as np
import Plot
import pickle

# WHAT TO DO
PLV = True
GCMI = False
Intra_Brain = False
Brain_Brain_sync = False

# Figures
Display = False
Save = True

# Define Parameters
# Stimuli and EEG
Stims = ['Spectrogram']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'All']
Bands = ['Theta']
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
total_subjects = len(sesiones)*2

situacion = 'Escucha'
situcaiones = ['Escucha', 'Ambos', 'Ambos_Habla', 'Habla_Propia', 'Silencio']
tmin, tmax = -0.4, 0.2
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))

# Paths
procesed_data_path = 'Saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'Saves/'

# Get info
info_path = 'Saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
f = open(info_path, 'rb')
info = pickle.load(f)
f.close()

for situacion in situcaiones:
    for Band in Bands:
        for stim in Stims:
            print('\nBand: ' + Band)
            print('Stimulus: ' + stim)
            print('Status: ' + situacion)
            print('tmin: {} - tmax: {}'.format(tmin, tmax))
            # Save Variables
            if PLV:
                total_phase_consistency = np.zeros((total_subjects, 128, len(delays)))
            if GCMI:
                total_gcmi = np.zeros((total_subjects, 128, len(delays)))
            if Brain_Brain_sync:
                Brain_Brain_phase_sync = np.zeros((total_subjects, 128, 128))
            if Intra_Brain:
                Intra_Brain_phase_sync = np.zeros((total_subjects, 128, 128))


        if PLV:
            graficos_save_path = 'Plots/PLV/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax,Band)
            # Save Cortical entrainment
            save_path = Run_saves_path + '/PLV/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)

            f = open(save_path + '{}.pkl'.format(Band), 'rb')
            total_phase_consistency = pickle.load(f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_phase_consistency, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='PLV',
                                 total_subjects=total_subjects, fontsize=18)