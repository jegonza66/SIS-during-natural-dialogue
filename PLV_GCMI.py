import os
import pickle
import numpy as np
from scipy import signal as sgn

import Plot
import gcmi

import Load_light as Load

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
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'saves/'

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

        sujeto_total = 0
        for sesion in sesiones:
            print('\nSesion {}'.format(sesion))

            if Brain_Brain_sync:
                graficos_save_path = 'gráficos/Brain_Brain/tmin{}_tmax{}/{}/'.format(tmin, tmax, Band)
                print('Runing Brain to Brain synchronization...')
                Sujeto_1_Escucha, Sujeto_2_Escucha = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                    procesed_data_path=procesed_data_path, situacion='Escucha')
                # LOAD EEG BY SUBJECT
                eeg_sujeto_1_Escucha, eeg_sujeto_2_Escucha, info = Sujeto_1_Escucha['EEG'], Sujeto_2_Escucha['EEG'], \
                                                                   Sujeto_1_Escucha['info']

                Sujeto_1_Habla, Sujeto_2_Habla = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin,
                                                                tmax=tmax, procesed_data_path=procesed_data_path,
                                                                situacion='Habla')

                eeg_sujeto_1_Habla, eeg_sujeto_2_Habla = Sujeto_1_Habla['EEG'], Sujeto_2_Habla['EEG']

                Sujeto_1_Silencio, Sujeto_2_Silencio = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin,
                                                                tmax=tmax, procesed_data_path=procesed_data_path,
                                                                situacion='Silencio')

                eeg_sujeto_1_Silencio, eeg_sujeto_2_Silencio = Sujeto_1_Silencio['EEG'], Sujeto_2_Silencio['EEG']

                analytic_signal_1_Silencio = sgn.hilbert(eeg_sujeto_1_Silencio, axis=0)
                analytic_signal_2_Silencio = sgn.hilbert(eeg_sujeto_2_Silencio, axis=0)
                phase_1_Silencio = np.angle(analytic_signal_1_Silencio).transpose()
                phase_2_Silencio = np.angle(analytic_signal_2_Silencio).transpose()

                print('Running Silence Sync.')
                for channel in range(info['nchan']):
                    phase_diff = phase_1_Silencio - phase_2_Silencio[channel]
                    # Inter-Site Phase Clustering:
                    average_phase_sync_Silencio[channel] = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
                    print("\rProgress: {}%".format(int((channel + 1) * 100 / info['nchan'])), end='')
                print()
                analytic_signal_1_Escucha = sgn.hilbert(eeg_sujeto_1_Escucha, axis=0)
                analytic_signal_1_Habla = sgn.hilbert(eeg_sujeto_1_Habla, axis=0)
                analytic_signal_2_Escucha = sgn.hilbert(eeg_sujeto_2_Escucha, axis=0)
                analytic_signal_2_Habla = sgn.hilbert(eeg_sujeto_2_Habla, axis=0)

                phase_1_Escucha = np.angle(analytic_signal_1_Escucha).transpose()
                phase_1_Habla = np.angle(analytic_signal_1_Habla).transpose()
                phase_2_Escucha = np.angle(analytic_signal_2_Escucha).transpose()
                phase_2_Habla = np.angle(analytic_signal_2_Habla).transpose()

                average_phase_sync = np.zeros((info['nchan'], info['nchan']))
                average_phase_sync_Silencio = np.zeros((info['nchan'], info['nchan']))

                print('Running Listening-Speaking Sync.')
                for sujeto, phase_Escucha, phase_Habla in zip([1, 2], [phase_1_Escucha, phase_2_Escucha],
                                                              [phase_2_Habla, phase_1_Habla]):
                    print('Subject: {}'.format(sujeto))
                    for channel in range(info['nchan']):
                        phase_diff = phase_Escucha - phase_Habla[channel]
                        # Inter-Site Phase Clustering:
                        average_phase_sync[channel] = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
                        print("\rProgress: {}%".format(int((channel + 1) * 100 / info['nchan'])), end='')
                    print()
                    if sujeto == 1:
                        average_phase_sync -= average_phase_sync_Silencio
                    elif sujeto == 2:
                        average_phase_sync -= average_phase_sync_Silencio.transpose()

                    average_phase_sync[average_phase_sync < 0] = 0
                    Brain_Brain_phase_sync[sujeto_total] = average_phase_sync

                    graficos_save_path_subj = graficos_save_path + 'Subjects/'
                    Plot.Brain_sync(data=average_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
                                    graficos_save_path=graficos_save_path_subj, total_subjects=total_subjects,
                                    sesion=sesion, sujeto=sujeto)
                    sujeto_total += 1

            else:
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

                    if Intra_Brain:
                        graficos_save_path = 'gráficos/Intra_Brain_Phase_sync/{}/tmin{}_tmax{}/{}/'.format(situacion,
                                                                                                           tmin, tmax,
                                                                                                           Band)
                        print('Runing Intra Brain synchronization...')
                        analytic_signal = sgn.hilbert(eeg, axis=0)
                        phase = np.angle(analytic_signal).transpose()
                        average_phase_sync = np.zeros((len(phase), len(phase)))

                        for channel in range(len(phase)):
                            phase_diff = phase - phase[channel]
                            # Inter-Site Phase Clustering:
                            average_phase_sync[channel] = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
                            print("\rProgress: {}%".format(int((channel + 1) * 100 / info['nchan'])), end='')

                            # Phase difference (forma mili):
                            # angle: arg(complex) = arctan(imaginary/real)
                            # real_diff = np.cos(abs(phase_diff))
                            # imaginary_diff = np.sin(abs(phase_diff))
                            # vector_diff = imaginary_diff.mean(1) / real_diff.mean(1)
                            # average_phase_diff[channel] = np.arctan(vector_diff)
                        print()
                        Intra_Brain_phase_sync[sujeto_total] = average_phase_sync

                        graficos_save_path_subj = graficos_save_path + 'Subjects/'
                        Plot.Brain_sync(data=average_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
                                        graficos_save_path=graficos_save_path_subj, total_subjects=total_subjects,
                                        sesion=sesion, sujeto=sujeto)

                    if PLV:
                        graficos_save_path = 'gráficos/PLV/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax,
                                                                                              Band)
                        print('Runing Cortical entrainment...')
                        for t_lag in range(len(delays)):
                            # env phase
                            analytic_envelope_signal = sgn.hilbert(dstims[0][:, t_lag], axis=0)
                            env_phase = np.angle(analytic_envelope_signal).transpose()

                            # eeg phase
                            analytic_signal = sgn.hilbert(eeg, axis=0)
                            eeg_phase = np.angle(analytic_signal).transpose()

                            phase_diff = eeg_phase - env_phase
                            # real_diff = np.cos(abs(phase_diff))
                            # imaginary_diff = np.sin(abs(phase_diff))
                            # vector_diff = imaginary_diff.mean(1) / real_diff.mean(1)
                            # # mean phase difference
                            # average_phase_diff[sujeto_total, :, t_lag] = np.arctan(vector_diff)

                            # Inter-Site Phase Clustering:
                            total_phase_consistency[sujeto_total, :, t_lag] = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))

                            print("\rProgress: {}%".format(int((t_lag + 1) * 100 / len(delays))), end='')
                        print()
                        # Graficos save path
                        graficos_save_path_subj = graficos_save_path + 'Subjects/'

                        Plot.ch_heatmap_topo(total_data=total_phase_consistency[sujeto_total], Band=Band, info=info,
                                             delays=delays, times=times, Display=Display, Save=Save,
                                             graficos_save_path=graficos_save_path_subj,  title='Phase Sync',
                                             total_subjects=total_subjects, sesion=sesion, sujeto=sujeto)

                    if GCMI:
                        graficos_save_path = 'gráficos/GCMI/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax, Band)
                        print('Runing GCMI...')
                        gcmi_subj = np.zeros((info['nchan'], len(delays)))
                        for i in range(info['nchan']):
                            for j in range(len(delays)):
                                gcmi_subj[i, j] = gcmi.gcmi_cc(eeg.transpose()[i], dstims[0].transpose()[j])
                            print("\rProgress: {}%".format(int((i + 1) * 100 / info['nchan'])), end='')
                        print()
                        total_gcmi[sujeto_total] = gcmi_subj

                        # graficos save path
                        graficos_save_path_subj = graficos_save_path + 'Subjects/'

                        Plot.ch_heatmap_topo(total_data=total_gcmi[sujeto_total], Band=Band, info=info,
                                             delays=delays, times=times, Display=Display, Save=Save,
                                             graficos_save_path=graficos_save_path_subj, title='GCMI',
                                             total_subjects=total_subjects, sesion=sesion, sujeto=sujeto)
                    sujeto_total += 1

        if GCMI:
            # Variables save path
            save_path = Run_saves_path + '/GCMI/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
            os.makedirs(save_path, exist_ok=True)

            f = open(save_path + '{}.pkl'.format(Band), 'wb')
            pickle.dump(total_gcmi, f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_gcmi, Band=Band, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='GCMI', total_subjects=total_subjects)

        if PLV:
            # Save Cortical entrainment
            save_path = Run_saves_path + '/PLV/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
            os.makedirs(save_path, exist_ok=True)

            f = open(save_path + '{}.pkl'.format(Band), 'wb')
            pickle.dump(total_phase_consistency, f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_phase_consistency, Band=Band, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='PLV',
                                 total_subjects=total_subjects)

        if Intra_Brain:
            Plot.Brain_Brain_sync(data=Intra_Brain_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
                                  graficos_save_path=graficos_save_path, total_subjects=total_subjects)

        if Brain_Brain_sync:
            # Plot
            Plot.Brain_sync(data=Brain_Brain_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
                            graficos_save_path=graficos_save_path, total_subjects=total_subjects)

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
procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Run_saves_path = 'saves/'

# Get info
info_path = 'saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
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

        if GCMI:
            # Variables save path
            save_path = Run_saves_path + '/GCMI/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
            os.makedirs(save_path, exist_ok=True)

            f = open(save_path + '{}.pkl'.format(Band), 'rb')
            total_gcmi = pickle.load(f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_gcmi, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='GCMI', total_subjects=total_subjects)

        if PLV:
            graficos_save_path = 'gráficos/PLV/{}/tmin{}_tmax{}/{}/'.format(situacion, tmin, tmax,Band)
            # Save Cortical entrainment
            save_path = Run_saves_path + '/PLV/{}/tmin{}_tmax{}/'.format(situacion, tmin, tmax)

            f = open(save_path + '{}.pkl'.format(Band), 'rb')
            total_phase_consistency = pickle.load(f)
            f.close()

            Plot.ch_heatmap_topo(total_data=total_phase_consistency, info=info,
                                 delays=delays, times=times, Display=Display, Save=Save,
                                 graficos_save_path=graficos_save_path, title='PLV',
                                 total_subjects=total_subjects, fontsize=18)
        #
        # if Intra_Brain:
        #     Plot.Brain_Brain_sync(data=Intra_Brain_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
        #                           graficos_save_path=graficos_save_path, total_subjects=total_subjects)
        #
        # if Brain_Brain_sync:
        #     # Plot
        #     Plot.Brain_sync(data=Brain_Brain_phase_sync, Band=Band, info=info, Display=Display, Save=Save,
        #                     graficos_save_path=graficos_save_path, total_subjects=total_subjects)