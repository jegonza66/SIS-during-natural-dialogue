import numpy as np
import mne
from mne.decoding import ReceptiveField
import os
import pickle
import copy

import matplotlib.pyplot as plt
import Models

def simular_iteraciones_Ridge(alpha, iteraciones, sesion, sujeto, fold, dstims_train_val, eeg_train_val,
                              dstims_test, eeg_test, Pesos_fake, Correlaciones_fake, Errores_fake):
    print("\nSesion {} - Sujeto {} - Fold {}".format(sesion, sujeto, fold + 1))
    for iteracion in np.arange(iteraciones):
        # Random permutations of stimuli
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)

        # Fit Model
        Fake_Model = Models.Ridge(alpha)
        Fake_Model.fit(dstims_train_random, eeg_train_val)  # entreno el modelo
        Pesos_fake[fold, iteracion] = Fake_Model.coefs

        # Test
        predicho_fake = Fake_Model.predict(dstims_test)

        # Correlacion
        Rcorr_fake = np.array(
            [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicho_fake[:, ii]).ravel())[0, 1] for ii in
             range(eeg_test.shape[1])])
        Correlaciones_fake[fold, iteracion] = Rcorr_fake

        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test), 2).mean(0)))
        Errores_fake[fold, iteracion] = Rmse_fake

        print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')
    return np.array(Pesos_fake, dtype=np.float16), np.array(Correlaciones_fake, dtype=np.float16), np.array(Errores_fake, dtype=np.float16)


def simular_iteraciones_Ridge_plot(info, times, situacion, alpha, iteraciones, sesion, sujeto, fold,
                                   dstims_train_val, eeg_train_val, dstims_test, eeg_test, fmin, fmax, Display=False):
    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, fold + 1))
    psds_rand_correlations = []
    for iteracion in range(iteraciones):
        # Random permutation of stimuli
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)

        # Fit Model
        Fake_Model = Models.Ridge(alpha)
        Fake_Model.fit(dstims_train_random, eeg_train_val)

        # Test
        predicho_fake = Fake_Model.predict(dstims_test)

        # Correlacion
        Rcorr_fake = np.array(
            [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicho_fake[:, ii]).ravel())[0, 1] for ii in
             range(eeg_test.shape[1])])

        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test), 2).mean(0)))

        # PSD
        psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'], fmin, fmax)
        psds_random, freqs_mean = mne.time_frequency.psd_array_welch(predicho_fake.transpose(), info['sfreq'], fmin,
                                                                     fmax)

        psds_channel_corr = np.array([np.corrcoef(psds_test[ii].ravel(), np.array(psds_random[ii]).ravel())[0, 1]
                                      for ii in range(len(psds_test))])
        psds_rand_correlations.append(np.mean(psds_channel_corr))

        # PLOTS
        if Display:
            plt.ion()
        else:
            plt.ioff()

        # Plot weights
        if not iteracion % iteraciones:
            fig, ax = plt.subplots()
            fig.suptitle('Sesion {} - Sujeto {} - Corr {:.2f}'.format(sesion, sujeto, np.mean(Rcorr_fake)))

            evoked = mne.EvokedArray(Fake_Model.coefs, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', show=False,
                        spatial_colors=True, unit=False, units='w', axes=ax)

            ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)

            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.legend(fontsize=13)
            ax.grid()

            # Plot signal and fake prediction
            fig = plt.figure()
            fig.suptitle('Fake prediction')
            plt.plot(eeg_test[:, 0], label='Signal')
            plt.plot(predicho_fake[:, 0], label='Prediction')
            plt.title('Pearson Correlation = {}'.format(Rcorr_fake[0]))
            plt.xlim([2000, 3000])
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            # Plot PSD
            fig, ax = plt.subplots()
            fig.suptitle('Sesion {} - Sujeto {} - Situacion {}'.format(sesion, sujeto, situacion))
            evoked = mne.EvokedArray(psds_random, info)
            evoked.times = freqs_mean
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s', show=False,
                        spatial_colors=True, unit=False, units='w', axes=ax)
            ax.set_xlabel('Frequency [Hz]')
            ax.grid()

        print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')
    return psds_rand_correlations


def simular_iteraciones_mtrf(iteraciones, sesion, sujeto, fold, sr, info, tmin, tmax, dstims_train, eeg_train,
                             dstims_test, eeg_test, scores, coefs, Correlaciones_fake, Errores_fake, Path_it):
    rf_fake = ReceptiveField(tmin, tmax, sr, feature_names=['envelope'],
                             estimator=1., scoring='corrcoef')

    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, fold + 1))
    for iteracion in np.arange(iteraciones):
        # Randomizo estimulos

        dstims_train_random = copy.deepcopy(dstims_train)
        np.random.shuffle(dstims_train_random)
        speech_train_random = dstims_train_random[:, 0]
        speech_train_random = speech_train_random.reshape([speech_train_random.shape[0], 1])

        dstims_test_random = copy.deepcopy(dstims_test)
        np.random.shuffle(dstims_test_random)
        speech_test_random = dstims_test_random[:, 0]
        speech_test_random = speech_test_random.reshape([speech_test_random.shape[0], 1])

        # raw_train = mne.io.RawArray(eeg_train.transpose(), info)
        # raw_test = mne.io.RawArray(eeg_test.transpose(), info)

        # Ajusto modelo Random
        rf_fake.fit(speech_train_random, eeg_train)
        # Predigo sobre estimulos random
        predicho_fake = rf_fake.predict(speech_test_random)

        # Metricas
        scores[fold] = rf_fake.score(speech_test_random, eeg_test)
        coefs[fold] = rf_fake.coef_[:, 0, :]

        # Correlacion
        Rcorr_fake = np.array(
            [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicho_fake[:, ii]).ravel())[0, 1] for ii in
             range(eeg_test.shape[1])])
        Correlaciones_fake[fold, iteracion] = Rcorr_fake

        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test), 2).mean(0)))
        Errores_fake[fold, iteracion] = Rmse_fake

        print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')

    try:
        os.makedirs(Path_it)
    except:
        pass
    f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
    pickle.dump([Correlaciones_fake, Errores_fake], f)
    f.close()
