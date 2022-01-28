import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.model_selection import KFold
import Load
import Models
import Processing

Display_figures_Trace = False
Save_figures_Trace = True


Stims = ['Envelope_Spectrogram', 'Envelope_Shimmer', 'Pitch_Spectrogram', 'Pitch_Shimmer', 'Spectrogram_Shimmer',
         'Envelope_Pitch_Spectrogram', 'Envelope_Pitch_Shimmer', 'Envelope_Spectrogram_Shimmer',
         'Envelope_Pitch_Spectrogram_Shimmer']
Bands = ['Beta_2', 'All']

min_trace_derivate = 0
Corr_limit = 0.01

alphas_fname = 'saves/Alphas/Alphas_Corr{}.pkl'.format(Corr_limit)
alpha_info_fname = 'saves/Alphas/Alphas_How_Corr{}.pkl'.format(Corr_limit)

try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    Alphas = {}

try:
    f = open(alpha_info_fname, 'rb')
    Info = pickle.load(f)
    f.close()
except:
    Info = {}

# DEFINO PARAMETROS
for Band in Bands:
    print('\n\n{}'.format(Band))
    try:
        Alphas_Band = Alphas[Band]
    except:
        Alphas_Band = {}
    try:
        Info_Band = Info[Band]
    except:
        Info_Band = {}

    for stim in Stims:
        print('\n\n' + stim + '\n')
        try:
            Alphas_Stim = Alphas[Band][stim]
        except:
            Alphas_Stim = {}
        try:
            Info_Stim = Info[Band][stim]
        except:
            Info_Stim = {}

        # Defino situacion de interes
        situacion = 'Escucha'
        # Defino estandarizacion
        Stims_preprocess = 'Normalize'
        EEG_preprocess = 'Standarize'
        # Defino tiempos
        sr = 128
        n_canales = 128
        tmin, tmax = -0.6, -0.003
        delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
        times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr,
                            len(delays))

        # Paths
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/Ridge_Trace_{}/Stims_{}_EEG_{}/tmin{}_tmax{}/'.format(Corr_limit, Stims_preprocess,
                                                                                            EEG_preprocess, tmin, tmax)

        min_busqueda, max_busqueda = -1, 6
        pasos = 32
        alphas_swept = np.logspace(min_busqueda, max_busqueda, pasos)
        alpha_step = np.diff(np.log(alphas_swept))[0]

        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
        # Empiezo corrida
        sujeto_total = 0
        for sesion in sesiones:
            print('Sesion {}'.format(sesion))
            try:
                Alphas_Sesion = Alphas[Band][stim][sesion]
            except:
                Alphas_Sesion = {}
            try:
                Info_Sesion = Info[Band][stim][sesion]
            except:
                Info_Sesion = {}

            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                procesed_data_path=procesed_data_path)

            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1,
                                                                        Sujeto_2=Sujeto_2)
            Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Sujeto {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                n_splits = 5

                Standarized_Betas = np.zeros(len(alphas_swept))
                Correlaciones = np.zeros(len(alphas_swept))
                Std_Corr = np.zeros(len(alphas_swept))
                Errores = np.zeros(len(alphas_swept))

                for alpha_num, alpha in enumerate(alphas_swept):
                    # print('Alpha: {}'.format(alpha))

                    # Defino variables donde voy a guardar cosas para el alpha
                    Pesos_ronda_canales = np.zeros((n_splits, info['nchan'], sum(Len_Estimulos)))
                    Corr_buenas_ronda_canal = np.zeros((n_splits, info['nchan']))
                    Rmse_buenos_ronda_canal = np.zeros((n_splits, info['nchan']))

                    # Empiezo el KFold de test
                    kf_test = KFold(n_splits, shuffle=False)
                    for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                        # Take train and validation sets
                        eeg_train_val = eeg[train_val_index]
                        dstims_train_val = list()
                        for stimulus in list(dstims):
                            dstims_train_val.append(stimulus[train_val_index])

                        train_percent = 0.8
                        eeg_train = eeg_train_val[:int(train_percent * len(eeg_train_val))]
                        eeg_val = eeg_train_val[int(train_percent * len(eeg_train_val)):]

                        dstims_train = list()
                        dstims_val = list()

                        for stimulus in list(dstims_train_val):
                            dstims_train.append(stimulus[:int(train_percent * len(eeg_train_val))])
                            dstims_val.append(stimulus[int(train_percent * len(eeg_train_val)):])

                        axis = 0
                        porcent = 5
                        eeg_train, eeg_val, dstims_train, dstims_val = \
                            Processing.standarize_normalize(eeg_train, eeg_val, dstims_train, dstims_val,
                                                            Stims_preprocess, EEG_preprocess, axis=0, porcent=5)

                        # Ajusto el modelo y guardo
                        Model = Models.Ridge(alpha)
                        Model.fit(dstims_train, eeg_train)
                        Pesos_ronda_canales[fold] = Model.coefs

                        # Predigo en val set y guardo
                        predicted = Model.predict(dstims_val)

                        # Calculo Correlacion y guardo
                        Rcorr = np.array(
                            [np.corrcoef(eeg_val[:, ii].ravel(), np.array(predicted[:, ii]).ravel())[0, 1] for ii in
                             range(eeg_val.shape[1])])
                        Corr_buenas_ronda_canal[fold] = Rcorr

                        # Calculo Error y guardo
                        Rmse = np.array(np.sqrt(np.power((predicted - eeg_val), 2).mean(0)))
                        Rmse_buenos_ronda_canal[fold] = Rmse

                    Correlaciones[alpha_num] = Corr_buenas_ronda_canal.mean()
                    Std_Corr[alpha_num] = Corr_buenas_ronda_canal.std()
                    Standarized_Betas[alpha_num] = np.sum(abs(Pesos_ronda_canales).mean(0).mean(0))
                    Trace_derivate = np.diff(Standarized_Betas) / alpha_step
                    Trace_derivate_2 = np.diff(Trace_derivate) / alpha_step

                    print("\rProgress: {}%".format(int((alpha_num + 1) * 100 / pasos)), end='')
                print("\n")
                # Individual Ridge Trace
                if (Trace_derivate_2 < min_trace_derivate).any():
                    Trace_range = np.arange((np.where(Trace_derivate_2 < min_trace_derivate)[0] + 1)[0],
                                            (np.where(Trace_derivate_2 < min_trace_derivate)[0] + 1)[-1])
                # +1 because 2nd derivate is defined in intervals
                else:
                    Trace_range = np.array([])

                Corr_range = np.where(abs(Correlaciones.max() - Correlaciones) < abs(Correlaciones.max() * Corr_limit))[0]

                Overlap = sorted(set(Trace_range).intersection(set(Corr_range)), key=list(Trace_range).index)

                alpha_index = Corr_range[-1]
                Alpha_Sujeto = alphas_swept[int(alpha_index)]
                Info_sujeto = 'MAX_CORR'
                #if Correlaciones.argmax() in Overlap or not Overlap:
                #    Alpha_Sujeto = alphas_swept[Correlaciones.argmax()]
                #    Info_sujeto = 'MAX_CORR'

                #elif Overlap:
                #    alpha_index = Overlap[-1]
                #    Alpha_Sujeto = alphas_swept[int(alpha_index)]
                #    Info_sujeto = 'MAX_OVERLAP'

                if Display_figures_Trace:
                    plt.ion()
                else:
                    plt.ioff()

                fig, ax = plt.subplots()
                fig.suptitle('{} - {}'.format(Band, stim))
                plt.xlabel('Ridge Parameter')
                plt.xscale('log')

                # axs[0].set_ylabel('Standarized Coefficents')
                # axs[0].plot(alphas_swept, Standarized_Betas, 'o--')
                # axs[0].vlines(Alpha_Sujeto, axs[0].get_ylim()[0], axs[0].get_ylim()[1], linestyle='dashed',
                #               color='red', linewidth=1.5, label='Selected value')
#                if Trace_range.size > 1:
#                    axs[0].axvspan(alphas_swept[Trace_range[0]], alphas_swept[Trace_range[-1]], alpha=0.4, color='grey',
#                                   label='Concave range')
#                if Overlap:
#                    axs[0].axvspan(alphas_swept[Overlap[0]], alphas_swept[Overlap[-1]], alpha=0.4, color='green',
#                                   label='Overlap')
#                 axs[0].grid()
#                 axs[0].legend()

                ax.set_ylabel('Mean Correlation')
                ax.plot(alphas_swept, Correlaciones, 'o--')
                ax.errorbar(alphas_swept, Correlaciones, yerr=Std_Corr, fmt='none', ecolor='black',
                                elinewidth=0.5, capsize=0.5)
                ax.vlines(alphas_swept[Correlaciones.argmax()], ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
                          color='black', linewidth=1.5, label='Max. Correlation')
                ax.vlines(Alpha_Sujeto, ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed', color='red',
                          linewidth=1.5, label='Selected value')
                if Corr_range.size > 1:
                    ax.axvspan(alphas_swept[Corr_range[0]], alphas_swept[Corr_range[-1]], alpha=0.4, color='green',
                                   label='{}% Max. Correlation'.format(Corr_limit*100))
#                if Overlap:
#                    axs[1].axvspan(alphas_swept[Overlap[0]], alphas_swept[Overlap[-1]], alpha=0.4, color='green',
#                                   label='Overlap')
                ax.grid()
                ax.legend()

                fig.tight_layout()

                if Save_figures_Trace:
                    save_path = Run_graficos_path + 'Band_{}/Stim_{}/'.format(Band, stim)
                    os.makedirs(save_path, exist_ok=True)
                    plt.savefig(save_path + 'Sesion_{}_Sujeto_{}.png'.format(sesion, sujeto))

                Alphas_Sesion[sujeto] = Alpha_Sujeto
                Info_Sesion[sujeto] = Info_sujeto
            Alphas_Stim[sesion] = Alphas_Sesion
            Info_Stim[sesion] = Info_Sesion
        Alphas_Band[stim] = Alphas_Stim
        Info_Band[stim] = Info_Stim
        Alphas[Band] = Alphas_Band
        Info[Band] = Info_Band

        # Save Alphas
        f = open(alphas_fname, 'wb')
        pickle.dump(Alphas, f)
        f.close()

        f = open(alpha_info_fname, 'wb')
        pickle.dump(Info, f)
        f.close()
