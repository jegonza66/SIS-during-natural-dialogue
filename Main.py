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
Stims = ['Envelope_Spectrogram']
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
        Run_graficos_path = 'gráficos/Ridge/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_origial = 'saves/Ridge/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_it = 'saves/Ridge/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

        # Start Run
        sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
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
                Pesos_ronda_canales = np.zeros((n_folds, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Corr_buenas_ronda_canal = np.zeros((n_folds, info['nchan']))
                Rmse_buenos_ronda_canal = np.zeros((n_folds, info['nchan']))

                Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))

                Prob_Corr_ronda_canales = np.ones((n_folds, info['nchan']))
                Prob_Rmse_ronda_canales = np.ones((n_folds, info['nchan']))

                Canales_repetidos_corr_sujeto = np.zeros(info['nchan'])
                Canales_repetidos_rmse_sujeto = np.zeros(info['nchan'])

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
                    alpha = Alphas[Band][stim][sesion][sujeto]
                    if alpha == 'FAILED':
                        alpha = np.mean([value for sesion_dict in Alphas[Band][stim].keys() for value in list(Alphas[Band][stim][sesion_dict].values()) if type(value) != str])

                    # Ajusto el modelo y guardo
                    Model = Models.Ridge(alpha)
                    Model.fit(dstims_train_val, eeg_train_val)
                    Pesos_ronda_canales[fold] = Model.coefs

                    # Predigo en test set y guardo
                    predicted = Model.predict(dstims_test)
                    Predicciones[fold] = predicted

                    # Calculo Correlacion y guardo
                    Rcorr = np.array(
                        [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicted[:, ii]).ravel())[0, 1] for ii in
                         range(eeg_test.shape[1])])
                    Corr_buenas_ronda_canal[fold] = Rcorr

                    # Calculo Error y guardo
                    Rmse = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                    Rmse_buenos_ronda_canal[fold] = Rmse

                    if Statistical_test:
                        try:
                            f = open(
                                Path_it + 'Corr_Rmse_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto),
                                'rb')
                            Correlaciones_fake, Errores_fake = pickle.load(f)
                            f.close()
                        except:
                            if Run_permutations:
                                Pesos_fake, Correlaciones_fake, Errores_fake = \
                                    Permutations.simular_iteraciones_Ridge(alpha, iteraciones, sesion, sujeto, fold,
                                    dstims_train_val, eeg_train_val, dstims_test, eeg_test,
                                    Pesos_fake, Correlaciones_fake, Errores_fake)
                            else:
                                Statistical_test = False

                        # TEST ESTADISTICO
                        Rcorr_fake = Correlaciones_fake[fold]
                        Rmse_fake = Errores_fake[fold]

                        p_corr = ((Rcorr_fake > Rcorr).sum(0) + 1) / (iteraciones + 1)
                        p_rmse = ((Rmse_fake < Rmse).sum(0) + 1) / (iteraciones + 1)

                        # Umbral de 5% y aplico Bonferroni (/128 Divido el umbral por el numero de intentos)
                        umbral = 0.001
                        Prob_Corr_ronda_canales[fold][p_corr < umbral] = p_corr[p_corr < umbral]
                        Prob_Rmse_ronda_canales[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]

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

                # Save Model Weights and Correlations
                try:
                    os.makedirs(Path_origial)
                except:
                    pass
                f = open(Path_origial + 'Corr_Rmse_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal], f)
                f.close()

                f = open(Path_origial + 'Pesos_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump(Pesos_ronda_canales.mean(0), f)
                f.close()

                # Tomo promedio de pesos Corr y Rmse entre los folds para todos los canales
                Pesos_promedio = Pesos_ronda_canales.mean(0)
                Corr_promedio = Corr_buenas_ronda_canal.mean(0)
                Rmse_promedio = Rmse_buenos_ronda_canal.mean(0)

                Canales_sobrevivientes_corr = []
                Canales_sobrevivientes_rmse = []
                if Statistical_test:
                    # Armo lista con canales que pasan el test
                    Canales_sobrevivientes_corr, = np.where(np.all((Prob_Corr_ronda_canales < 1), axis=0))
                    Canales_sobrevivientes_rmse, = np.where(np.all((Prob_Rmse_ronda_canales < 1), axis=0))

                    # Guardo los canales sobrevivientes de cada sujeto
                    Canales_repetidos_corr_sujeto[Canales_sobrevivientes_corr] += 1
                    Canales_repetidos_rmse_sujeto[Canales_sobrevivientes_rmse] += 1

                    # Grafico Shadows
                    Plot.plot_grafico_shadows(Display_Ind_Figures, sesion, sujeto, alpha,
                                              Canales_sobrevivientes_corr, info, sr,
                                              Corr_promedio, Save_Ind_Figures, Run_graficos_path,
                                              Corr_buenas_ronda_canal, Correlaciones_fake)

                # Grafico cabezas y canales
                Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Corr_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Correlación', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_corr)
                Plot.plot_cabezas_canales(info.ch_names, info, sr, sesion, sujeto, Rmse_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Rmse', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_rmse)

                # Grafico Pesos
                Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Pesos_promedio,
                                        info, times, Corr_promedio, Rmse_promedio, Save_Ind_Figures,
                                        Run_graficos_path, Len_Estimulos, stim, subjects_pitch, sujeto_total)

                # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
                # para promediar entre canales de sujetos
                if not sujeto_total:
                    Pesos_totales_sujetos_todos_canales = Pesos_promedio
                    Correlaciones_totales_sujetos = Corr_promedio
                    Rmse_totales_sujetos = Rmse_promedio

                    Canales_repetidos_corr_sujetos = Canales_repetidos_corr_sujeto
                    Canales_repetidos_rmse_sujetos = Canales_repetidos_rmse_sujeto
                else:
                    Pesos_totales_sujetos_todos_canales = np.dstack(
                        (Pesos_totales_sujetos_todos_canales, Pesos_promedio))
                    Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
                    Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))

                    Canales_repetidos_corr_sujetos = np.vstack(
                        (Canales_repetidos_corr_sujetos, Canales_repetidos_corr_sujeto))
                    Canales_repetidos_rmse_sujetos = np.vstack(
                        (Canales_repetidos_rmse_sujetos, Canales_repetidos_rmse_sujeto))
                sujeto_total += 1

        # Armo cabecita con correlaciones promedio entre sujetos
        Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_Total_Figures, Save_Total_Figures,
                                   Run_graficos_path, title='Correlation')
        Plot.Cabezas_corr_promedio(Rmse_totales_sujetos, info, Display_Total_Figures, Save_Total_Figures,
                                   Run_graficos_path,
                                   title='Rmse')

        # Armo cabecita con canales repetidos
        if Statistical_test:
            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures,
                                     Run_graficos_path, title='Correlation')
            Plot.Cabezas_canales_rep(Canales_repetidos_rmse_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures,
                                     Run_graficos_path, title='Rmse')

        # Grafico Pesos
        Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim, subjects_pitch, sujeto_total)
        # Plot.regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times,
        #                                                      Display_Total_Figures, Save_Total_Figures,
        #                                                      Run_graficos_path,
        #                                                      Len_Estimulos, stim)

        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, Display_Total_Figures, Save_Total_Figures,
                                      Run_graficos_path)

        # Cabezas de correlacion de pesos por canal
        Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_Total_Figures,
                                              Save_Total_Figures, Run_graficos_path)

        # SAVE FINAL CORRELATION
        if Save_Final_Correlation and sujeto_total == 17:
            save_path = 'saves/Ridge/Final_Correlation/tmin{}_tmax{}/'.format(tmin, tmax)
            try:
                os.makedirs(save_path)
            except:
                pass
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump([Correlaciones_totales_sujetos, Canales_repetidos_corr_sujetos], f)
            f.close()


print(datetime.now() - startTime)
