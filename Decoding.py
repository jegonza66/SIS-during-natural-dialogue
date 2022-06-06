import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

import Load_Envelope as Load
import Models
import Plot
import Processing

from datetime import datetime
startTime = datetime.now()

# Define Parameters
tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
situacion = 'Habla_Propia'

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
Mean_Correlations_fname = 'saves/Decoding/{}/Final_Correlation/tmin{}_tmax{}/Mean_Correlations.pkl'.format(situacion, tmin, tmax)
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
        save_path = 'saves/Decoding/{}/Final_Correlation/tmin{}_tmax{}/'.format(situacion, tmin, tmax)
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/Decoding/{}/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_origial = 'saves/Decoding/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_it = 'saves/Decoding/{}/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
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
                Predicciones = {}
                n_folds = 5
                iteraciones = 100

                # Defino variables donde voy a guardar mil cosas
                Pesos_ronda_canales = np.zeros((n_folds, info['nchan'], sum(Len_Estimulos)), dtype=np.float32)
                Patterns_ronda_canales = np.zeros((n_folds, info['nchan'], sum(Len_Estimulos)), dtype=np.float32)
                Corr_buenas_ronda = np.zeros((n_folds))
                Rmse_buenos_ronda = np.zeros((n_folds))

                if Statistical_test:
                    Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)), dtype=np.float32)
                    Correlaciones_fake = np.zeros((n_folds, iteraciones))
                    Errores_fake = np.zeros((n_folds, iteraciones))

                Prob_Corr_ronda = np.ones((n_folds))
                Prob_Rmse_ronda = np.ones((n_folds))

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

                    # Ajusto el modelo y guardo
                    t_lag = -1 #Most recent audio sample
                    Model = Models.mne_mtrf_decoding(tmin, tmax, sr, info, alpha)
                    Model.fit(eeg_train_val, dstims_train_val, t_lag)
                    Pesos_ronda_canales[fold] = Model.coefs
                    Patterns_ronda_canales[fold] = Model.patterns

                    # Predigo en test set y guardo
                    predicted = Model.predict(eeg_test)
                    Predicciones[fold] = predicted

                    # Calculo Correlacion y guardo
                    Rcorr = np.array(
                        [np.corrcoef(dstims_test[:, -1].ravel(), np.array(predicted).ravel())[0, 1]])
                    Corr_buenas_ronda[fold] = Rcorr

                    # Calculo Error y guardo
                    Rmse = np.sqrt(np.power((predicted.ravel() - dstims_test[:, -1]).ravel(), 2).mean(0))
                    Rmse_buenos_ronda[fold] = Rmse

                    if Statistical_test:
                        try:
                            f = open(Path_it + 'Corr_Rmse_fake_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto),'rb')
                            Correlaciones_fake, Errores_fake = pickle.load(f)
                            f.close()
                        except:
                            Statistical_test = False

                        # TEST ESTADISTICO
                        Rcorr_fake = Correlaciones_fake[fold]
                        Rmse_fake = Errores_fake[fold]

                        p_corr = ((Rcorr_fake > Rcorr).sum() + 1) / (iteraciones + 1)
                        p_rmse = ((Rmse_fake < Rmse).sum() + 1) / (iteraciones + 1)

                        # Umbral
                        umbral = 0.01
                        if p_corr < umbral:
                            Prob_Corr_ronda[fold] = p_corr
                            Prob_Rmse_ronda[fold] = p_rmse

                # Save Model Weights and Correlations
                os.makedirs(Path_origial, exist_ok=True)
                f = open(Path_origial + 'Corr_Rmse_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Corr_buenas_ronda, Rmse_buenos_ronda], f)
                f.close()

                f = open(Path_origial + 'Pesos_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Pesos_ronda_canales.mean(0), Patterns_ronda_canales.mean(0)], f)
                f.close()

                # Tomo promedio de pesos Corr y Rmse entre los folds para todos los canales
                Pesos_promedio = Pesos_ronda_canales.mean(0)
                Patterns_promedio = Patterns_ronda_canales.mean(0)

                # Grafico cabezas y canales
                Plot.corr_sujeto_decoding(sesion, sujeto, Corr_buenas_ronda, Display_Ind_Figures, 'Correlation', Save_Ind_Figures, Run_graficos_path)
                Plot.corr_sujeto_decoding(sesion, sujeto, Rmse_buenos_ronda, Display_Ind_Figures, 'Rmse', Save_Ind_Figures, Run_graficos_path)

                # Grafico Pesos
                Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Pesos_promedio,
                                        info, times, Corr_buenas_ronda, Rmse_buenos_ronda, Save_Ind_Figures,
                                        Run_graficos_path, Len_Estimulos, stim)
                # Grafico Patterns
                Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Patterns_promedio,
                                        info, times, Corr_buenas_ronda, Rmse_buenos_ronda, Save_Ind_Figures,
                                        Run_graficos_path, Len_Estimulos, stim, title='Patterns')

                # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
                # para promediar entre canales de sujetos
                if not sujeto_total:
                    Pesos_totales_sujetos_todos_canales = Pesos_promedio
                    Patterns_totales_sujetos_todos_canales = Patterns_promedio
                    Correlaciones_totales_sujetos = Corr_buenas_ronda
                    Rmse_totales_sujetos = Rmse_buenos_ronda

                    Folds_passed_corr_sujetos = Prob_Corr_ronda
                    Folds_passed_rmse_sujetos = Prob_Rmse_ronda

                else:
                    Pesos_totales_sujetos_todos_canales = np.dstack(
                        (Pesos_totales_sujetos_todos_canales, Pesos_promedio))
                    Patterns_totales_sujetos_todos_canales = np.dstack(
                        (Patterns_totales_sujetos_todos_canales, Patterns_promedio))
                    Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_buenas_ronda))
                    Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_buenos_ronda))

                    Folds_passed_corr_sujetos = np.vstack((Folds_passed_corr_sujetos, Prob_Corr_ronda))
                    Folds_passed_rmse_sujetos = np.vstack((Folds_passed_rmse_sujetos, Prob_Rmse_ronda))

                sujeto_total += 1

        # Armo cabecita con correlaciones promedio entre sujetos
        Mean_Correlations_Band[stim] = Plot.violin_plot_decoding(Correlaciones_totales_sujetos,
                                                                  Display_Total_Figures, Save_Total_Figures,
                                                                  Run_graficos_path, title='Correlation')

        # Armo cabecita con canales repetidos
        if Statistical_test:
            Total_Folds_Passed = sum(Folds_passed_corr_sujetos.ravel() < 1)

        # Grafico Pesos
        Pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                                Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim)
        # Grafico Patterns
        Patterns_totales = Plot.regression_weights(Patterns_totales_sujetos_todos_canales, info, times,
                                                   Display_Total_Figures, Save_Total_Figures, Run_graficos_path,
                                                   Len_Estimulos, stim, title='Patterns')

        Plot.regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                       Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim, Band)
        Plot.regression_weights_matrix(Patterns_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                       Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim, Band,  title='Patterns')

        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, Display_Total_Figures, Save_Total_Figures,
                                      Run_graficos_path)
        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Patterns_totales_sujetos_todos_canales, Display_Total_Figures, Save_Total_Figures,
                                      Run_graficos_path, title='Patterns')
        try:
            _ = Plot.Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_Total_Figures,
                                            Save_Total_Figures, Run_graficos_path)
        except:
            pass
        # Cabezas de correlacion de pesos por canal
        Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_Total_Figures,
                                              Save_Total_Figures, Run_graficos_path)

        # Save final weights
        f = open(Path_origial + 'Pesos_Totales_{}_{}.pkl'.format(stim, Band), 'wb')
        pickle.dump([Pesos_totales, Patterns_totales], f)
        f.close()

        # SAVE FINAL CORRELATION
        Mean_Correlations[Band] = Mean_Correlations_Band
        if Save_Final_Correlation and sujeto_total == 18:
            os.makedirs(save_path, exist_ok=True)
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump([Correlaciones_totales_sujetos, Folds_passed_corr_sujetos], f)
            f.close()

            f = open(Mean_Correlations_fname, 'wb')
            pickle.dump(Mean_Correlations, f)
            f.close()


print(datetime.now() - startTime)
