import os
import pickle
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

import Load
# import Load_light as Load
import Models
import Plot
import Processing


startTime = datetime.now()
# Define Parameters
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
total_subjects = len(sesiones)*2
tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
times = np.flip(-times)
situacion = 'Escucha'

# Model parameters ('Ridge' or 'mtrf')
model = 'Ridge'

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
Stims = ['Spectrogram']
Bands = ['Theta']

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
Mean_Correlations_fname = 'saves/25_folds/{}/{}/Final_Correlation/tmin{}_tmax{}/Mean_Correlations.pkl'.format(model, situacion, tmin, tmax)
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
        print('\nModel: ' + model)
        print('Band: ' + Band)
        print('Stimulus: ' + stim)
        print('Status: ' + situacion)
        print('tmin: {} - tmax: {}'.format(tmin, tmax))
        # Paths
        save_path = 'saves/25_folds/{}/{}/Final_Correlation/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
        procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
        Run_graficos_path = 'gráficos/25_folds/{}/{}/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            model, situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_original = 'saves/25_folds/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            model, situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
        Path_it = 'saves/25_folds/{}/{}/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/'.format(
            model, situacion, Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)

        # Start Run
        sujeto_total = 0
        for sesion in sesiones:
            print('\nSession {}'.format(sesion))

            # LOAD DATA BY SUBJECT
            Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                                procesed_data_path=procesed_data_path, situacion=situacion,
                                                SilenceThreshold=0.03)
            # LOAD EEG BY SUBJECT
            eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

            # LOAD STIMULUS BY SUBJECT
            dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1, Sujeto_2=Sujeto_2)
            Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

            for sujeto, eeg, dstims in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2),
                                           (dstims_para_sujeto_1, dstims_para_sujeto_2)):
                # for sujeto, eeg, dstims in zip([2], [eeg_sujeto_2], [dstims_para_sujeto_2]):
                print('Subject {}'.format(sujeto))
                # Separo los datos en 5 y tomo test set de 20% de datos con kfold (5 iteraciones)
                Predicciones = {}
                n_folds = 25
                iteraciones = 3000

                # Defino variables donde voy a guardar mil cosas
                Pesos_ronda_canales = np.zeros((n_folds, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                Corr_buenas_ronda_canal = np.zeros((n_folds, info['nchan']))
                Rmse_buenos_ronda_canal = np.zeros((n_folds, info['nchan']))

                if Statistical_test:
                    Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(Len_Estimulos)), dtype=np.float16)
                    Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                    Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))

                # Variable to store all channel's p-value
                topo_pvalues_corr = np.zeros((n_folds, info['nchan']))
                topo_pvalues_rmse = np.zeros((n_folds, info['nchan']))

                # Variable to store p-value of significant channels
                Prob_Corr_ronda_canales = np.ones((n_folds, info['nchan']))
                Prob_Rmse_ronda_canales = np.ones((n_folds, info['nchan']))

                # Variable to store significant channels
                Canales_repetidos_corr_sujeto = np.zeros(info['nchan'])
                Canales_repetidos_rmse_sujeto = np.zeros(info['nchan'])

                # Empiezo el KFold de test
                kf_test = KFold(n_folds, shuffle=False)
                for fold, (train_val_index, test_index) in enumerate(kf_test.split(eeg)):
                    eeg_train_val, eeg_test = eeg[test_index], eeg[train_val_index]

                    dstims_train_val = list()
                    dstims_test = list()

                    for stimulus in list(dstims):
                        dstims_train_val.append(stimulus[test_index])
                        dstims_test.append(stimulus[train_val_index])

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
                    if model == 'Ridge':
                        Model = Models.Ridge(alpha)
                        Model.fit(dstims_train_val, eeg_train_val)
                        Pesos_ronda_canales[fold] = Model.coefs

                        # Predigo en test set y guardo
                        predicted = Model.predict(dstims_test)
                        Predicciones[fold] = predicted

                    elif model == 'mtrf':
                        # get the time lag of the present index to take from the delayed matrix stimuli
                        present_stim_index = np.where(delays==0)[0][0]

                        Model = Models.mne_mtrf(-tmax, -tmin, sr, alpha, present_stim_index)
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
                            Statistical_test = False

                        # TEST ESTADISTICO
                        Rcorr_fake = Correlaciones_fake[fold]
                        Rmse_fake = Errores_fake[fold]

                        p_corr = ((Rcorr_fake > Rcorr).sum(0) + 1) / (iteraciones + 1)
                        p_rmse = ((Rmse_fake < Rmse).sum(0) + 1) / (iteraciones + 1)

                        # Umbral
                        umbral = 0.05/128
                        Prob_Corr_ronda_canales[fold][p_corr < umbral] = p_corr[p_corr < umbral]
                        Prob_Rmse_ronda_canales[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]
                        # p-value topographic distribution
                        topo_pvalues_corr[fold] = p_corr
                        topo_pvalues_rmse[fold] = p_rmse

                # Save Model Weights and Correlations
                os.makedirs(Path_original, exist_ok=True)
                f = open(Path_original + 'Corr_Rmse_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
                pickle.dump([Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal], f)
                f.close()

                f = open(Path_original + 'Pesos_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
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

                # Adapt to yield p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(0)

                # Grafico cabezas y canales
                Plot.plot_cabezas_canales(info.ch_names, info, sesion, sujeto, Corr_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Correlación', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_corr)
                Plot.plot_cabezas_canales(info.ch_names, info, sesion, sujeto, Rmse_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Rmse', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_rmse)

                # Grafico Pesos
                Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Pesos_promedio,
                                        info, times, Corr_promedio, Rmse_promedio, Save_Ind_Figures,
                                        Run_graficos_path, Len_Estimulos, stim)

                # Guardo las correlaciones y los pesos promediados entre folds de cada canal del sujeto y lo adjunto a lista
                # para promediar entre canales de sujetos
                if not sujeto_total:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = Pesos_promedio
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = Corr_promedio
                    Rmse_totales_sujetos = Rmse_promedio
                    # Save p-values for all subjects
                    pvalues_corr_subjects = topo_pval_corr_sujeto
                    pvalues_rmse_subjects = topo_pval_rmse_sujeto
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = Canales_repetidos_corr_sujeto
                    Canales_repetidos_rmse_sujetos = Canales_repetidos_rmse_sujeto
                else:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, Pesos_promedio))
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
                    Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))
                    # Save p-values for all subjects
                    pvalues_corr_subjects = np.vstack((pvalues_corr_subjects, topo_pval_corr_sujeto))
                    pvalues_rmse_subjects = np.vstack((pvalues_rmse_subjects, topo_pval_rmse_sujeto))
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = np.vstack((Canales_repetidos_corr_sujetos, Canales_repetidos_corr_sujeto))
                    Canales_repetidos_rmse_sujetos = np.vstack((Canales_repetidos_rmse_sujetos, Canales_repetidos_rmse_sujeto))
                sujeto_total += 1

        # Armo cabecita con correlaciones promedio entre sujetos
        Mean_Correlations_Band[stim], lat_test_results_corr = Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info,
                                                                  Display_Total_Figures, Save_Total_Figures,
                                                                  Run_graficos_path, title='Correlation')

        Mean_Correlations_Band[stim], lat_test_results_rmse = Plot.Cabezas_corr_promedio(Rmse_totales_sujetos, info,
                                                                  Display_Total_Figures, Save_Total_Figures,
                                                                  Run_graficos_path, title='Rmse')

        # Armo cabecita con canales repetidos
        if Statistical_test:
            Plot.topo_pval(pvalues_corr_subjects.mean(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Correlation')
            Plot.topo_pval(pvalues_rmse_subjects.mean(0), info, Display_Total_Figures,
                           Save_Total_Figures, Run_graficos_path, title='Rmse')

            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Correlation')
            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Rmse')

        # Grafico Pesos
        Pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                                Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim, ERP=True)

        Plot.regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                       Save_Total_Figures, Run_graficos_path, Len_Estimulos, stim, Band, ERP=True)

        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, stim, Len_Estimulos, info, times, Display_Total_Figures, Save_Total_Figures,
                                      Run_graficos_path)
        try:
            _ = Plot.Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, Band, stim, times, sr, Display_Total_Figures,
                                            Save_Total_Figures, Run_graficos_path, Len_Estimulos)
        except:
            pass
        # Cabezas de correlacion de pesos por canal
        Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_Total_Figures,
                                              Save_Total_Figures, Run_graficos_path)

        # Save final weights
        f = open(Path_original + 'Pesos_Totales_{}_{}.pkl'.format(stim, Band), 'wb')
        pickle.dump(Pesos_totales, f)
        f.close()

        # SAVE FINAL CORRELATION
        Mean_Correlations[Band] = Mean_Correlations_Band
        if Save_Final_Correlation and sujeto_total == 18:
            os.makedirs(save_path, exist_ok=True)
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump([Correlaciones_totales_sujetos, Canales_repetidos_corr_sujetos], f)
            f.close()

            f = open(Mean_Correlations_fname, 'wb')
            pickle.dump(Mean_Correlations, f)
            f.close()

print(datetime.now() - startTime)
