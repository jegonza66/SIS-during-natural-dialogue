# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 03:36:32 2021

@author: joaco
"""

import numpy as np
from sklearn import linear_model
import mne
from mne.decoding import ReceptiveField
import os
import pickle
import copy

import matplotlib.pyplot as plt
import Plot


def simular_iteraciones_Ridge_plot(info, sr, situacion, best_alpha, iteraciones, sesion, sujeto, test_round, dstims_train_val, eeg_train_val, dstims_test, eeg_test):
    
    mod_fake = linear_model.Ridge(alpha = best_alpha, random_state=123)
    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, test_round + 1))
    fmin,fmax = 0,40
    psds_rand_correlations = []
    for iteracion in np.arange(iteraciones):
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)
        
        mod_fake.fit(dstims_train_random, eeg_train_val) ## entreno el modelo
        
        ###### TESTEO EN VAL SET  ######
        # Predigo
        dstims_test_random = copy.deepcopy(dstims_test)
        np.random.shuffle(dstims_test_random)
        predicho_fake = mod_fake.predict(dstims_test_random)
        
        # Correlacion
        Rcorr_fake = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicho_fake[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])

        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test),2).mean(0)))
           
        psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'], fmin, fmax)
        
        psds_random, freqs_mean = mne.time_frequency.psd_array_welch(predicho_fake.transpose(), info['sfreq'], fmin, fmax)          
        
        psds_channel_corr = np.array([np.corrcoef(psds_test[ii].ravel(), np.array(psds_random[ii]).ravel())[0,1] for ii in range(len(psds_test))])   
        
        psds_rand_correlations.append(np.mean(psds_channel_corr))
        
        print("\rProgress: {}%".format(int((iteracion+1)*100/iteraciones)), end='')
    return psds_rand_correlations
    
    
def simular_iteraciones_Ridge(best_alpha, iteraciones, sesion, sujeto, test_round, dstims_train_val, eeg_train_val, dstims_test, eeg_test, Correlaciones_fake, Errores_fake, Save_iterations, Path_it):
    
    mod_fake = linear_model.Ridge(alpha = best_alpha, random_state=123)
    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, test_round + 1))
    for iteracion in np.arange(iteraciones):
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)
        
        mod_fake.fit(dstims_train_random, eeg_train_val) ## entreno el modelo
        
        ###### TESTEO EN VAL SET  ######
        # Predigo
        dstims_test_random = copy.deepcopy(dstims_test)
        np.random.shuffle(dstims_test_random)
        predicho_fake = mod_fake.predict(dstims_test_random)
        
        # Correlacion
        Rcorr_fake = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicho_fake[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])
        Correlaciones_fake[test_round, iteracion] = Rcorr_fake
        
        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test),2).mean(0)))
        Errores_fake[test_round, iteracion] = Rmse_fake
        
        print("\rProgress: {}%".format(int((iteracion+1)*100/iteraciones)), end='')
        
    if Save_iterations:
        try: os.makedirs(Path_it)
        except: pass                      
        f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
        pickle.dump([Correlaciones_fake, Errores_fake], f)
        f.close()
        
def simular_iteraciones_mtrf(iteraciones, sesion, sujeto, test_round, sr, info, tmin, tmax, dstims_train, eeg_train, dstims_test, eeg_test, scores, coefs, Correlaciones_fake, Errores_fake, Save_iterations, Path_it):
    
    rf_fake = ReceptiveField(tmin, tmax, sr, feature_names=['envelope'],
                                    estimator=1., scoring='corrcoef')
    
    print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, test_round + 1))
    for iteracion in np.arange(iteraciones):
        # Randomizo estimulos
        
        dstims_train_random = copy.deepcopy(dstims_train)
        np.random.shuffle(dstims_train_random)
        speech_train_random = dstims_train_random[:,0]
        speech_train_random = speech_train_random.reshape([speech_train_random.shape[0],1])
        
        dstims_test_random = copy.deepcopy(dstims_test)
        np.random.shuffle(dstims_test_random)    
        speech_test_random = dstims_test_random[:,0]
        speech_test_random = speech_test_random.reshape([speech_test_random.shape[0],1])
       
        # Ajusto modelo Random
        rf_fake.fit(speech_train_random, eeg_train)
        # Predigo sobre estimulos random
        predicho_fake = rf_fake.predict(speech_test_random)
        
        # Metricas
        scores[test_round] = rf_fake.score(speech_test_random, eeg_test)
        coefs[test_round] = rf_fake.coef_[:, 0, :]
        
        # Correlacion
        Rcorr_fake = np.array([np.corrcoef(eeg_test[:,ii].ravel(), np.array(predicho_fake[:,ii]).ravel())[0,1] for ii in range(eeg_test.shape[1])])
        Correlaciones_fake[test_round, iteracion] = Rcorr_fake
        
        # Error
        Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test),2).mean(0)))
        Errores_fake[test_round, iteracion] = Rmse_fake
        
        print("\rProgress: {}%".format(int((iteracion+1)*100/iteraciones)), end='')
        
    if Save_iterations:
        try: os.makedirs(Path_it)
        except: pass                      
        f = open(Path_it + 'Corr_Rmse_fake_ronda_it_canal_Sesion{}_Sujeto{}.pkl'.format(sesion, sujeto), 'wb')
        pickle.dump([Correlaciones_fake, Errores_fake], f)
        f.close()
