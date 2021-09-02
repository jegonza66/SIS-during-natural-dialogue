<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 17055a93ec525c0153e6b8105814ecb64d38a89e
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 03:29:31 2021

@author: joaco
"""
import numpy as np
from sklearn import linear_model
import Processing
import Plot


def alphas_val(eeg_train_val, dstims_train_val, n_val_splits, train_percent, Normalizar, Estandarizar, n_best_channels, min_busqueda, max_busqueda, Plot_alphas, Prints_alphas):
    best_alphas_val = []

    all_index = np.arange(len(eeg_train_val))  
    for i in range(n_val_splits):
        
        train_index = all_index[round(i/(n_val_splits-1)*len(eeg_train_val)*(1-train_percent)):round((i/(n_val_splits-1)*(1-train_percent)+train_percent)*len(eeg_train_val))]
        val_index = np.array(list(set(all_index) - set(train_index)))
        
        eeg_train, eeg_val = eeg_train_val[train_index], eeg_train_val[val_index]
        dstims_train, dstims_val = dstims_train_val[train_index], dstims_train_val[val_index]
        
        if Normalizar: 
                axis = 0
                porcent = 5
                Processing.normalizacion(eeg_train, eeg_val, dstims_train, dstims_val, Normalizar, axis, porcent)
                          
        elif Estandarizar:
                axis = 0
                Processing.estandarizacion(eeg_train, eeg_val, dstims_train, dstims_val, Estandarizar, axis)
                
        ###### Busco en grueso hiperparametro alpha ######
        best_alpha_overall, pesos, intercepts, lista_Rmse = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = False)
        
        ###### refino busqeuda de alpha cerca del encontrado #######
        if best_alpha_overall == 10**(max_busqueda):
            min_busqueda,max_busqueda = max_busqueda-1, max_busqueda + 1
        elif best_alpha_overall == 10**(min_busqueda):
            min_busqueda,max_busqueda = min_busqueda -1, min_busqueda + 1
        else:
            min_busqueda,max_busqueda = np.log10(best_alpha_overall)-1,np.log10(best_alpha_overall)+1
        
        best_alpha_val, pesos, intercepts, lista_Rmse  = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = True) 
        best_alphas_val.append(best_alpha_val)
        print(best_alpha_val)
    return best_alphas_val    


def buscar_alpha(tStim, tResp, rStim, rResp, Plot_alphas, Prints, min_busqueda, max_busqueda, n_best_channels, fino):
    
    if not fino: 
        pasos = 13
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    else: 
        pasos = 50 
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    
    correlaciones = []
    pesos = []
    intercepts = []
    lista_Rmse = []
    corrmin = 0.1
    
    for alfa in alphas:
        mod = linear_model.Ridge(alpha=alfa, random_state=123)
        mod.fit(tStim,tResp) ## entreno el modelo
        predicho = mod.predict(rStim) ## testeo en ridge set
        Rcorr = np.array([np.corrcoef(rResp[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(rResp.shape[1])])
        Rmse = np.array(np.sqrt(np.power((predicho - rResp),2).mean(0)))
        lista_Rmse.append(np.array(Rmse))
        Rcorr[np.isnan(Rcorr)] = 0
        correlaciones.append(Rcorr)
        pesos.append(mod.coef_)
        intercepts.append(mod.intercept_)
        if Prints: print("Training: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f, over-under(%0.2f)=%d"%(alfa, np.mean(Rcorr), np.max(Rcorr), corrmin, (Rcorr>corrmin).sum()-( -Rcorr>corrmin).sum()))
    correlaciones_abs = np.array(np.abs(correlaciones))
    lista_Rmse = np.array(lista_Rmse)
    
    correlaciones_abs.sort()
    best_alpha_index = correlaciones_abs[:,-n_best_channels:].mean(1).argmax()
    best_alpha = alphas[best_alpha_index]
    
    if Prints:                     
        print("El mejor alfa es de: ", best_alpha)
        print("Con una correlación media de: ",correlaciones[best_alpha_index].mean())
        print("Y una correlación máxima de: ",correlaciones[best_alpha_index].max())
    
    if Plot_alphas: 
        if fino: Plot.plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 1, fino = True)
        else: Plot.plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 2, fino = False)
        
    return best_alpha, pesos, intercepts, lista_Rmse
<<<<<<< HEAD
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 03:29:31 2021

@author: joaco
"""
import numpy as np
from sklearn import linear_model
import Processing
import Plot


def alphas_val(eeg_train_val, dstims_train_val, n_val_splits, train_percent, Normalizar, Estandarizar, n_best_channels, min_busqueda, max_busqueda, Plot_alphas, Prints_alphas):
    best_alphas_val = []

    all_index = np.arange(len(eeg_train_val))  
    for i in range(n_val_splits):
        
        train_index = all_index[round(i/(n_val_splits-1)*len(eeg_train_val)*(1-train_percent)):round((i/(n_val_splits-1)*(1-train_percent)+train_percent)*len(eeg_train_val))]
        val_index = np.array(list(set(all_index) - set(train_index)))
        
        eeg_train, eeg_val = eeg_train_val[train_index], eeg_train_val[val_index]
        dstims_train, dstims_val = dstims_train_val[train_index], dstims_train_val[val_index]
        
        if Normalizar: 
                axis = 0
                porcent = 5
                Processing.normalizacion(eeg_train, eeg_val, dstims_train, dstims_val, Normalizar, axis, porcent)
                          
        elif Estandarizar:
                axis = 0
                Processing.estandarizacion(eeg_train, eeg_val, dstims_train, dstims_val, Estandarizar, axis)
                
        ###### Busco en grueso hiperparametro alpha ######
        best_alpha_overall, pesos, intercepts, lista_Rmse = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = False)
        
        ###### refino busqeuda de alpha cerca del encontrado #######
        if best_alpha_overall == 10**(max_busqueda):
            min_busqueda,max_busqueda = max_busqueda-1, max_busqueda + 1
        elif best_alpha_overall == 10**(min_busqueda):
            min_busqueda,max_busqueda = min_busqueda -1, min_busqueda + 1
        else:
            min_busqueda,max_busqueda = np.log10(best_alpha_overall)-1,np.log10(best_alpha_overall)+1
        
        best_alpha_val, pesos, intercepts, lista_Rmse  = buscar_alpha(dstims_train, eeg_train, dstims_val, eeg_val, Plot_alphas, Prints_alphas, min_busqueda, max_busqueda, n_best_channels, fino = True) 
        best_alphas_val.append(best_alpha_val)
        print(best_alpha_val)
    return best_alphas_val    


def buscar_alpha(tStim, tResp, rStim, rResp, Plot_alphas, Prints, min_busqueda, max_busqueda, n_best_channels, fino):
    
    if not fino: 
        pasos = 13
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    else: 
        pasos = 50 
        alphas = np.logspace(min_busqueda, max_busqueda, pasos)
    
    correlaciones = []
    pesos = []
    intercepts = []
    lista_Rmse = []
    corrmin = 0.1
    
    for alfa in alphas:
        mod = linear_model.Ridge(alpha=alfa, random_state=123)
        mod.fit(tStim,tResp) ## entreno el modelo
        predicho = mod.predict(rStim) ## testeo en ridge set
        Rcorr = np.array([np.corrcoef(rResp[:,ii].ravel(), np.array(predicho[:,ii]).ravel())[0,1] for ii in range(rResp.shape[1])])
        Rmse = np.array(np.sqrt(np.power((predicho - rResp),2).mean(0)))
        lista_Rmse.append(np.array(Rmse))
        Rcorr[np.isnan(Rcorr)] = 0
        correlaciones.append(Rcorr)
        pesos.append(mod.coef_)
        intercepts.append(mod.intercept_)
        if Prints: print("Training: alpha=%0.3f, mean corr=%0.3f, max corr=%0.3f, over-under(%0.2f)=%d"%(alfa, np.mean(Rcorr), np.max(Rcorr), corrmin, (Rcorr>corrmin).sum()-( -Rcorr>corrmin).sum()))
    correlaciones_abs = np.array(np.abs(correlaciones))
    lista_Rmse = np.array(lista_Rmse)
    
    correlaciones_abs.sort()
    best_alpha_index = correlaciones_abs[:,-n_best_channels:].mean(1).argmax()
    best_alpha = alphas[best_alpha_index]
    
    if Prints:                     
        print("El mejor alfa es de: ", best_alpha)
        print("Con una correlación media de: ",correlaciones[best_alpha_index].mean())
        print("Y una correlación máxima de: ",correlaciones[best_alpha_index].max())
    
    if Plot_alphas: 
        if fino: Plot.plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 1, fino = True)
        else: Plot.plot_alphas(alphas, correlaciones, best_alpha, lista_Rmse, linea = 2, fino = False)
        
    return best_alpha, pesos, intercepts, lista_Rmse
>>>>>>> 84f0a87c261eaf6f64ecdfe637ee7050c0c0dc44
=======
>>>>>>> 17055a93ec525c0153e6b8105814ecb64d38a89e
