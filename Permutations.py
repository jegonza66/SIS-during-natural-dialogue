import numpy as np
import copy

def simular_iteraciones_Ridge(Fake_Model, iteraciones, sesion, sujeto, fold, dstims_train_val, eeg_train_val,
                              dstims_test, eeg_test, Pesos_fake, Correlaciones_fake, Errores_fake):
    print("\nSesion {} - Sujeto {} - Fold {}".format(sesion, sujeto, fold + 1))
    for iteracion in np.arange(iteraciones):
        # Random permutations of stimuli
        dstims_train_random = copy.deepcopy(dstims_train_val)
        np.random.shuffle(dstims_train_random)

        # Fit Model
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
    return Pesos_fake, Correlaciones_fake, Errores_fake
