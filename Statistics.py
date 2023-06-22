import numpy as np
import copy
from mne.stats import permutation_cluster_1samp_test

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



def tfce(Pesos_totales_sujetos_todos_canales, times, Len_Estimulos, n_permutations=1024, threshold_tfce=dict(start=0, step=0.2)):

    n_features = int(sum(Len_Estimulos)/len(times))
    n_subjects = Pesos_totales_sujetos_todos_canales.shape[2]

    if n_features > 1:
        weights_subjects = Pesos_totales_sujetos_todos_canales.copy().mean(0)
        weights_subjects = weights_subjects.reshape(n_features, len(times), n_subjects)
    else:
        weights_subjects = Pesos_totales_sujetos_todos_canales.copy()
        weights_subjects = weights_subjects.reshape(128, len(times), n_subjects)


    weights_subjects = weights_subjects.swapaxes(0, 2)
    weights_subjects = weights_subjects.swapaxes(1, 2)

    weights_subjects = np.flip(weights_subjects, axis=-1)
    weights_subjects = np.flip(weights_subjects, axis=1)

    t_tfce, clusters, p_tfce, H0 = permutation_cluster_1samp_test(
        weights_subjects,
        n_jobs=1,
        threshold=threshold_tfce,
        adjacency=None,
        n_permutations=n_permutations,
        out_type="mask",
    )

    return t_tfce, clusters, p_tfce, H0, weights_subjects, n_permutations


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2

    cohne_d = abs((np.mean(x) - np.mean(y))) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)
    return cohne_d
