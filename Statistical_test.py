import scipy
import pathlib
import pickle
import numpy as np

# Para cada modelo=feature/banda
# 	Para el dato original
# 		Matriz de canales (128) x tiempos (77) x sujetos (18)
# 		Para canal y tiempo,
# 			Correr t-test sujetos contra 0 (t-test simple, d.f. = 17)
# 	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
# 			tt_orig(canal, tiempo) = stats.ttest_1samp( pesos(canal, tiempo, :) , 0.0)
#
#
# 	Para cada permut (1 a 3000)
# Para canal y tiempo,
# 			Correr t-test sujetos contra 0 (t-test simple, d.f. = 17)
# 			tt_permut(canal, tiempo, permut) = stats.ttest_1samp( pesos(canal, tiempo, :) , 0.0)

tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
n_channels = 128

permutations_folders_path_str = 'saves/Ridge/Fake_it/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax-0.003'
permutations_folders_path = pathlib.Path(permutations_folders_path_str)
permutations_folders = list(permutations_folders_path.glob('*'))

original_folders_path_str = 'saves/Ridge/Corr_Rmse_Pesos_Predicciones/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax-0.003'
original_folders_path = pathlib.Path(original_folders_path_str)
original_folders = list(original_folders_path.glob('*'))

tt_original=np.zeros((n_channels, len(delays))) # Ojo la dimension de espectrograma
tt_permutations = np.zeros((n_channels, len(delays), 3000, 18))

for folder in original_folders:
    files_path = folder
    files = list(files_path.glob('Pesos*.pkl'))

    for file in files:
        f = open(file, 'rb')
        original_weights = pickle.load(f)
        f.close()

        for channel in range(128):
            for delay in range(len(delays)):
                tt_original[channel, delay] = scipy.stats.ttest_1samp(original_weights[channel, delay], 0.0)


for folder in permutations_folders:
    files_path = folder
    files = list(files_path.glob('Pesos*.pkl'))

    for file in files:
        f = open(file, 'rb')
        permutations_weights = pickle.load(f)
        f.close()

        tt_permutations[:,:] = permutations_weights

