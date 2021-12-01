import scipy
import pathlib
import pickle
import numpy as np
import Load

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

#Defino parametros
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
stim = 'Envelope'
Band = 'Theta'

tmin, tmax = -0.6, -0.003
sr = 128
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
n_iterations = 1000

procesed_data_path = 'saves/Preprocesed_Data/tmin{}_tmax{}/'.format(tmin, tmax)
Sujeto_1, Sujeto_2 = Load.Load_Data(sesion=21, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                    procesed_data_path=procesed_data_path)
info = Sujeto_1['info']
# LOAD STIMULUS BY SUBJECT
dstims_para_sujeto_1, dstims_para_sujeto_2 = Load.Estimulos(stim=stim, Sujeto_1=Sujeto_1, Sujeto_2=Sujeto_2)
Len_Estimulos = [len(dstims_para_sujeto_1[i][0]) for i in range(len(dstims_para_sujeto_1))]

# Paths para cargar datos
permutations_folders_path_str = 'saves/Ridge/Fake_it/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}'.format(
    Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
permutations_folders_path = pathlib.Path(permutations_folders_path_str)
permutations_files = list(permutations_folders_path.glob('*'))

original_folders_path_str = 'saves/Ridge/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}'.format(
    Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
original_folders_path = pathlib.Path(original_folders_path_str)
original_files = list(original_folders_path.glob('*'))


original_weights = np.zeros((len(permutations_files), info['chan'], sum(Len_Estimulos)))
permutations_weights = np.zeros((len(permutations_files), info['chan'], sum(Len_Estimulos)))

# Armo las matrices enteras para testear
for subject, file in enumerate(original_files):
    f = open(file, 'rb')
    original_weights_subject = pickle.load(f)
    original_weights[subject] = original_weights_subject
    f.close()

for subject, file in enumerate(permutations_files):
    f = open(file, 'rb')
    permutations_weights_subject = pickle.load(f)
    permutations_weights[subject] = permutations_weights_subject
    f.close()


# Armo las matrices de t valores (Testeo)
tt_original = np.zeros((info['chan'], sum(Len_Estimulos)))
tt_permutations = np.zeros((n_iterations, info['chan'], sum(Len_Estimulos)))

for channel in range(128):
    for delay in range(sum(Len_Estimulos)):
        tt_original[channel, delay] = scipy.stats.ttest_1samp(original_weights[:, channel, delay], 0.0)

for permutation in n_iterations:
    for channel in range(128):
        for delay in range(sum(Len_Estimulos)):
            tt_permutations[permutation, channel, delay] = scipy.stats.ttest_1samp(permutations_weights[:, channel, delay], 0.0)
