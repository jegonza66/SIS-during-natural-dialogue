import scipy
import pathlib
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

import Load


#Defino parametros
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
stim = 'Envelope'
Band = 'Theta'
save_path = 'saves/Ridge/T_value_matrices/{}'.format(Band)
Save_matrices = True

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
permutations_files = list(permutations_folders_path.glob('Pesos*'))

original_folders_path_str = 'saves/Ridge/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}'.format(
    Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band)
original_folders_path = pathlib.Path(original_folders_path_str)
original_files = list(original_folders_path.glob('Pesos*'))


original_weights = np.zeros((len(original_files), info['nchan'], sum(Len_Estimulos)))
permutations_weights = np.zeros((n_iterations, len(permutations_files), info['nchan'], sum(Len_Estimulos)))

# Armo las matrices enteras para testear
for subject, file in enumerate(original_files):
    f = open(file, 'rb')
    original_weights_subject = pickle.load(f)
    f.close()
    original_weights[subject] = original_weights_subject


for subject, file in enumerate(permutations_files):
    f = open(file, 'rb')
    permutations_weights_subject = pickle.load(f)
    f.close()
    permutations_weights[:,subject,:,:] = permutations_weights_subject

# Armo las matrices de t valores (Testeo)
tt_original = np.zeros((info['nchan'], sum(Len_Estimulos)))
tt_permutations = np.zeros((n_iterations, info['nchan'], sum(Len_Estimulos)))

for channel in range(128):
    for delay in range(sum(Len_Estimulos)):
        tt_original[channel, delay] = scipy.stats.ttest_1samp(original_weights[:, channel, delay], 0.0)[0]

for permutation in range(n_iterations):
    for channel in range(128):
        for delay in range(sum(Len_Estimulos)):
            tt_permutations[permutation, channel, delay] = scipy.stats.ttest_1samp(permutations_weights[permutation,:,channel, delay], 0.0)[0]
    print("\rProgress: {}%".format(int((permutation + 1) * 100 / n_iterations)), end='')

if Save_matrices:
    os.makedirs(save_path, exist_ok=True)
    f = open(save_path + '{}_original_tvalues.pkl'.format(stim), 'wb')
    pickle.dump(tt_original, f)
    f.close()

    f = open(save_path + '{}_permutations_tvalues.pkl'.format(stim), 'wb')
    pickle.dump(tt_permutations, f)
    f.close()

# plt.figure()
# plt.imshow(original_weights.mean(0))
# plt.title('Original data t-values')
# plt.colorbar()
# plt.savefig('gráficos/Original_t-values.png')
#
# plt.figure()
# plt.imshow(tt_permutations[0])
# plt.title('First permutation t-values')
# plt.colorbar()
# plt.savefig('gráficos/Permutation_t-values.png')

## TFCE TEST
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

n_subjects = 18
X = original_weights
n_permutations = 100
threshold_tfce = None
threshold_tfce = dict(start=1, step=0.2)
t_tfce, _, p_tfce, H0 = permutation_cluster_1samp_test(
    X, n_jobs=1, threshold=threshold_tfce, adjacency=None,
    n_permutations=n_permutations, out_type='mask')


fig, ax = plt.subplots()
ax.imshow(t_tfce, cmap='inferno')

fig, ax = plt.subplots()
ax.imshow(p_tfce.reshape(128,77), cmap='inferno')
