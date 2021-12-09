import scipy
import pathlib
import pickle
import numpy as np
import Load
import matplotlib.pyplot as plt

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

plt.figure()
plt.imshow(tt_original)
plt.title('Original data t-values')
plt.colorbar()
plt.savefig('gráficos/Original_t-values.png')

plt.figure()
plt.imshow(tt_permutations[0])
plt.title('First permutation t-values')
plt.colorbar()
plt.savefig('gráficos/Permutation_t-values.png')
## TFCE TEST
from mne.stats import permutation_cluster_1samp_test
from scipy import stats
import mne

def plot_t_p(t, p, title, mcc, axes=None):
    if axes is None:
        fig = plt.figure(figsize=(6, 3))
        axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122)]
        show = True
    else:
        show = False
    p_lims = [0.1, 0.001]
    t_lims = -stats.distributions.t.ppf(p_lims, n_subjects - 1)
    p_lims = [-np.log10(p) for p in p_lims]
    # t plot
    x, y = np.mgrid[0:width, 0:width]
    surf = axes[0].plot_surface(x, y, np.reshape(t, (width, width)),
                                rstride=1, cstride=1, linewidth=0,
                                vmin=t_lims[0], vmax=t_lims[1], cmap='viridis')
    axes[0].set(xticks=[], yticks=[], zticks=[],
                xlim=[0, width - 1], ylim=[0, width - 1])
    axes[0].view_init(30, 15)
    cbar = plt.colorbar(ax=axes[0], shrink=0.75, orientation='horizontal',
                        fraction=0.1, pad=0.025, mappable=surf)
    cbar.set_ticks(t_lims)
    cbar.set_ticklabels(['%0.1f' % t_lim for t_lim in t_lims])
    cbar.set_label('t-value')
    cbar.ax.get_xaxis().set_label_coords(0.5, -0.3)
    if not show:
        axes[0].set(title=title)
        if mcc:
            axes[0].title.set_weight('bold')
    # p plot
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (width, width)))
    img = axes[1].imshow(use_p, cmap='inferno', vmin=p_lims[0], vmax=p_lims[1],
                         interpolation='nearest')
    axes[1].set(xticks=[], yticks=[])
    cbar = plt.colorbar(ax=axes[1], shrink=0.75, orientation='horizontal',
                        fraction=0.1, pad=0.025, mappable=img)
    cbar.set_ticks(p_lims)
    cbar.set_ticklabels(['%0.1f' % p_lim for p_lim in p_lims])
    cbar.set_label(r'$-\log_{10}(p)$')
    cbar.ax.get_xaxis().set_label_coords(0.5, -0.3)
    if show:
        text = fig.suptitle(title)
        if mcc:
            text.set_weight('bold')
        plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
        mne.viz.utils.plt_show()


width = 40
n_subjects = 10
signal_mean = 100
signal_sd = 100
noise_sd = 0.01
gaussian_sd = 5
sigma = 1e-3  # sigma for the "hat" method
n_permutations = 'all'  # run an exact test
n_src = width * width

rng = np.random.RandomState(2)
X = noise_sd * rng.randn(n_subjects, width, width)

threshold_tfce = dict(start=0, step=0.2)
t_tfce, _, p_tfce, H0 = permutation_cluster_1samp_test(
    X, n_jobs=1, threshold=threshold_tfce, adjacency=None,
    n_permutations=n_permutations, out_type='mask')

plot_t_p(t_tfce, p_tfce, 'Hola', True)