## Violin / Topo (Bands)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import mne

tmin, tmax = -0.6, -0.003
model = 'Ridge'
situacion = 'Escucha'

info_path = 'Saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
f = open(info_path, 'rb')
info = pickle.load(f)
f.close()

Run_graficos_path = 'Plots/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Topo/'.format(model, situacion, tmin, tmax)
Save_fig = True
Correlaciones = {}

stim = 'Spectrogram'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

for Band in Bands:
    f = open('Saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[Band] = Corr.mean(0)

my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta_1': 'darkgrey'}

fontsize = 15
plt.rcParams.update({'font.size': fontsize})
# fig, axs = plt.subplots(figsize=(10, 4), ncols=5, nrows=2, gridspec_kw={'wspace': 0.25})
fig, axs = plt.subplots(figsize=(14, 5), ncols=5, nrows=2)

for i, Band in enumerate(Bands):
    ax = axs[0, i]
    fig.tight_layout()
    im = mne.viz.plot_topomap(Correlaciones[Band].ravel(), info, axes=ax, show=False, sphere=0.07, cmap='Reds',
                              vmin=Correlaciones[Band].min(), vmax=Correlaciones[Band].max())
    # cbar = plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.5)
    # cbar.ax.tick_params(labelsize=fontsize)

for ax_row in axs[1:]:
    for ax in ax_row:
        ax.remove()

ax = fig.add_subplot(2, 1, (2, 3))

plt.suptitle(situacion)
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=ax)
ax.set_ylabel('Correlation')
ax.set_ylim([-0.1, 0.5])
ax.grid()
ax.set_xticklabels(['Broad band\n(0.1 - 40 Hz)', 'Delta\n(1 - 4 Hz)', 'Theta\n(4 - 8 Hz)', 'Alpha\n(8 - 13 Hz)',
                    'Low Beta\n(13 - 19 Hz)'])
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

## Violin / mTRF (Bands)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import mne
import numpy as np

info_path = 'Saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
f = open(info_path, 'rb')
info = pickle.load(f)
f.close()

tmin, tmax = -0.6, -0.003
delays = - np.arange(np.floor(tmin * info['sfreq']), np.ceil(tmax * info['sfreq']), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / info['sfreq'], np.abs(delays[-1]) * np.sign(tmax) * 1 / info['sfreq'], len(delays))
times = np.flip(-times)

model = 'Ridge'
situacion = 'Escucha'

Run_graficos_path = 'Plots/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin mTRF/'.format(model, situacion, tmin, tmax)
Save_fig = True
Correlaciones = {}
mTRFs = {}

stim = 'Spectrogram'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

for Band in Bands:
    f = open('Saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()
    Correlaciones[Band] = Corr.mean(0)

f = open('Saves/{}/{}/Original/Stims_Normalize_EEG_Standarize/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.format(model, situacion, tmin, tmax, stim, 'Theta', stim, 'Theta'), 'rb')
mTRFs_Theta = pickle.load(f)
f.close()

my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta_1': 'darkgrey'}

fontsize = 12
plt.rcParams.update({'font.size': fontsize})
fig, axs = plt.subplots(figsize=(6, 4), nrows=2, gridspec_kw={'height_ratios': [1, 1]})

plt.suptitle(situacion)
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=axs[0])
axs[0].set_ylabel('Correlation')
axs[0].set_ylim([-0.1, 0.5])
axs[0].grid()
axs[0].set_xticklabels(['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'])


spectrogram_weights_chanels = mTRFs_Theta.reshape(info['nchan'], 16, len(times)).mean(1)

# Adapt for ERP
spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
evoked.times = times
evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
            show=False, spatial_colors=True, unit=False, units='w', axes=axs[1])
axs[1].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
if times[0] < 0:
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')

axs[1].set_ylim([-0.016, 0.015])
axs[1].grid()

fig.tight_layout()
plt.show()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

## Violin Plot Stims
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn

Save_fig = True
model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003
Run_graficos_path = 'Plots/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)

Band = 'Theta'
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

Correlaciones = {}
for stim in Stims:
    f = open('Saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[stim] = Corr.mean(0)

# Violin plot
plt.ion()
plt.figure(figsize=(19, 5))
sn.violinplot(data=pd.DataFrame(Correlaciones))
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=24)
plt.grid()
plt.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(Band))
    plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

# Box plot
# my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta_1': 'C0'}

fig, ax = plt.subplots()
ax = sn.boxplot(data=pd.DataFrame(Correlaciones), width=0.35)
ax.set_ylabel('Correlation')
# ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

# Create an array with the colors you want to use
# colors = ["C0", "grey"]
# # Set your custom color palette
# palette = sn.color_palette(colors)
sn.swarmplot(data=pd.DataFrame(Correlaciones), size=2, alpha=0.4)
plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
plt.grid()
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(Band))
    plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

## Violin Plot Situation
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sn
from scipy.stats import wilcoxon
import mne

model = 'Ridge'
Band = 'Theta'
stim = 'Spectrogram'
situaciones = ['Escucha', 'Habla_Propia', 'Ambos', 'Ambos_Habla', 'Silencio']
tmin, tmax = -0.6, -0.003
Run_graficos_path = 'Plots/SIS_statistics/{}/{}/tmin{}_tmax{}/log/'.format(Band, stim, tmin, tmax)
Save_fig = True
Display_fig = False
if Display_fig:
    plt.ion()
else:
    plt.ioff()

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

Correlaciones = {}

for situacion in situaciones:
    f = open('Saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[situacion] = Corr.transpose()

# Calculate wilcoxon test over rows of the dictionary
stats = {}
pvals = {}

for sit1, sit2 in zip(('Escucha', 'Escucha', 'Escucha', 'Habla_Propia', 'Habla_Propia', 'Ambos', 'Ambos', 'Ambos_Habla'),
                      ('Habla_Propia', 'Ambos', 'Silencio', 'Silencio', 'Ambos_Habla', 'Ambos_Habla', 'Silencio', 'Silencio')):

    dist1 = Correlaciones[sit1]
    dist2 = Correlaciones[sit2]

    stat = []
    pval = []
    for i in range(len(dist1)):
        res = wilcoxon(dist1[i], dist2[i])
        stat.append(res[0])
        pval.append(res[1])

    stats[f'{sit1}-{sit2}'] = stat
    pvals[f'{sit1}-{sit2}'] = pval

    log_pval = np.log10(pval)

    # Plot pvalue
    fig, ax = plt.subplots()
    fig.suptitle(f'p-value: {sit1}-{sit2}\n'
                 f'mean: {round(np.mean(pvals[f"{sit1}-{sit2}"]), 6)} - '
                 f'min: {round(np.min(pvals[f"{sit1}-{sit2}"]), 6)} - '
                 f'max: {round(np.max(pvals[f"{sit1}-{sit2}"]), 6)}\n'
                 f'passed: {sum(np.array(pval)< 0.05/128)}', fontsize=17)
    im = mne.viz.plot_topomap(log_pval, vmin=-6, vmax=np.log10(0.0004), pos=info, axes=ax, show=False, sphere=0.07, cmap='Reds_r')
    cbar = plt.colorbar(im[0], ax=ax, shrink=0.85, ticks=[-6, -5, -4])
    cbar.ax.yaxis.set_tick_params(labelsize=17)
    cbar.ax.set_yticklabels(['<10-6', '10-5', '>10-4'])
    cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)


    fig.tight_layout()

    if Save_fig:
        os.makedirs(Run_graficos_path, exist_ok=True)
        plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
        plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

    # Plot statistic
    fig, ax = plt.subplots()
    fig.suptitle(f'stat: {sit1}-{sit2}\n'
                 f'Mean: {round(np.mean(stats[f"{sit1}-{sit2}"]), 3)}', fontsize=19)
    im = mne.viz.plot_topomap(stats[f'{sit1}-{sit2}'], info, axes=ax, show=False, sphere=0.07, cmap='Reds')
    cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(labelsize=17)
    cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)
    fig.tight_layout()

    if Save_fig:
        os.makedirs(Run_graficos_path, exist_ok=True)
        plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.png'.format(Band))
        plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.svg'.format(Band))



for situacion in situaciones:
    Correlaciones[situacion] = Correlaciones[situacion].ravel()

my_pal = {'Escucha': 'darkgrey', 'Habla_Propia': 'darkgrey', 'Ambos': 'darkgrey', 'Ambos_Habla': 'darkgrey', 'Silencio': 'darkgrey'}

plt.figure(figsize=(19, 5))
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal)
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=24)
plt.grid()
ax = plt.gca()
ax.set_xticklabels(['Listening', 'Speech\nproduction', 'Listening \n (speaking)', 'Speaking \n (listening)',
                    'Silence'], fontsize=24)
plt.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Violin_plot.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Violin_plot.svg'.format(Band))

## TRF amplitude

import numpy as np
import pickle
import matplotlib.pyplot as plt
import mne

model = 'Ridge'
Band = 'Theta'
stim = 'Spectrogram'
sr = 128
tmin, tmax = -0.6, -0.003
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
times = np.flip(-times)
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
Save_fig = True

Listening_25_folds_path = 'Saves/25_folds/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
Listening_path = 'Saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
Ambos_path = 'Saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Ambos', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)

Run_graficos_path = 'Plots/Amplitude_Comparison/{}/{}/tmin{}_tmax{}/'.format(Band, stim, tmin, tmax)

# Load TRFs
f = open(Listening_25_folds_path, 'rb')
TRF_25 = pickle.load(f)
f.close()
TRF_25 = np.flip(TRF_25.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

f = open(Listening_path, 'rb')
TRF_escucha = pickle.load(f)
f.close()
TRF_escucha = np.flip(TRF_escucha.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

f = open(Ambos_path, 'rb')
TRF_ambos = pickle.load(f)
f.close()
TRF_ambos = np.flip(TRF_ambos.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

mean_25 = TRF_25.mean(0)
mean_escucha = TRF_escucha.mean(0)
mean_ambos = TRF_ambos.mean(0)

plt.ion()

plt.figure(figsize=(15, 5))
plt.plot(times*1000, mean_escucha, label='L|O')
plt.plot(times*1000, mean_25, label='L|O downsampled')
plt.plot(times*1000, mean_ambos, label='L|B')
plt.xlim([(times*1000).min(), (times*1000).max()])
plt.xlabel('Time (ms)')
plt.ylabel('TRF (a.u.)')
plt.grid()
plt.legend()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Plot.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Plot.svg'.format(Band))

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)
for i, plot_data, title in zip(range(3), [TRF_escucha, TRF_25, TRF_ambos], ['L|O', 'L|O downsampled', 'L | B']):
    evoked = mne.EvokedArray(plot_data, info)
    evoked.times = times
    evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[i])
    axs[i].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
    axs[i].xaxis.label.set_size(14)
    axs[i].yaxis.label.set_size(14)
    axs[i].set_ylim([-0.016, 0.015])
    axs[i].tick_params(axis='both', labelsize=14)
    axs[i].grid()
    axs[i].legend(fontsize=12)
    axs[i].set_title(f'{title}', fontsize=15)
    if i != 2:
        axs[i].set_xlabel('', fontsize=14)
    else:
        axs[i].set_xlabel('Time (ms)', fontsize=14)

fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Subplots.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Subplots.svg'.format(Band))

## Heatmaps
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003
Run_graficos_path = 'Plots/Model_Comparison/{}/{}/tmin{}_tmax{}/Heatmaps/'.format(model, situacion, tmin, tmax)
Save_fig = True

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

Corrs_map = np.zeros((len(Stims), len(Bands)))
Sig_map = np.zeros((len(Stims), len(Bands)))

for i, stim in enumerate(Stims):
    Corr_stim = []
    Sig_stim = []
    for Band in Bands:
        f = open('Saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
        Corr_Band, Sig_Band = pickle.load(f)
        f.close()
        Corr_stim.append(Corr_Band.mean())
        Sig_stim.append(Sig_Band.mean(1).sum(0))
    Corrs_map[i] = Corr_stim
    Sig_map[i] = Sig_stim

fig, ax = plt.subplots()
plt.imshow(Corrs_map)
plt.title('Correlation', fontsize=15)
ax.set_yticks(np.arange(len(Stims)))
ax.set_yticklabels(Stims, fontsize=13)
ax.set_xticks(np.arange(len(Bands)))
ax.set_xticklabels(Bands, fontsize=13)
# ax.xaxis.tick_top()
cbar = plt.colorbar(shrink=0.7, aspect=15)
cbar.ax.tick_params(labelsize=13)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Corr.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Corr.svg'.format(Band))

fig, ax = plt.subplots()
plt.imshow(Sig_map)
plt.title('Significance', fontsize=15)
ax.set_yticks(np.arange(len(Stims)))
ax.set_yticklabels(Stims, fontsize=13)
ax.set_xticks(np.arange(len(Bands)))
ax.set_xticklabels(Bands, fontsize=13)
# ax.xaxis.tick_top()
cbar = plt.colorbar(shrink=0.7, aspect=15)
cbar.ax.tick_params(labelsize=13)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Stat.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Stat.svg'.format(Band))