import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import mne
from scipy.spatial import ConvexHull

model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Convex_Hull/'.format(model, situacion, tmin, tmax)
Save_fig = True

Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'All']

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(model, situacion, tmin, tmax, Band), 'rb')
    Corr_Envelope, Pass_Envelope = pickle.load(f)
    f.close()

    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(model, situacion, tmin, tmax, Band), 'rb')
    Corr_Pitch, Pass_Pitch = pickle.load(f)
    f.close()

    f = open(
        'saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Spectrogram_EEG_{}.pkl'.format(model, situacion, tmin, tmax, Band), 'rb')
    Corr_Spectrogram, Pass_Spectrogram = pickle.load(f)
    f.close()

    f = open(
        'saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Envelope_Pitch_Spectrogram_EEG_{}.pkl'.format(model, situacion, tmin, tmax, Band),
        'rb')
    Corr_Envelope_Pitch_Spectrogram, Pass_Envelope_Pitch_Spectrogram = pickle.load(f)
    f.close()


    Corr_Envelope = Corr_Envelope.ravel()
    Corr_Pitch = Corr_Pitch.ravel()
    Corr_Spectrogram = Corr_Spectrogram.ravel()
    Corr_Envelope_Pitch_Spectrogram = Corr_Envelope_Pitch_Spectrogram.ravel()

    Pass_Envelope = Pass_Envelope.ravel()
    Pass_Pitch = Pass_Pitch.ravel()
    Pass_Spectrogram = Pass_Spectrogram.ravel()
    Pass_Envelope_Pitch_Spectrogram = Pass_Envelope_Pitch_Spectrogram.ravel()

    Envelope_points = np.array([Corr_Envelope_Pitch_Spectrogram, Corr_Envelope]).transpose()
    Pitch_points = np.array([Corr_Envelope_Pitch_Spectrogram, Corr_Pitch]).transpose()
    Spectrogram_points = np.array([Corr_Envelope_Pitch_Spectrogram, Corr_Spectrogram]).transpose()

    Envelope_hull = ConvexHull(Envelope_points)
    Pitch_hull = ConvexHull(Pitch_points)
    Spectrogram_hull = ConvexHull(Spectrogram_points)

    # PLOT
    plt.ion()
    plt.figure()
    plt.title(Band)

    # Plot surviving
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Envelope != 0], Corr_Envelope[Pass_Envelope != 0], '.', color='C0',
             label='Envelope', ms=2.5)
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Pitch != 0], Corr_Pitch[Pass_Pitch != 0], '.', color='C1', label='Pitch'
             , ms=2.5)
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Spectrogram != 0], Corr_Spectrogram[Pass_Spectrogram != 0], '.', color='C2',
             label='Spectrogram', ms=2.5)

    # Plot dead
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Envelope == 0], Corr_Envelope[Pass_Envelope == 0], '.', color='grey',
             alpha=0.5, label='Perm. test failed', ms=2)
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Pitch == 0], Corr_Pitch[Pass_Pitch == 0], '.', color='grey', alpha=0.5,
             label='Perm. test failed', ms=2)
    plt.plot(Corr_Envelope_Pitch_Spectrogram[Pass_Spectrogram == 0], Corr_Spectrogram[Pass_Spectrogram == 0], '.', color='grey',
             alpha=0.5, label='Perm. test failed', ms=2)

    plt.fill(Envelope_points[Envelope_hull.vertices, 0], Envelope_points[Envelope_hull.vertices, 1], color='C0',
             alpha=0.3, lw=0)
    plt.fill(Pitch_points[Pitch_hull.vertices, 0], Pitch_points[Pitch_hull.vertices, 1], color='C1', alpha=0.3, lw=0)
    plt.fill(Spectrogram_points[Spectrogram_hull.vertices, 0], Spectrogram_points[Spectrogram_hull.vertices, 1], color='C2',
             alpha=0.3, lw=0)

    # Lines
    plt.plot([plt.xlim()[0], 0.7], [plt.xlim()[0], 0.7], 'k--', zorder = 0)
    xlimit, ylimit = plt.xlim(), plt.ylim()
    plt.hlines(0, xlimit[0], xlimit[1], color='grey', linestyle='dashed')
    plt.vlines(0, ylimit[0], ylimit[1], color='grey', linestyle='dashed')
    plt.set_xlim = xlimit
    plt.set_ylim = ylimit
    plt.ylabel('Individual model (r)')
    plt.xlabel('Full model (r)')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), markerscale=3)

    # Save
    plt.tight_layout()
    if Save_fig:
        Run_graficos_path
        os.makedirs(Run_graficos_path, exist_ok=True)
        plt.savefig(Run_graficos_path + '{}.png'.format(Band))
        plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

## Violin Plot Bands
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn

tmin, tmax = -0.6, -0.003
model = 'Ridge'
situacion = 'Escucha'

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)
Save_fig = True
Correlaciones = {}

stim = 'Envelope'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[Band] = Corr.mean(0)

my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta_1': 'darkgrey'}

plt.ion()
plt.figure(figsize=(19, 5))
plt.title(situacion, fontsize=24)
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal)
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.ylim([-0, 0.5])
plt.grid()
ax = plt.gca()
ax.set_xticklabels(['All\n(0.1 - 40 Hz)', 'Delta\n(1 - 4 Hz)', 'Theta\n(4 - 8 Hz)', 'Alpha\n(8 - 13 Hz)',
                    'Low Beta\n(13 - 19 Hz)'], fontsize=24)
plt.tight_layout()

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

model = 'Ridge'
situacion = 'Escucha'
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)
Save_fig = True

tmin, tmax = -0.6, -0.003

Correlaciones = {}

Band = 'Theta'
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

for stim in Stims:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[stim] = Corr.mean(0)

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

## Violin Plot Situation
import pandas as pd
import pickle
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
Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/'.format(Band, stim, tmin, tmax)
Save_fig = True

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

Correlaciones = {}

for situacion in situaciones:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[situacion] = Corr.transpose()

# Calculate wilcoxon test over rows of the dictionary
stats = {}
pvals = {}

for sit1, sit2 in zip(('Escucha', 'Escucha', 'Escucha', 'Habla_Propia', 'Ambos', 'Ambos', 'Ambos_Habla'),
                      ('Habla_Propia', 'Ambos', 'Silencio', 'Silencio', 'Ambos_Habla', 'Silencio', 'Silencio')):

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

    # Plot pvalue
    fig, ax = plt.subplots()
    fig.suptitle(f'p-value: {sit1}-{sit2}\n'
                 f'Mean: {round(np.mean(pvals[f"{sit1}-{sit2}"]), 3)}', fontsize=19)
    im = mne.viz.plot_topomap(pvals[f'{sit1}-{sit2}'], info, axes=ax, show=False, sphere=0.07, cmap='Reds')
    cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(labelsize=17)
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

## Venn Diagrams
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
import pickle
import os
import numpy as np

model = 'Ridge'
situacion = 'Escucha'
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Venn_Diagrams/'.format(model, situacion, tmin, tmax)
Save_fig = False

tmin, tmax = -0.6, -0.003


f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Mean_Correlations.pkl'.format(model, situacion, tmin, tmax), 'rb')
Mean_Correlations = pickle.load(f)
f.close()

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']
Bands = ['Theta']

for Band in Bands:
    stims = ['Envelope', 'Pitch', 'Spectrogram']

    stims.append('{}_{}'.format(stims[0], stims[1]))
    stims.append('{}_{}'.format(stims[0], stims[2]))
    stims.append('{}_{}'.format(stims[1], stims[2]))
    stims.append('{}_{}_{}'.format(stims[0], stims[1], stims[2]))

    r2_1 = Mean_Correlations[Band][stims[0]][0]**2
    r2_2 = Mean_Correlations[Band][stims[1]][0]**2
    r2_3 = Mean_Correlations[Band][stims[2]][0]**2
    r2_12 = Mean_Correlations[Band][stims[3]][0]**2
    r2_13 = Mean_Correlations[Band][stims[4]][0]**2
    r2_23 = Mean_Correlations[Band][stims[5]][0]**2
    r2_123 = Mean_Correlations[Band][stims[6]][0]**2

    r2_int_12 = r2_1 + r2_2 - r2_12
    r2_int_13 = r2_1 + r2_3 - r2_13
    r2_int_23 = r2_2 + r2_3 - r2_23

    r2_int_123 = r2_123 + r2_1 + r2_2 + r2_3 - r2_12 - r2_13 - r2_23

    r2u_int_12 = r2_1 + r2_2 - r2_12 - r2_int_123
    r2u_int_13 = r2_1 + r2_3 - r2_13 - r2_int_123
    r2u_int_23 = r2_2 + r2_3 - r2_23 - r2_int_123

    # r2u_1 = Mean_Correlations[Band][stims[0]][0]**2 + r2_int_123 - r2_int_12 - r2_int_13
    r2u_1 = r2_123 - r2_23
    # r2u_2 = Mean_Correlations[Band][stims[1]][0]**2 + r2_int_123 - r2_int_12 - r2_int_23
    r2u_2 = r2_123 - r2_13
    # r2u_3 = Mean_Correlations[Band][stims[2]][0]**2 + r2_int_123 - r2_int_13 - r2_int_23
    r2u_3 = r2_123 - r2_12

    sets = [r2u_1, r2u_2, r2u_int_12, r2u_3, r2u_int_13, r2u_int_23, r2_int_123]
    sets_0 = []
    for set in sets:
        if set < 0:
            set = 0
        sets_0.append(set)

    plt.figure()
    plt.title('{}'.format(Band))
    venn3(subsets=np.array(sets_0).round(5), set_labels=(stims[0], stims[1], stims[2]), set_colors=('C0', 'C1', 'purple'), alpha=0.45)

    plt.tight_layout()
    if Save_fig:
        os.makedirs(Run_graficos_path, exist_ok=True)
        plt.savefig(Run_graficos_path + '{}.png'.format(Band))
        plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

    Envelope_percent = r2_1 * 100 /np.sum(sets_0)
    Pitch_percent = r2_2 * 100 /np.sum(sets_0)
    Spectrogram_percent = r2_3 * 100 /np.sum(sets_0)

    Envelope_u_percent = r2u_1 * 100 /np.sum(sets_0)
    Pitch_u_percent = r2u_2 * 100 /np.sum(sets_0)
    Spectrogram_u_percent = r2u_3 * 100 /np.sum(sets_0)

## Box Plot Bandas Decoding
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import numpy as np

tmin, tmax = -0.4, 0.2
model = 'Decoding'
situacion = 'Escucha'

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Box Plots/'.format(model, situacion, tmin, tmax)
Save_fig = False

stim = 'Envelope'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

Correlaciones = pd.DataFrame(columns=['Corr', 'Sig'])

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr_Pass = pickle.load(f)
    f.close()
    Corr = Corr_Pass[0]
    Pass = Corr_Pass[1]
    temp_df = pd.DataFrame({'Corr': Corr.ravel(), 'Sig': Pass.ravel(), 'Band': [Band]*len(Corr.ravel())})
    Correlaciones = pd.concat((Correlaciones, temp_df))

Correlaciones['Permutations test'] = np.where(Correlaciones['Sig'] == 1, 'NonSignificant', 'Significant')

my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta_1': 'C0'}

fig, ax = plt.subplots()
ax = sn.boxplot(x='Band', y='Corr', data=Correlaciones, width=0.35, palette=my_pal)
# ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

# Create an array with the colors you want to use
colors = ["C0", "grey"]
# Set your custom color palette
palette = sn.color_palette(colors)
sn.swarmplot(x='Band', y='Corr', data=Correlaciones, hue='Permutations test', size=3, palette=palette)
plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

for Band in Bands:
    print(f'\n{Band}')
    Passed_folds = (Correlaciones.loc[Correlaciones['Band'] == Band]['Permutations test'] == 'Significant').sum()
    Passed_subj = 0
    for subj in range(len(Pass)):
        Passed_subj += all(Pass[subj] < 1)
    print(f'Passed folds: {Passed_folds}/{len(Correlaciones.loc[Correlaciones["Band"] == Band])}')
    print(f'Passed subjects: {Passed_subj}/{len(Pass)}')

## Heatmaps
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Heatmaps/'.format(model, situacion, tmin, tmax)
Save_fig = True

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

Corrs_map = np.zeros((len(Stims), len(Bands)))
Sig_map = np.zeros((len(Stims), len(Bands)))

for i, stim in enumerate(Stims):
    Corr_stim = []
    Sig_stim = []
    for Band in Bands:
        f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
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


## Plot por subjects
import mne

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

model = 'Ridge'
situacion = 'Escucha'
alpha = 100
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
Save_fig = False

Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']
for Band in Bands:

    f = open('saves/Ridge/Final_Correlation/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Envelope, Pass_Envelope = pickle.load(f)
    f.close()

    f = open('saves/Ridge/Final_Correlation/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Pitch, Pass_Pitch = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/tmin{}_tmax{}/Envelope_Pitch_Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band),
        'rb')
    Corr_Envelope_Pitch_Spectrogram, Pass_Envelope_Pitch_Spectrogram = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/tmin{}_tmax{}/Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Spectrogram, Pass_Spectrogram = pickle.load(f)
    f.close()

    plt.ion()
    plt.figure()
    plt.title(Band)
    plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
    for i in range(len(Corr_Envelope)):
        plt.plot(Corr_Envelope[i], Corr_Pitch[i], '.', label='Subject {}'.format(i + 1))
    plt.legend()
    plt.grid()
