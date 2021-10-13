# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:20:42 2021

@author: joaco
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn
import pandas as pd
import mne
from scipy.spatial import ConvexHull

alpha = 100
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/Alpha_{}/tmin{}_tmax{}/'.format(alpha, tmin, tmax)
Save_fig = True

Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']
for Band in Bands:

    f = open('saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
             'rb')
    Corr_Envelope, Pass_Envelope = pickle.load(f)
    f.close()

    f = open('saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
             'rb')
    Corr_Pitch, Pass_Pitch = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Envelope_Pitch_Pitch_der_EEG_{}.pkl'.format(alpha, tmin,
                                                                                                          tmax, Band),
        'rb')
    Corr_Envelope_Pitch_Pitch_der, Pass_Envelope_Pitch_Pitch_der = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Pitch_der_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
        'rb')
    Corr_Pitch_der, Pass_Pitch_der = pickle.load(f)
    f.close()

    Corr_Envelope = Corr_Envelope.ravel()
    Corr_Pitch = Corr_Pitch.ravel()
    Corr_Pitch_der = Corr_Pitch_der.ravel()
    Corr_Envelope_Pitch_Pitch_der = Corr_Envelope_Pitch_Pitch_der.ravel()

    Pass_Envelope = Pass_Envelope.ravel()
    Pass_Pitch = Pass_Pitch.ravel()
    Pass_Pitch_der = Pass_Pitch_der.ravel()
    Pass_Envelope_Pitch_Pitch_der = Pass_Envelope_Pitch_Pitch_der.ravel()

    Envelope_points = np.array([Corr_Envelope_Pitch_Pitch_der, Corr_Envelope]).transpose()
    Pitch_points = np.array([Corr_Envelope_Pitch_Pitch_der, Corr_Pitch]).transpose()
    Pitch_der_points = np.array([Corr_Envelope_Pitch_Pitch_der, Corr_Pitch_der]).transpose()

    Envelope_hull = ConvexHull(Envelope_points)
    Pitch_hull = ConvexHull(Pitch_points)
    Pitch_der_hull = ConvexHull(Pitch_der_points)

    # PLOT
    plt.ion()
    plt.figure()
    plt.title(Band)

    # Plot surviving
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Envelope != 0], Corr_Envelope[Pass_Envelope != 0], '.', color='C0',
             label='Envelope', ms=2.5)
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Pitch != 0], Corr_Pitch[Pass_Pitch != 0], '.', color='C1', label='Pitch'
             , ms=2.5)
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Pitch_der != 0], Corr_Pitch_der[Pass_Pitch_der != 0], '.', color='C2',
             label='Pitch derivate', ms=2.5)

    # Plot dead
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Envelope == 0], Corr_Envelope[Pass_Envelope == 0], '.', color='grey',
             alpha=0.5, label='Perm. test failed', ms=2)
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Pitch == 0], Corr_Pitch[Pass_Pitch == 0], '.', color='grey', alpha=0.5,
             label='Perm. test failed', ms=2)
    plt.plot(Corr_Envelope_Pitch_Pitch_der[Pass_Pitch_der == 0], Corr_Pitch_der[Pass_Pitch_der == 0], '.', color='grey',
             alpha=0.5, label='Perm. test failed', ms=2)

    plt.fill(Envelope_points[Envelope_hull.vertices, 0], Envelope_points[Envelope_hull.vertices, 1], color='C0',
             alpha=0.3, lw=0)
    plt.fill(Pitch_points[Pitch_hull.vertices, 0], Pitch_points[Pitch_hull.vertices, 1], color='C1', alpha=0.3, lw=0)
    plt.fill(Pitch_der_points[Pitch_der_hull.vertices, 0], Pitch_der_points[Pitch_der_hull.vertices, 1], color='C2',
             alpha=0.3, lw=0)

    # Lines
    plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
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
        save_path_graficos = Run_graficos_path
        try:
            os.makedirs(save_path_graficos)
        except:
            pass
        plt.savefig(save_path_graficos + '{}.png'.format(Band))
        plt.savefig(save_path_graficos + '{}.svg'.format(Band))

## Violin Plot Bandas
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn


alpha = 100
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/Alpha_{}/tmin{}_tmax{}/'.format(alpha, tmin, tmax)
Save_fig = True
Correlaciones = {}

stim = 'Envelope'

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2']
for Band in Bands:
    f = open('saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(alpha, tmin, tmax, stim, Band),
             'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[Band] = Corr.mean(0)

plt.ion()
plt.figure(figsize=(19, 5))
sn.violinplot(data=pd.DataFrame(Correlaciones))
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.grid()
ax = plt.gca()
ax.set_xticklabels(['All\n(0.1 - 40 Hz)', 'Delta\n(1 - 4 Hz)', 'Theta\n(4 - 8 Hz)',
                    'Alpha\n(8 - 13 Hz)', 'Low Beta\n(13 - 19 Hz)', 'High Beta\n(19 - 25 Hz)'], fontsize=24)
plt.tight_layout()

if Save_fig:
    save_path_graficos = Run_graficos_path
    try:
        os.makedirs(save_path_graficos)
    except:
        pass
    plt.savefig(save_path_graficos + '{}.png'.format(stim))
    plt.savefig(save_path_graficos + '{}.svg'.format(stim))

## Plot por subjects

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

alpha = 100
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/Alpha_{}/tmin{}_tmax{}/'.format(alpha, tmin, tmax)
Save_fig = False

Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']
for Band in Bands:

    f = open('saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
             'rb')
    Corr_Envelope, Pass_Envelope = pickle.load(f)
    f.close()

    f = open('saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
             'rb')
    Corr_Pitch, Pass_Pitch = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Envelope_Pitch_Pitch_der_EEG_{}.pkl'.format(alpha, tmin,
                                                                                                          tmax, Band),
        'rb')
    Corr_Envelope_Pitch_Pitch_der, Pass_Envelope_Pitch_Pitch_der = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/Alpha_{}/tmin{}_tmax{}/Pitch_der_EEG_{}.pkl'.format(alpha, tmin, tmax, Band),
        'rb')
    Corr_Pitch_der, Pass_Pitch_der = pickle.load(f)
    f.close()

    plt.ion()
    plt.figure()
    plt.title(Band)
    plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
    for i in range(len(Corr_Envelope)):
        plt.plot(Corr_Envelope[i], Corr_Pitch[i], '.', label='Subject {}'.format(i + 1))
    plt.legend()
    plt.grid()