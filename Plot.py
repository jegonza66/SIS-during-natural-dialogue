import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as sgn
import os
import seaborn as sn
import mne
import Funciones
import librosa
from statannot import add_stat_annotation
from scipy.stats import wilcoxon


def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


def lateralized_channels(info, channels_right=None, channels_left=None, path=None, Display=False, Save=True):

    if Display:
        plt.ion()
    else:
        plt.ioff()

    # Lateralization comparison
    if channels_right == None:
        channels_right = ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
    if channels_left == None:
        channels_left = ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']

    # Plot masked channels
    mask = [i in channels_right + channels_left for i in info['ch_names']]
    plt.ion()
    fig = plt.figure()
    plt.title('Masked channels for lateralization comparisson', fontsize=19)
    mne.viz.plot_topomap(np.zeros(info['nchan']), info, show=Display, sphere=0.07, mask=np.array(mask),
                         mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k', linewidth=0, markersize=12))
    fig.tight_layout()

    if Save:
        path += 'Lateralization/'
        os.makedirs(path, exist_ok=True)
        fig.savefig(path + f'left_vs_right_chs_{len(channels_right)}.png')
        fig.savefig(path + f'left_vs_right_chs_{len(channels_right)}.svg')


def plot_cabezas_canales(channel_names, info, sesion, sujeto, Valores_promedio, Display,
                         n_canales, name, Save, Run_graficos_path, Canales_sobrevivientes):
    if Display:
        plt.ion()
    else:
        plt.ioff()

    # Grafico cabezas Correlaciones
    if len(Canales_sobrevivientes):
        fig, axs = plt.subplots(1, 2)
        plt.suptitle("Sesion{} Sujeto{}\n{} = {:.3f} +/- {:.3f}".format(sesion, sujeto, name, Valores_promedio.mean(),
                                                                        Valores_promedio.std()), fontsize=19)
        im = mne.viz.plot_topomap(Valores_promedio, info, axes=axs[0], show=False, sphere=0.07, cmap='Greys',
                                  vmin=Valores_promedio.min(), vmax=Valores_promedio.max())

        mask = [i in Canales_sobrevivientes for i in range(n_canales)]
        im2 = mne.viz.plot_topomap(np.zeros(n_canales), info, axes=axs[1], show=False, sphere=0.07,
                                   mask=np.array(mask), mask_params=dict(marker='o', markerfacecolor='g',
                                                                         markeredgecolor='k', linewidth=0,
                                                                         markersize=4))

        plt.colorbar(im[0], ax=[axs[0], axs[1]], shrink=0.85, label=name, orientation='horizontal',
                     boundaries=np.linspace(Valores_promedio.min().round(decimals=3),
                                            Valores_promedio.max().round(decimals=3), 100),
                     ticks=np.linspace(Valores_promedio.min(), Valores_promedio.max(), 9).round(decimals=3))

    else:
        fig, ax = plt.subplots()
        plt.suptitle("Sesion{} Sujeto{}\n{} = {:.3f} +/- {:.3f}".format(sesion, sujeto, name, Valores_promedio.mean(),
                                                                        Valores_promedio.std()), fontsize=19)
        im = mne.viz.plot_topomap(Valores_promedio, info, axes=ax, show=False, sphere=0.07, cmap='Greys',
                                  vmin=Valores_promedio.min(), vmax=Valores_promedio.max())
        plt.colorbar(im[0], ax=ax, shrink=0.85, label=name, orientation='horizontal',
                     boundaries=np.linspace(Valores_promedio.min().round(decimals=3),
                                            Valores_promedio.max().round(decimals=3), 100),
                     ticks=np.linspace(Valores_promedio.min(), Valores_promedio.max(), 9).round(decimals=3))
        fig.tight_layout()
    if Save:
        save_path_cabezas = Run_graficos_path + 'Cabezas_canales/'
        try:
            os.makedirs(save_path_cabezas)
        except:
            pass
        fig.savefig(save_path_cabezas + '{}_Cabeza_Sesion{}_Sujeto{}.png'.format(name, sesion, sujeto))


def plot_grafico_pesos(Display, sesion, sujeto, best_alpha, Pesos_promedio,
                       info, times, Corr_promedio, Rmse_promedio, Save,
                       Run_graficos_path, Len_Estimulos, stim, title=None):
    # Defino cosas que voy a graficar
    Corr_mejor_canal = Corr_promedio.max()
    Rmse_mejor_canal = Rmse_promedio.max()

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig = plt.figure()
    fig.suptitle('Sesion {} - Sujeto {} - Corr max {:.2f} - Rmse max {:.2f} - '
                 'alpha: {:.2f}'.format(sesion, sujeto, Corr_mejor_canal, Rmse_mejor_canal, best_alpha))
    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for i in range(Cant_Estimulos):
        ax = fig.add_subplot(Cant_Estimulos, 1, i + 1)

        if Stims_Order[i] == 'Spectrogram':

            spectrogram_weights = Pesos_promedio[:, sum(Len_Estimulos[j] for j in range(i)):sum(
                Len_Estimulos[j] for j in range(i + 1))].mean(0)
            spectrogram_weights = spectrogram_weights.reshape(16, len(times))

            im = ax.pcolormesh(times * 1000, np.arange(16), spectrogram_weights, cmap='jet',
                               vmin=-spectrogram_weights.max(), vmax=spectrogram_weights.max(), shading='auto')

            Bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
            ax.set(xlabel='Time (ms)', ylabel='Hz')
            ticks_positions = np.arange(0, 16, 2)
            ticks_labels = [int(Bands_center[i + 1]) for i in np.arange(0, len(Bands_center - 1), 2)]
            ax.set_yticks(ticks_positions)
            ax.set_yticklabels(ticks_labels)

            fig.colorbar(im, ax=ax, orientation='vertical')

        else:
            evoked = mne.EvokedArray(
                Pesos_promedio[:, sum(Len_Estimulos[j] for j in range(i)):sum(Len_Estimulos[j] for j in range(i + 1))],
                info)
            evoked.times = times

            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                        show=False, spatial_colors=True, unit=False, units=dict(eeg='w', grad='fT/cm', mag='fT'),
                        axes=ax)

            ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)

            ax.xaxis.label.set_size(13)
            ax.yaxis.label.set_size(13)
            ax.legend(fontsize=13)
            ax.grid()
            ax.set_title('{}'.format(Stims_Order[i]))

    if Save:
        if title:
            save_path_graficos = Run_graficos_path + 'Individual weights {}/'.format(title)
        else:
            save_path_graficos = Run_graficos_path + 'Individual weights/'
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'Sesion{}_Sujeto{}.png'.format(sesion, sujeto))


def Plot_PSD(sesion, sujeto, Band, situacion, Display, Save, save_path, info, data, fmin=0, fmax=40):
    psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(data, info['sfreq'], fmin, fmax)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    fig.suptitle('Sesion {} - Sujeto {} - Situacion {} - Band {}'.format(sesion, sujeto, situacion, Band))

    evoked = mne.EvokedArray(psds_welch_mean, info)
    evoked.times = freqs_mean
    evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
                show=False, spatial_colors=True, unit=False, units='w', axes=ax)
    ax.set_xlabel('Frequency [Hz]')
    ax.grid()

    if Save:
        save_path_graficos = 'Plots/PSD/Zoom/{}/{}/'.format(save_path, Band)
        os.makedirs(save_path_graficos, exist_ok=True)
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.png'.format(sesion, sujeto, Band))
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.svg'.format(sesion, sujeto, Band))


def Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display, Save, Run_graficos_path, title, lat_max_chs=12):

    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fontsize = 24
    fig = plt.figure(figsize=(5, 4))
    plt.suptitle('{} = {:.3f} +/- {:.3f}'.format(title, Correlaciones_promedio.mean(), Correlaciones_promedio.std()), fontsize=fontsize)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['greenyellow', 'yellow', 'orange', 'red'])
    im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='OrRd',
                              vmin=Correlaciones_promedio.min(), vmax=Correlaciones_promedio.max(),
                              show=False, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
    cb.ax.tick_params(labelsize=fontsize)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + '{}_promedio.svg'.format(title))
        fig.savefig(save_path_graficos + '{}_promedio.png'.format(title))

    if title == 'Correlation':
        # Lateralization comparison
        # good_channels_right = ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
        # good_channels_left = ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']
        all_channels_right = ['B27', 'B28', 'B29', 'B30', 'B31', 'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                               'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
        all_channels_left = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13',
                              'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32']

        ordered_chs_right = [i for i in info['ch_names'] if i in all_channels_right]
        ordered_chs_left = [i for i in info['ch_names'] if i in all_channels_left]

        mask_right = [i in all_channels_right for i in info['ch_names']]
        mask_left = [i in all_channels_left for i in info['ch_names']]
        corr_right = np.sort(Correlaciones_promedio[mask_right])
        corr_left = np.sort(Correlaciones_promedio[mask_left])

        sorted_chs_right = [x for _, x in sorted(zip(Correlaciones_promedio[mask_right], ordered_chs_right))]
        sorted_chs_left = [x for _, x in sorted(zip(Correlaciones_promedio[mask_left], ordered_chs_left))]

        if lat_max_chs:
            corr_right = np.sort(corr_right)[-lat_max_chs:]
            corr_left = np.sort(corr_left)[-lat_max_chs:]

            sorted_chs_right = sorted_chs_right[-lat_max_chs:]
            sorted_chs_left = sorted_chs_left[-lat_max_chs:]

        fontsize = 18
        fig = plt.figure(figsize=(6.5, 4))
        data = pd.DataFrame({'Left': corr_left, 'Right': corr_right})
        ax = sn.boxplot(data=data, width=0.35)
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, .8))
        sn.swarmplot(data=data, color=".25")
        plt.tick_params(labelsize=fontsize)
        ax.set_ylabel('Correlation', fontsize=fontsize)

        add_stat_annotation(ax, data=data, box_pairs=[('Left', 'Right')],
                            test='Wilcoxon', text_format='full', loc='outside', fontsize='xx-large')
        test_results = wilcoxon(data['Left'], data['Right'])

        fig.tight_layout()

        if Save:
            save_path_graficos = Run_graficos_path + 'Lateralization/'
            os.makedirs(save_path_graficos, exist_ok=True)
            fig.savefig(save_path_graficos + f'left_vs_right_{title}_{len(sorted_chs_right)}.svg')
            fig.savefig(save_path_graficos + f'left_vs_right_{title}_{len(sorted_chs_right)}.png')

        lateralized_channels(info, channels_right=sorted_chs_right, channels_left=sorted_chs_left, path=Run_graficos_path,
                             Display=Display, Save=Save)
    else:
        test_results = None

    return (Correlaciones_promedio.mean(), Correlaciones_promedio.std()), test_results


def Cabezas_canales_rep(Canales_repetidos_sujetos, info, Display, Save, Run_graficos_path, title):
    if Display:
        plt.ion()
    else:
        plt.ioff()

    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['greenyellow', 'yellow', 'orange', 'red'])
    fig = plt.figure()
    plt.suptitle("Channels passing 5 test per subject - {}".format(title), fontsize=19)
    plt.title('Mean: {:.3f} +/- {:.3f}'.format(Canales_repetidos_sujetos.mean(), Canales_repetidos_sujetos.std()),
              fontsize=19)
    im = mne.viz.plot_topomap(Canales_repetidos_sujetos, info, cmap='OrRd',
                              vmin=0, vmax=18,
                              show=False, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
    cb.ax.tick_params(labelsize=19)
    cb.set_label(label='Number of subjects passed', size=21)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'Canales_repetidos_{}.png'.format(title))
        fig.savefig(save_path_graficos + 'Canales_repetidos_{}.svg'.format(title))


def topo_pval(topo_pval, info, Display, Save, Run_graficos_path, title):
    if Display:
        plt.ion()
    else:
        plt.ioff()

    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['greenyellow', 'yellow', 'orange', 'red'])
    fig = plt.figure()
    plt.suptitle("Mean p-values - {}".format(title), fontsize=19)
    plt.title('Mean: {:.3f} +/- {:.3f}'.format(topo_pval.mean(), topo_pval.std()),
              fontsize=19)
    im = mne.viz.plot_topomap(topo_pval, info, cmap='OrRd',
                              vmin=0, vmax=topo_pval.max(),
                              show=False, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
    cb.ax.tick_params(labelsize=19)
    cb.set_label(label='p-value', size=21)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'p-value_topo_{}.png'.format(title))
        fig.savefig(save_path_graficos + 'p-value_topo_{}.svg'.format(title))


def regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display, Save, Run_graficos_path,
                       Len_Estimulos, stim, fontsize=16, ERP=True, title=None, decorrelation_times=None):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.mean(2)

    # Ploteo pesos y cabezas
    if Display:
        plt.ion()
    else:
        plt.ioff()

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for i in range(Cant_Estimulos):
        fig, ax = plt.subplots(figsize=(9, 3))
        fig.suptitle('{}'.format(Stims_Order[i]), fontsize=fontsize)

        if Stims_Order[i] == 'Spectrogram':

            spectrogram_weights_chanels = Pesos_totales_sujetos_todos_canales_copy[:,
                                          sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))]. \
                reshape(info['nchan'], 16, len(times)).mean(1)

            if ERP:
                # Adapt for ERP
                spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)

        else:
            mean_coefs = Pesos_totales_sujetos_todos_canales_copy[:,
                                     sum(Len_Estimulos[j] for j in range(i)):sum(
                                         Len_Estimulos[j] for j in range(i + 1))]
            if ERP:
                # Adapt for ERP
                mean_coefs = np.flip(mean_coefs, axis=1)

            evoked = mne.EvokedArray(mean_coefs, info)

        evoked.times = times
        evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                    show=False, spatial_colors=True, unit=True, units='mTRF (a.u.)', axes=ax, gfp=False)
        if times[0] < 0:
            ax.axvline(x=0, ymin=0, ymax=1, color='grey')
        if decorrelation_times and times[0] < 0:
            ax.axvspan(-np.mean(decorrelation_times), 0, alpha=0.4, color='red', label=' Mean decorrelation time')
            ax.legend(fontsize=fontsize)

        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        if Stims_Order[i] == 'Spectrogram':
            ax.set_ylim([-0.016, 0.013])
        ax.tick_params(axis='both', labelsize=fontsize)
        fig.tight_layout()

        if Save:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)
            if title:
                fig.savefig(save_path_graficos + 'Regression_Weights_{}_{}.svg'.format(title, Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_{}_{}.png'.format(title, Stims_Order[i]))
            else:
                fig.savefig(save_path_graficos + 'Regression_Weights_{}.svg'.format(Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_{}.png'.format(Stims_Order[i]))

    return Pesos_totales_sujetos_todos_canales_copy


def regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times, Display,
                              Save, Run_graficos_path, Len_Estimulos, stim, Band, ERP=True, title=None):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.mean(2)

    # Ploteo pesos y cabezas
    if Display:
        plt.ion()
    else:
        plt.ioff()

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for i in range(Cant_Estimulos):

        if Stims_Order[i] == 'Spectrogram':
            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 4]})
            fig.suptitle('{} - {}'.format(Stims_Order[i], Band), fontsize=18)

            spectrogram_weights_chanels = Pesos_totales_sujetos_todos_canales_copy[:,
                                          sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))].\
                reshape(info['nchan'], 16, len(times)).mean(1)

            spectrogram_weights_bands = Pesos_totales_sujetos_todos_canales_copy[:,
                                        sum(Len_Estimulos[j] for j in range(i)):sum(
                                            Len_Estimulos[j] for j in range(i + 1))].mean(0)
            spectrogram_weights_bands = spectrogram_weights_bands.reshape(16, len(times))

            if ERP:
                # Adapt for ERP
                spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)
                spectrogram_weights_bands = np.flip(spectrogram_weights_bands, axis=1)

            axs[0].axvline(0, axs[0].get_ylim()[0], axs[0].get_ylim()[1], color='grey')
            axs[0].axhline(0, axs[0].get_xlim()[0], axs[0].get_xlim()[1], color='grey')
            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=axs[0])
            if times[0] < 0:
                axs[0].axvline(x=0, ymin=0, ymax=1, color='grey')
            axs[0].axis('off')

            im = axs[1].pcolormesh(times * 1000, np.arange(16), spectrogram_weights_bands, cmap='jet',
                               vmin=-spectrogram_weights_bands.max(), vmax=spectrogram_weights_bands.max(), shading='auto')
            axs[1].set(xlabel='Time (ms)', ylabel='Hz')

            Bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
            ticks_positions = np.arange(0, 16, 2)
            ticks_labels = [int(Bands_center[i]) for i in np.arange(0, len(Bands_center), 2)]
            axs[1].set_yticks(ticks_positions)
            axs[1].set_yticklabels(ticks_labels)
            axs[1].xaxis.label.set_size(14)
            axs[1].yaxis.label.set_size(14)
            axs[1].tick_params(axis='both', labelsize=14)

            cbar = fig.colorbar(im, ax=axs[1], orientation='vertical')
            cbar.set_label('Amplitude (a.u.)', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

            # Change axis 0 to match axis 1 width after adding colorbar
            ax1_box = axs[1].get_position().bounds
            ax0_box = axs[0].get_position().bounds
            ax0_new_box = (ax0_box[0], ax0_box[1], ax1_box[2], ax0_box[3])
            axs[0].set_position(ax0_new_box)

        else:
            mean_coefs = Pesos_totales_sujetos_todos_canales_copy[:,
                         sum(Len_Estimulos[j] for j in range(i)):sum(Len_Estimulos[j] for j in range(i + 1))]

            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 3]})
            fig.suptitle('{} - {}'.format(Stims_Order[i], Band), fontsize=18)

            if ERP:
                # Adapt for ERP
                mean_coefs = np.flip(mean_coefs, axis=1)
            evoked = mne.EvokedArray(mean_coefs, info)

            axs[0].axvline(0, axs[0].get_ylim()[0], axs[0].get_ylim()[1], color='grey')
            axs[0].axhline(0, axs[0].get_xlim()[0], axs[0].get_xlim()[1], color='grey')
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=axs[0])
            axs[0].axis('off')

            im = axs[1].pcolormesh(times * 1000, np.arange(info['nchan']), mean_coefs, cmap='jet',
                                   vmin=-(mean_coefs).max(), vmax=(mean_coefs).max(), shading='auto')
            axs[1].set(xlabel='Time (ms)', ylabel='Channel')
            axs[1].xaxis.label.set_size(14)
            axs[1].yaxis.label.set_size(14)
            axs[1].tick_params(axis='both', labelsize=14)
            cbar = fig.colorbar(im, ax=axs[1], orientation='vertical')
            cbar.set_label('Amplitude (a.u.)', fontsize=14)
            cbar.ax.tick_params(labelsize=14)

            # Change axis 0 to match axis 1 width after adding colorbar
            ax1_box = axs[1].get_position().bounds
            ax0_box = axs[0].get_position().bounds
            ax0_new_box = (ax0_box[0], ax0_box[1], ax1_box[2], ax0_box[3])
            axs[0].set_position(ax0_new_box)

        if Save:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)
            if title:
                fig.savefig(save_path_graficos + 'Regression_Weights_matrix_{}_{}.svg'.format(title, Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_marix_{}_{}.png'.format(title, Stims_Order[i]))
            else:
                fig.savefig(save_path_graficos + 'Regression_Weights_matrix_{}.svg'.format(Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_marix_{}.png'.format(Stims_Order[i]))


def Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, Band, stim, times, sr, Display_figure_instantes,
                          Save_figure_instantes, Run_graficos_path, Len_Estimulos, ERP=True):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.mean(2)

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for i in range(Cant_Estimulos):
        if Stims_Order[i] == 'Spectrogram':
            Pesos = Pesos_totales_sujetos_todos_canales_copy[:, sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))].reshape(info['nchan'], 16, len(times)).mean(1)
        else:
            Pesos = Pesos_totales_sujetos_todos_canales_copy[:, sum(Len_Estimulos[j] for j in range(i)):sum(Len_Estimulos[j] for j in range(i + 1))]

        if ERP:
            Pesos = np.flip(Pesos, axis=1)

        offset = 0
        instantes_index = sgn.find_peaks(np.abs(Pesos.mean(0)[offset:]), height=np.abs(Pesos.mean(0)).max() * 0.3)[0] + offset

        instantes_de_interes = [i/ sr + times[0] for i in instantes_index if i / sr + times[0] >= 0]

        # Ploteo pesos y cabezas
        if Display_figure_instantes:
            plt.ion()
        else:
            plt.ioff()

        Blues = plt.cm.get_cmap('Blues').reversed()
        cmaps = ['Reds' if Pesos.mean(0)[i] > 0 else Blues for i in instantes_index if
                 i / sr + times[0] >= 0]

        fig, axs = plt.subplots(figsize=(10, 5), ncols=len(cmaps))
        fig.suptitle('Mean of $w$ among subjects - {} - {} Band'.format(Stims_Order[i], Band))
        for j in range(len(instantes_de_interes)):
            if len(cmaps)>1:
                ax = axs[j]
            else:
                ax = axs
            ax.set_title('{} ms'.format(int(instantes_de_interes[j] * 1000)), fontsize=18)
            fig.tight_layout()
            im = mne.viz.plot_topomap(Pesos[:, instantes_index[j]].ravel(), info, axes=ax,
                                      show=False,
                                      sphere=0.07, cmap=cmaps[j],
                                      vmin=Pesos[:, instantes_index[j]].min().round(3),
                                      vmax=Pesos[:, instantes_index[j]].max().round(3))
            cbar = plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.4,
                         boundaries=np.linspace(
                             Pesos[:, instantes_index[j]].min(),
                             Pesos[:, instantes_index[j]].max(), 100).round(3),
                         ticks=np.linspace(Pesos[:, instantes_index[j]].min(),
                                            Pesos[:, instantes_index[j]].max(), 4).round(3))
            cbar.ax.tick_params(labelsize=15)

        fig.tight_layout()

        if Save_figure_instantes:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)
            fig.savefig(save_path_graficos + 'Instantes_interes.png')
            fig.savefig(save_path_graficos + 'Instantes_interes.svg')

    return Pesos.mean(0)


def pearsonr_pval(x, y):
    return stats.pearsonr(x, y)[1]


def Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, stim, Len_Estimulos, info, times, sesiones, Display, Save, Run_graficos_path):

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for k in range(Cant_Estimulos):

        if Stims_Order[k] == 'Spectrogram':
            Pesos_totales_sujetos_todos_canales_reshaped = Pesos_totales_sujetos_todos_canales[:,
                                          sum(Len_Estimulos[j] for j in range(k)):sum(
                                              Len_Estimulos[j] for j in range(k + 1)), :]. \
                reshape(info['nchan'], 16, len(times), len(sesiones)*2).mean(1)
        else:
            Pesos_totales_sujetos_todos_canales_reshaped = Pesos_totales_sujetos_todos_canales

        Pesos_totales_sujetos_todos_canales_average = np.dstack(
            (Pesos_totales_sujetos_todos_canales_reshaped, Pesos_totales_sujetos_todos_canales_reshaped.mean(2)))
        Correlation_matrices = np.zeros((Pesos_totales_sujetos_todos_canales_average.shape[0],
                                         Pesos_totales_sujetos_todos_canales_average.shape[2],
                                         Pesos_totales_sujetos_todos_canales_average.shape[2]))
        for channel in range(len(Pesos_totales_sujetos_todos_canales_average)):
            Correlation_matrices[channel] = np.array(
                pd.DataFrame(Pesos_totales_sujetos_todos_canales_average[channel]).corr(method='pearson'))

        # Correlacion por sujeto
        Correlation_matrix = Correlation_matrices.mean(0)

        for i in range(len(Correlation_matrix)):
            Correlation_matrix[i, i] = Correlation_matrix[-1, i]

        lista_nombres = [i for i in np.arange(1, Pesos_totales_sujetos_todos_canales_reshaped.shape[-1] + 1)] + ['Promedio']
        Correlation_matrix = pd.DataFrame(Correlation_matrix[:-1, :-1])
        Correlation_matrix.columns = lista_nombres[:len(Correlation_matrix) - 1] + [lista_nombres[-1]]

        if Display:
            plt.ion()
        else:
            plt.ioff()

        mask = np.ones_like(Correlation_matrix)
        mask[np.tril_indices_from(mask)] = False

        # Calculo promedio
        corr_values = Correlation_matrices[channel][np.tril_indices(Correlation_matrices[channel].shape[0], k=0)]
        Correlation_mean, Correlation_std = np.mean(np.abs(corr_values)), np.std(np.abs(corr_values))

        fig, (ax, cax) = plt.subplots(nrows=2, figsize=(15, 13), gridspec_kw={"height_ratios": [1, 0.05]})
        fig.suptitle(f'Similarity among subject\'s {Stims_Order[k]} mTRFs', fontsize=26)
        ax.set_title('Mean: {:.3f} +/- {:.3f}'.format(Correlation_mean, Correlation_std), fontsize=18)
        sn.heatmap(Correlation_matrix, mask=mask, cmap="coolwarm", fmt='.2f', ax=ax,
                   annot=True, center=0, xticklabels=True, annot_kws={"size": 15},
                   cbar=False)

        ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(Correlation_matrix)], rotation='horizontal',
                           fontsize=19)
        ax.set_xticklabels(lista_nombres[:len(Correlation_matrix) - 1] + ['Mean of subjects'], rotation='horizontal',
                           ha='left', fontsize=19)

        sn.despine(right=True, left=True, bottom=True, top=True)
        fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
        cax.yaxis.set_tick_params(labelsize=20)
        cax.xaxis.set_tick_params(labelsize=20)
        cax.set_xlabel(xlabel= 'Correlation', fontsize=20)

        fig.tight_layout()

        if Save:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix_{}.png'.format(Stims_Order[k]))
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix_{}.svg'.format(Stims_Order[k]))


def Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display, Save, Run_graficos_path):
    Correlation_matrices = np.zeros((Pesos_totales_sujetos_todos_canales.shape[0],
                                     Pesos_totales_sujetos_todos_canales.shape[2],
                                     Pesos_totales_sujetos_todos_canales.shape[2]))
    for channel in range(len(Pesos_totales_sujetos_todos_canales)):
        Correlation_matrices[channel] = np.array(
            pd.DataFrame(Pesos_totales_sujetos_todos_canales[channel]).corr(method='pearson'))

    # Correlacion por canal
    Correlation_abs_channel_wise = np.zeros(len(Correlation_matrices))
    for channel in range(len(Correlation_matrices)):
        channel_corr_values = Correlation_matrices[channel][
            np.tril_indices(Correlation_matrices[channel].shape[0], k=-1)]
        Correlation_abs_channel_wise[channel] = np.mean(np.abs(channel_corr_values))

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    fig.suptitle('Channel-wise mTRFs similarity', fontsize=19)
    im = mne.viz.plot_topomap(Correlation_abs_channel_wise, info, axes=ax, show=False, sphere=0.07,
                              cmap='Greens', vmin=Correlation_abs_channel_wise.min(),
                              vmax=Correlation_abs_channel_wise.max())
    cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
    cbar.ax.yaxis.set_tick_params(labelsize=17)
    cbar.ax.set_ylabel(ylabel= 'Correlation', fontsize=17)

    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'Channel_correlation_topo.png')
        fig.savefig(save_path_graficos + 'Channel_correlation_topo.svg')


def plot_trf_tfce(Pesos_totales_sujetos_todos_canales, p, times, title, mcc, shape, graficos_save_path, Band, stim, n_permutations,  pval_trhesh=None, axes=None, fontsize=15,
                Display=False, Save=True):
    if Display:
        plt.ion()
    else:
        plt.ioff()

    if axes is None:
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.mean(2)

    fig.suptitle('{} - {}'.format(stim, Band), fontsize=18)


    spectrogram_weights_bands = Pesos_totales_sujetos_todos_canales_copy.mean(0)
    spectrogram_weights_bands = spectrogram_weights_bands.reshape(16, len(times))
    # Adapt for ERP time
    spectrogram_weights_bands = np.flip(spectrogram_weights_bands, axis=1)

    im = axs[0].pcolormesh(times * 1000, np.arange(16), spectrogram_weights_bands, cmap='jet',
                           vmin=-spectrogram_weights_bands.max(), vmax=spectrogram_weights_bands.max(),
                           shading='auto')
    axs[0].set(xlabel='Time (ms)', ylabel='Frequency (Hz)')

    Bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
    yticks_positions = np.arange(0, 16, 2)
    xticks_positions = [-100, 0, 100, 200, 300, 400, 500, 600]
    ticks_labels = [int(Bands_center[i]) for i in np.arange(0, len(Bands_center), 2)]
    axs[0].set_yticks(yticks_positions)
    axs[0].set_xticks(xticks_positions)
    axs[0].set_yticklabels(ticks_labels)
    axs[0].xaxis.label.set_size(fontsize)
    axs[0].yaxis.label.set_size(fontsize)
    axs[0].tick_params(axis='both', labelsize=fontsize)

    cbar = fig.colorbar(im, ax=axs[0], orientation='horizontal', shrink=0.7)
    cbar.set_label('mTRF amplitude (a.u.)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)


    if pval_trhesh:
        # Mask p-values over threshold
        p[p > pval_trhesh] = 1
    # p plot
    use_p = -np.log10(np.reshape(np.maximum(p, 1e-5), (shape[1], shape[2])))

    img = axs[1].pcolormesh(times * 1000, np.arange(shape[1]), np.flip(use_p, axis=0), cmap="inferno", shading='auto',
                        vmin=0, vmax=2.5)

    axs[1].set(xlabel='Time (ms)')
    axs[1].xaxis.label.set_size(fontsize)
    plt.xticks(fontsize=fontsize)

    cbar = fig.colorbar(ax=axs[1], orientation="horizontal", mappable=img, shrink=0.7)
    cbar.set_label(r"$-\log_{10}(p)$", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    if Display:
        text = fig.suptitle(title)
        if mcc:
            text.set_weight("bold")
        plt.subplots_adjust(0, 0.05, 1, 0.9, wspace=0, hspace=0)
        mne.viz.utils.plt_show()

    fig.tight_layout()

    if Save:
        graficos_save_path += 'TFCE/'
        os.makedirs(graficos_save_path, exist_ok=True)
        plt.savefig(graficos_save_path + f'trf_tfce_{pval_trhesh}_{n_permutations}.png')
        plt.savefig(graficos_save_path + f'trf_tfce_{pval_trhesh}_{n_permutations}.svg')


def ch_heatmap_topo(total_data, info, delays, times, Display, Save, graficos_save_path, title, total_subjects=18,
                    sesion=None, sujeto=None, fontsize=14):

    if total_data.shape == (info['nchan'], len(delays)):
        phase_sync_ch = total_data
    elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
        phase_sync_ch = total_data.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(figsize=(9, 5), nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1]})

    # Remove axes of column 2
    for ax_col in axs[:, 1]:
        ax_col.remove()

    # Add one axis in column
    ax = fig.add_subplot(1, 3, (3, 3))

    # Plot topo
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)
    max_pahse_sync = phase_sync_ch[:, max_t_lag]

    # ax.set_title('Mean = {:.3f} +/- {:.3f}'.format(max_pahse_sync.mean(), max_pahse_sync.std()))
    im = mne.viz.plot_topomap(max_pahse_sync, info, cmap='Reds',
                              vmin=max_pahse_sync.min(),
                              vmax=max_pahse_sync.max(),
                              show=False, sphere=0.07, axes=ax)
    cb = plt.colorbar(im[0], shrink=1, orientation='horizontal')
    cb.set_label('r')


    # Invert times for PLV plot
    phase_sync_ch = np.flip(phase_sync_ch)
    phase_sync_std = phase_sync_ch.std(0)
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)

    times_plot = np.flip(-times)

    im = axs[0, 0].pcolormesh(times_plot * 1000, np.arange(info['nchan']), phase_sync_ch, shading='auto')
    axs[0, 0].set_ylabel('Channels')
    axs[0, 0].set_xticks([])

    cbar = plt.colorbar(im, orientation='vertical', ax=axs[0, 0])
    cbar.set_label('PLV')

    axs[1, 0].plot(times_plot * 1000, phase_sync)
    axs[1, 0].fill_between(times_plot * 1000, phase_sync - phase_sync_std / 2, phase_sync + phase_sync_std / 2, alpha=.5)
    axs[1, 0].set_ylim([0, 0.08])
    axs[1, 0].vlines(times_plot[max_t_lag] * 1000, axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], linestyle='dashed', color='k',
                label='Max: {}ms'.format(int(times_plot[max_t_lag] * 1000)))
    axs[1, 0].set_xlabel('Time lag [ms]')
    axs[1, 0].set_ylabel('Mean {}'.format(title))
    # axs2.tick_params(axis='both', labelsize=12)
    axs[1, 0].set_xlim([times_plot[0] * 1000, times_plot[-1] * 1000])
    axs[1, 0].grid()
    axs[1, 0].legend()

    fig.tight_layout()

    # Change axis 0 to match axis 1 width after adding colorbar
    ax0_box = axs[0, 0].get_position().bounds
    ax1_box = axs[1, 0].get_position().bounds
    ax1_new_box = (ax1_box[0], ax1_box[1], ax0_box[2], ax1_box[3])
    axs[1, 0].set_position(ax1_new_box)

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if total_data.shape == (info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.png'.format(title, sesion, sujeto))
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.svg'.format(title, sesion, sujeto))
        elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}.png'.format(title))
            plt.savefig(graficos_save_path + 't_lags_{}.svg'.format(title))


def ch_heatmap_topo(total_data, info, delays, times, Display, Save, graficos_save_path, title, total_subjects=18,
                    sesion=None, sujeto=None, fontsize=14):

    if total_data.shape == (info['nchan'], len(delays)):
        phase_sync_ch = total_data
    elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
        phase_sync_ch = total_data.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({'font.size': fontsize})
    fig, axs = plt.subplots(figsize=(9, 5), nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1]})

    # Remove axes of column 2
    for ax_col in axs[:, 1]:
        ax_col.remove()

    # Add one axis in column
    ax = fig.add_subplot(1, 3, (3, 3))

    # Plot topo
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)
    max_pahse_sync = phase_sync_ch[:, max_t_lag]

    # ax.set_title('Mean = {:.3f} +/- {:.3f}'.format(max_pahse_sync.mean(), max_pahse_sync.std()))
    im = mne.viz.plot_topomap(max_pahse_sync, info, cmap='Reds',
                              vmin=max_pahse_sync.min(),
                              vmax=max_pahse_sync.max(),
                              show=False, sphere=0.07, axes=ax)
    cb = plt.colorbar(im[0], shrink=1, orientation='horizontal')
    cb.set_label('r')

    # Invert times for PLV plot
    phase_sync_ch = np.flip(phase_sync_ch)
    phase_sync_std = phase_sync_ch.std(0)
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)

    times_plot = np.flip(-times)

    im = axs[0, 0].pcolormesh(times_plot * 1000, np.arange(info['nchan']), phase_sync_ch, shading='auto')
    axs[0, 0].set_ylabel('Channels')
    axs[0, 0].set_xticks([])

    cbar = plt.colorbar(im, orientation='vertical', ax=axs[0, 0])
    cbar.set_label('PLV')

    axs[1, 0].plot(times_plot * 1000, phase_sync)
    axs[1, 0].fill_between(times_plot * 1000, phase_sync - phase_sync_std / 2, phase_sync + phase_sync_std / 2, alpha=.5)
    # axs[1, 0].set_ylim([0, 0.2])
    axs[1, 0].vlines(times_plot[max_t_lag] * 1000, axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1], linestyle='dashed', color='k',
                label='Max: {}ms'.format(int(times_plot[max_t_lag] * 1000)))
    axs[1, 0].set_xlabel('Time lag [ms]')
    axs[1, 0].set_ylabel('Mean {}'.format(title))
    # axs2.tick_params(axis='both', labelsize=12)
    axs[1, 0].set_xlim([times_plot[0] * 1000, times_plot[-1] * 1000])
    axs[1, 0].grid()
    axs[1, 0].legend()

    fig.tight_layout()

    # Change axis 0 to match axis 1 width after adding colorbar
    ax0_box = axs[0, 0].get_position().bounds
    ax1_box = axs[1, 0].get_position().bounds
    ax1_new_box = (ax1_box[0], ax1_box[1], ax0_box[2], ax1_box[3])
    axs[1, 0].set_position(ax1_new_box)

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if total_data.shape == (info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.png'.format(title, sesion, sujeto))
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.svg'.format(title, sesion, sujeto))
        elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}.png'.format(title))
            plt.savefig(graficos_save_path + 't_lags_{}.svg'.format(title))



def Plot_instantes_interes(Pesos_totales_sujetos_todos_canales, info, times, Display,
                           Save, Run_graficos_path, Len_Estimulos, stim, plot_times,
                           fontsize=16):

    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

    # Ploteo pesos y cabezas
    if Display:
        plt.ion()
    else:
        plt.ioff()

    plt.rcParams.update({'font.size': fontsize})

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for i in range(Cant_Estimulos):

        if Stims_Order[i] == 'Spectrogram':

            spectrogram_weights_chanels = Pesos_totales_sujetos_todos_canales_copy[:,
                                          sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))]. \
                reshape(info['nchan'], 16, len(times)).mean(1)

            # Adapt for ERP
            spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)

        else:
            mean_coefs = Pesos_totales_sujetos_todos_canales_copy[:,
                                     sum(Len_Estimulos[j] for j in range(i)):sum(
                                         Len_Estimulos[j] for j in range(i + 1))]
            # Adapt for ERP
            mean_coefs = np.flip(mean_coefs, axis=1)

            evoked = mne.EvokedArray(mean_coefs, info)

        evoked.times = times

        fig = evoked.plot_joint(times=plot_times, show=Display,
                                ts_args=dict(units=dict(eeg='$mTRF (a.u.)$', grad='fT/cm', mag='fT'),
                                             scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms'),
                                topomap_args=dict(vmin=-0.016, vmax=0.016, time_unit='ms', scalings=dict(eeg=1, grad=1, mag=1)))

        fig.set_size_inches(13,6)
        fig.suptitle('{}'.format(Stims_Order[i] if Cant_Estimulos > 1 else stim))
        axs = fig.axes
        if times[-1] > 0:
            axs[0].vlines(0, axs[0].get_ylim()[0], axs[0].get_ylim()[1], color='gray')

        if Save:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)

            fig.savefig(
                save_path_graficos + 'Instantes_interes_{}.png'.format(Stims_Order[i] if Cant_Estimulos > 1 else stim))
            fig.savefig(
                save_path_graficos + 'Instantes_interes_{}.svg'.format(Stims_Order[i] if Cant_Estimulos > 1 else stim))
