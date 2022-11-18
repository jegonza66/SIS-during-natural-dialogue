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


def lateralized_channels(info, channels_right=None, channels_left=None):
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
    im = mne.viz.plot_topomap(np.zeros(info['nchan']), info, show=False, sphere=0.07,
                               mask=np.array(mask), mask_params=dict(marker='o', markerfacecolor='g',
                                                                     markeredgecolor='k', linewidth=0,
                                                                     markersize=4))
    fig.tight_layout()


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
                     ticks=[np.linspace(Valores_promedio.min(), Valores_promedio.max(), 9).round(decimals=3)])

    else:
        fig, ax = plt.subplots()
        plt.suptitle("Sesion{} Sujeto{}\n{} = {:.3f} +/- {:.3f}".format(sesion, sujeto, name, Valores_promedio.mean(),
                                                                        Valores_promedio.std()), fontsize=19)
        im = mne.viz.plot_topomap(Valores_promedio, info, axes=ax, show=False, sphere=0.07, cmap='Greys',
                                  vmin=Valores_promedio.min(), vmax=Valores_promedio.max())
        plt.colorbar(im[0], ax=ax, shrink=0.85, label=name, orientation='horizontal',
                     boundaries=np.linspace(Valores_promedio.min().round(decimals=3),
                                            Valores_promedio.max().round(decimals=3), 100),
                     ticks=[np.linspace(Valores_promedio.min(), Valores_promedio.max(), 9).round(decimals=3)])
        fig.tight_layout()
    if Save:
        save_path_cabezas = Run_graficos_path + 'Cabezas_canales/'
        try:
            os.makedirs(save_path_cabezas)
        except:
            pass
        fig.savefig(save_path_cabezas + '{}_Cabeza_Sesion{}_Sujeto{}.png'.format(name, sesion, sujeto))


def corr_sujeto_decoding(sesion, sujeto, Valores_promedio, Display, name, Save, Run_graficos_path):
    if Display:
        plt.ion()
    else:
        plt.ioff()

    data = pd.DataFrame({name: Valores_promedio})
    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel(name)
    plt.title('{}:{:.3f} +/- {:.3f}'.format(name, np.mean(Valores_promedio), np.std(Valores_promedio), fontsize=19))

    if Save:
        save_path_cabezas = Run_graficos_path + 'Corr_sujetos/'
        try:
            os.makedirs(save_path_cabezas)
        except:
            pass
        fig.savefig(save_path_cabezas + '{}_Sesion{}_Sujeto{}.png'.format(name, sesion, sujeto))


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


def plot_grafico_shadows(Display, sesion, sujeto, best_alpha,
                         Canales_sobrevivientes_corr, info, sr,
                         Corr_promedio, Save, Run_graficos_path,
                         Corr_buenas_ronda_canal, Correlaciones_fake):
    # Defino cosas que voy a graficar
    Correlaciones_fake_min = Correlaciones_fake.min(1).min(0)
    Correlaciones_fake_max = Correlaciones_fake.max(1).max(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('Session {} - Subject {}'.format(sesion, sujeto))

    ax.plot(Corr_promedio, '.', color='C0', label="Mean of correlations among folds (Discarded)")
    ax.plot(Canales_sobrevivientes_corr, Corr_promedio[Canales_sobrevivientes_corr], '*',
            color='C1', label="Mean of correlations among folds (Test passed)")

    ax.fill_between(np.arange(len(Corr_promedio)), Corr_buenas_ronda_canal.min(0),
                    Corr_buenas_ronda_canal.max(0), alpha=0.5,
                    label='Correlation distribution (Real data)')
    ax.fill_between(np.arange(len(Corr_promedio)), Correlaciones_fake_min,
                    Correlaciones_fake_max, alpha=0.5,
                    label='Correlation distribution (Random data)')
    ax.set_xlim([-1, 129])
    ax.set_xlabel('Channels', fontsize=15)
    ax.set_ylabel('Correlation', fontsize=15)
    ax.legend(fontsize=13, loc="lower right")
    ax.grid()

    ax.xaxis.set_tick_params(labelsize=13)
    ax.yaxis.set_tick_params(labelsize=13)
    fig.tight_layout()

    if not len(Canales_sobrevivientes_corr): plt.text(64, np.max(abs(Corr_buenas_ronda_canal)) / 2,
                                                      "No surviving channels", size='xx-large', ha='center')

    if Save:
        save_path_graficos = Run_graficos_path + 'Correlation_shadows/'
        try:
            os.makedirs(save_path_graficos)
        except:
            pass
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
        save_path_graficos = 'gráficos/PSD/Zoom/{}/{}/'.format(save_path, Band)
        os.makedirs(save_path_graficos, exist_ok=True)
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.png'.format(sesion, sujeto, Band))
        plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.svg'.format(sesion, sujeto, Band))


def Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display, Save, Run_graficos_path, title):

    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig = plt.figure()
    plt.suptitle("Mean {} per channel among subjects".format(title), fontsize=19)
    plt.title('{} = {:.3f} +/- {:.3f}'.format(title, Correlaciones_promedio.mean(), Correlaciones_promedio.std()),
              fontsize=19)
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['greenyellow', 'yellow', 'orange', 'red'])
    im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='OrRd',
                              vmin=Correlaciones_promedio.min(), vmax=Correlaciones_promedio.max(),
                              show=False, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
    cb.ax.tick_params(labelsize=19)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + '{}_promedio.svg'.format(title))
        fig.savefig(save_path_graficos + '{}_promedio.png'.format(title))

    # Lateralization comparison
    good_channels_right = ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'B31', 'C3']
    good_channels_left = ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'D12', 'D3']
    mask_right = [i in good_channels_right for i in info['ch_names']]
    mask_left = [i in good_channels_left for i in info['ch_names']]
    corr_right = Correlaciones_promedio[mask_right]
    corr_left = Correlaciones_promedio[mask_left]

    fig = plt.figure()
    data = pd.DataFrame({'Left': corr_left, 'Right': corr_right})
    ax = sn.boxplot(data=data, width=0.35)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .8))
    sn.swarmplot(data=data, color=".25")
    plt.tick_params(labelsize=15)

    add_stat_annotation(ax, data=data, box_pairs=[('Left', 'Right')],
                        test='Wilcoxon', text_format='full', loc='outside')
    test_results = wilcoxon(data['Left'], data['Right'])

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'left_vs_right_{}.svg'.format(title))
        fig.savefig(save_path_graficos + 'left_vs_right_{}.png'.format(title))

    return (Correlaciones_promedio.mean(), Correlaciones_promedio.std()), test_results


def violin_plot_decoding(Correlaciones_totales_sujetos, Display, Save, Run_graficos_path, title):

    data = pd.DataFrame({title: Correlaciones_totales_sujetos.ravel()})
    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel(title)
    plt.title('{}:{:.3f} +/- {:.3f}'.format(title, np.mean(Correlaciones_totales_sujetos),
                                                     np.std(Correlaciones_totales_sujetos), fontsize=19))

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + '{}_promedio.svg'.format(title))
        fig.savefig(save_path_graficos + '{}_promedio.png'.format(title))

    return Correlaciones_totales_sujetos.mean(), Correlaciones_totales_sujetos.std()


def Cabezas_3d(Correlaciones_totales_sujetos, info, Display, Save, Run_graficos_path, title):
    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    sample_data_folder = mne.datasets.sample.data_path()
    subjects_dir = os.path.join(sample_data_folder, 'subjects')
    sample_data_trans_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                          'sample_audvis_raw-trans.fif')

    evoked = mne.EvokedArray(np.array([Correlaciones_promedio,]).transpose(), info)
    field_map = mne.make_field_map(evoked, trans=sample_data_trans_file,
                                   subject='sample', subjects_dir=subjects_dir, ch_type='eeg',
                                   meg_surf='head')

    fig = evoked.plot_field(field_map, time=0)
    xy, im = mne.viz.snapshot_brain_montage(fig, info)
    # mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title('Correlation', size='large')
    ax.imshow(im)

    if Save:
        try:
            os.makedirs(Run_graficos_path)
        except:
            pass
        fig.savefig(Run_graficos_path + '{}.svg'.format(title))
        fig.savefig(Run_graficos_path + '{}.png'.format(title))

    return Correlaciones_promedio.mean(), Correlaciones_promedio.std()


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
                       Len_Estimulos, stim, ERP=True, title=None, decorrelation_times=None):
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
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.suptitle('{}'.format(Stims_Order[i]), fontsize=23)

        if Stims_Order[i] == 'Spectrogram':

            spectrogram_weights_chanels = Pesos_totales_sujetos_todos_canales_copy[:,
                                          sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))]. \
                reshape(info['nchan'], 16, len(times)).mean(1)

            if ERP:
                # Adapt for ERP
                spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=ax)
            ax.plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            if times[0] < 0:
                ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
            if decorrelation_times and times[0] < 0:
                # ax.vlines(-np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
                #           color='red', label='Decorrelation time')
                ax.axvspan(-np.mean(decorrelation_times), 0, alpha=0.4, color='red', label=' Mean decorrelation time')
                # ax.axvspan(-np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
                #            -np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
                #            alpha=0.4, color='red', label='Decorrelation time std.')

            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            # ax.set_ylim([-0.016, 0.015])
            ax.tick_params(axis='both', labelsize=14)
            ax.grid()
            ax.legend(fontsize=12)
            fig.tight_layout()

        else:
            mean_coefs = Pesos_totales_sujetos_todos_canales_copy[:,
                                     sum(Len_Estimulos[j] for j in range(i)):sum(
                                         Len_Estimulos[j] for j in range(i + 1))]
            if ERP:
                # Adapt for ERP
                mean_coefs = np.flip(mean_coefs, axis=1)

            evoked = mne.EvokedArray(mean_coefs, info)
            evoked.times = times

            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                        show=False, spatial_colors=True, unit=True, units='W', axes=ax)

            ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
            if times[0] < 0:
                ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
            # if decorrelation_times and times[0] < 0:
            #     ax.vlines(-np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
            #               color='red', label='Decorrelation time')
                ax.axvspan(-np.mean(decorrelation_times), 0, alpha=0.4, color='red', label=' Mean decorrelation time')
                # ax.axvspan(-np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
                #            -np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
                #            alpha=0.4, color='red', label='Decorrelation time std.')

            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid()
            ax.legend(fontsize=12)

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
            axs[0].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            # if times[0] < 0:
            #     axs[0].axvspan(axs[0].get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
            axs[0].axis('off')
            axs[0].legend(fontsize=10)

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
            cbar.set_label('TRF', fontsize=13)
            cbar.ax.tick_params(labelsize=12)

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

            axs[0].axvline(0, axs[0].get_ylim()[0], axs[0].get_ylim()[1], color='grey')
            axs[0].axhline(0, axs[0].get_xlim()[0], axs[0].get_xlim()[1], color='grey')
            evoked = mne.EvokedArray(mean_coefs, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=axs[0])
            axs[0].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            axs[0].axis('off')
            axs[0].legend(fontsize=12)

            im = axs[1].pcolormesh(times * 1000, np.arange(info['nchan']), mean_coefs, cmap='jet',
                                   vmin=-(mean_coefs).max(), vmax=(mean_coefs).max(), shading='auto')
            axs[1].set(xlabel='Time (ms)', ylabel='Channel')
            axs[1].xaxis.label.set_size(14)
            axs[1].yaxis.label.set_size(14)
            axs[1].tick_params(axis='both', labelsize=14)
            cbar = fig.colorbar(im, ax=axs[1], orientation='vertical')
            cbar.set_label('TRF', fontsize=13)
            cbar.ax.tick_params(labelsize=12)

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
                         ticks=[np.linspace(Pesos[:, instantes_index[j]].min(),
                                            Pesos[:, instantes_index[j]].max(), 4).round(3)])
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


def Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, stim, Len_Estimulos, info, times, Display, Save, Run_graficos_path):

    Stims_Order = stim.split('_')
    Cant_Estimulos = len(Len_Estimulos)

    for k in range(Cant_Estimulos):

        if Stims_Order[k] == 'Spectrogram':
            Pesos_totales_sujetos_todos_canales = Pesos_totales_sujetos_todos_canales[:,
                                          sum(Len_Estimulos[j] for j in range(k)):sum(
                                              Len_Estimulos[j] for j in range(k + 1)), :]. \
                reshape(info['nchan'], 16, len(times), 18).mean(1)

        Pesos_totales_sujetos_todos_canales_average = np.dstack(
            (Pesos_totales_sujetos_todos_canales, Pesos_totales_sujetos_todos_canales.mean(2)))
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

        lista_nombres = [i for i in np.arange(1, Pesos_totales_sujetos_todos_canales.shape[-1] + 1)] + ['Promedio']
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


def PSD_boxplot(psd_pred_correlations, psd_rand_correlations, Display, Save, Run_graficos_path):
    psd_rand_correlations = Funciones.flatten_list(psd_rand_correlations)

    data = pd.DataFrame({'Prediction': psd_pred_correlations, 'Random': psd_rand_correlations})
    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    sn.violinplot(data=data, ax=ax)
    plt.ylim([-0.2, 1])
    plt.ylabel('Correlation')
    plt.title('Prediction Correlation:{:.2f} +/- {:.2f}\n'
              'Random Correlation:{:.2f} +/- {:.2f}'.format(np.mean(psd_pred_correlations), np.std(psd_pred_correlations),
                                                            np.mean(psd_rand_correlations), np.std(psd_rand_correlations)))
    add_stat_annotation(ax, data=data, box_pairs=[(('Prediction'), ('Random'))],
                        test='t-test_ind', text_format='full', loc='inside', verbose=2)

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'PSD Boxplot.png')
        fig.savefig(save_path_graficos + 'PSD Boxplot.svg')


def weights_ERP(Pesos_totales_sujetos_todos_canales, info, times, Display,
                Save, Run_graficos_path, Len_Estimulos, stim, decorrelation_times=None):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

    # Ploteo pesos y cabezas
    if Display:
        plt.ion()
    else:
        plt.ioff()

    Stims_Order = stim.split('_')

    Cant_Estimulos = len(Len_Estimulos)
    for j in range(Cant_Estimulos):
        Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)

        evoked = mne.EvokedArray(
            np.flip(Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], axis=1), info)
        times = -np.flip(times)
        evoked.times = times

        fig, ax = plt.subplots(figsize=(15, 5))
        fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim), fontsize=23)
        evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                    show=False, spatial_colors=True, unit=True, units='W', axes=ax)

        ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
        if times[-1] > 0:
            ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Unheard stimuli')
        if decorrelation_times and times[0] < 0:
            ax.vlines(-np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
                      color='red',
                      label='Decorrelation time')
            ax.axvspan(-np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
                       -np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
                       alpha=0.4, color='red', label='Decorrelation time std.')

        ax.xaxis.label.set_size(23)
        ax.yaxis.label.set_size(23)
        ax.tick_params(axis='both', labelsize=23)
        ax.grid()
        ax.legend(fontsize=15, loc='lower right')

        fig.tight_layout()

        if Save:
            os.makedirs(Run_graficos_path, exist_ok=True)
            fig.savefig(
                Run_graficos_path + 'Regression_Weights_{}.svg'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))
            fig.savefig(
                Run_graficos_path + 'Regression_Weights_{}.png'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))


def decoding_t_lags(Correlaciones_totales_sujetos, times, Band, Display, Save, Run_graficos_path):
    Corr_time_sub = Correlaciones_totales_sujetos.mean(0)
    mean_time_corr = np.flip(Corr_time_sub.mean(1))
    std_time_corr = np.flip(Corr_time_sub.std(1))

    plot_times = -np.flip(times)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    # get max correlation t_lag
    max_t_lag = np.argmax(mean_time_corr)

    fig, ax = plt.subplots()
    plt.plot(plot_times, mean_time_corr)
    plt.title('{}'.format(Band))
    plt.fill_between(plot_times, mean_time_corr - std_time_corr/2, mean_time_corr + std_time_corr/2, alpha=.5)
    plt.vlines(plot_times[max_t_lag], ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed', color='k',
               label='Max. correlation delay: {:.2f}s'.format(plot_times[max_t_lag]))
    plt.xlabel('Time lag [s]')
    plt.ylabel('Correlation')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis='both', labelsize=15)
    plt.grid()
    plt.legend()

    if Save:
        os.makedirs(Run_graficos_path, exist_ok=True)
        fig.savefig(Run_graficos_path + 'Correlation_time_lags_{}.svg'.format(Band))
        fig.savefig(Run_graficos_path + 'Correlation_time_lags_{}.png'.format(Band))


def Brain_sync(data, Band, info, Display, Save, graficos_save_path, total_subjects=18, sesion=None, sujeto=None):

    if Display:
        plt.ion()
    else:
        plt.ioff()

    if data.shape == (total_subjects, info['nchan'], info['nchan']):
        data_ch = data.mean(0)
    elif data.shape == (info['nchan'], info['nchan']):
        data_ch = data

    plt.figure(figsize=(10, 8))
    plt.title('Inter Brain Phase Synchornization - {}'.format(Band), fontsize=14)
    plt.imshow(data_ch)
    plt.xticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4], rotation=45)
    plt.yticks(np.arange(0, info['nchan'], 4), labels=info['ch_names'][0:-1:4])
    plt.ylabel('Speaker', fontsize=13)
    plt.xlabel('Listener', fontsize=13)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=12)

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if data.shape == (total_subjects, info['nchan'], info['nchan']):
            plt.savefig(graficos_save_path + 'Inter Brain sync - {}.png'.format(Band))
            plt.savefig(graficos_save_path + 'Inter Brain sync - {}.svg'.format(Band))
        elif data.shape == (info['nchan'], info['nchan']):
            plt.savefig(graficos_save_path + 'Inter Brain sync - Sesion{}_Sujeto{}.png'.format(sesion, sujeto))
            plt.savefig(graficos_save_path + 'Inter Brain sync - Sesion{}_Sujeto{}.svg'.format(sesion, sujeto))


def ch_heatmap_topo(total_data, Band, info, delays, times, Display, Save, graficos_save_path, title, total_subjects=18,
                    sesion=None, sujeto=None, fontsize=None):

    if not fontsize:
        fontsize = 24

    if total_data.shape == (info['nchan'], len(delays)):
        phase_sync_ch = total_data
    elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
        phase_sync_ch = total_data.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    # Plot topo
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)
    max_pahse_sync = phase_sync_ch[:, max_t_lag]

    fig = plt.figure()
    plt.suptitle("{}".format(title), fontsize=fontsize)
    plt.title('Mean = {:.3f} +/- {:.3f}'.format(max_pahse_sync.mean(), max_pahse_sync.std()),
              fontsize=fontsize)
    im = mne.viz.plot_topomap(max_pahse_sync, info, cmap='Reds',
                              vmin=max_pahse_sync.min(),
                              vmax=max_pahse_sync.max(),
                              show=False, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.65, orientation='horizontal')
    cb.set_label('r', fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize)
    fig.tight_layout()

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if total_data.shape == (info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 'Topo_{}_Sesion{}_Sujeto{}.png'.format(title, sesion, sujeto))
            plt.savefig(graficos_save_path + 'Topo_{}_Sesion{}_Sujeto{}.svg'.format(title, sesion, sujeto))
        elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 'Topo_{}.png'.format(title))
            plt.savefig(graficos_save_path + 'Topo_{}.svg'.format(title))

    # Invert times for plot
    phase_sync_ch = np.flip(phase_sync_ch)
    phase_sync_std = phase_sync_ch.std(0)
    phase_sync = phase_sync_ch.mean(0)
    max_t_lag = np.argmax(phase_sync)

    times = np.flip(-times)

    fig = plt.figure()
    plt.suptitle("Time lagged {} - Band: {}\n"
                 "Max = {:.3f} +/- {:.3f}".format(title, Band, phase_sync[max_t_lag],
                                                  phase_sync_std[max_t_lag], fontsize=fontsize))

    axs1 = fig.add_axes([.2, 0.5, 0.65, 0.35])
    im = axs1.pcolormesh(times*1000, np.arange(info['nchan']), phase_sync_ch, shading='auto')
    axs1.set_ylabel('Channels', fontsize=fontsize)
    axs1.tick_params(axis='y', labelsize=fontsize)
    axs1.set_xticks([])

    cb_ax = fig.add_axes([.87, .5, .02, .35])
    cbar = fig.colorbar(im, orientation='vertical', cax=cb_ax)
    cbar.set_label('PLV', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    axs2 = fig.add_axes([.2, .1, 0.65, 0.35])
    axs2.plot(times*1000, phase_sync)
    axs2.fill_between(times*1000, phase_sync - phase_sync_std / 2, phase_sync + phase_sync_std / 2, alpha=.5)
    axs2.set_ylim([0, 0.08])
    axs2.vlines(times[max_t_lag]*1000, axs2.get_ylim()[0], axs2.get_ylim()[1], linestyle='dashed', color='k',
               label='Max. delay: {}ms'.format(int(times[max_t_lag]*1000)))
    axs2.set_xlabel('Time lag [ms]', fontsize=fontsize)
    axs2.set_ylabel('Mean {}'.format(title), fontsize=fontsize)
    axs2.tick_params(axis='both', labelsize=fontsize)
    axs2.set_xlim([times[0]*1000, times[-1]*1000])
    axs2.grid()
    axs2.legend()

    if Save:
        os.makedirs(graficos_save_path, exist_ok=True)
        if total_data.shape == (info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.png'.format(title, sesion, sujeto))
            plt.savefig(graficos_save_path + 't_lags_{}_Sesion{}_Sujeto{}.svg'.format(title, sesion, sujeto))
        elif total_data.shape == (total_subjects, info['nchan'], len(delays)):
            plt.savefig(graficos_save_path + 't_lags_{}.png'.format(title))
            plt.savefig(graficos_save_path + 't_lags_{}.svg'.format(title))























## VIEJAS NO SE USAN

def Plot_instantes_interes(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_figure_instantes,
                           Save_figure_instantes, Run_graficos_path, Cant_Estimulos, Stims_Order, stim,
                           Autocorrelation_value=0.1):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

    # Ploteo pesos y cabezas
    if Display_figure_instantes:
        plt.ion()
    else:
        plt.ioff()

    returns = []
    for j in range(Cant_Estimulos):
        curva_pesos_totales = Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)].mean(0)
        returns.append(curva_pesos_totales)

        if Autocorrelation_value and times[-1] > 0:
            weights_autocorr = Funciones.correlacion(curva_pesos_totales, curva_pesos_totales)

            for i in range(len(weights_autocorr)):
                if weights_autocorr[i] < Autocorrelation_value: break

                dif_paso = weights_autocorr[i - 1] - weights_autocorr[i]
                dif_01 = weights_autocorr[i - 1] - Autocorrelation_value
                dif_time = dif_01 / sr / dif_paso
                decorr_time = ((i - 1) / sr + dif_time) * 1000

            fig, ax = plt.subplots()
            plt.plot(np.arange(len(weights_autocorr)) * 1000 / sr, weights_autocorr)
            plt.title('Decorrelation time: {:.2f} ms'.format(decorr_time))
            plt.hlines(Autocorrelation_value, ax.get_xlim()[0], decorr_time, linestyle='dashed', color='black')
            plt.vlines(decorr_time, ax.get_ylim()[0], Autocorrelation_value, linestyle='dashed', color='black')
            plt.grid()
            plt.ylabel('Autocorrelation')
            plt.xlabel('Time [ms]')
            if Save_figure_instantes:
                save_path_graficos = Run_graficos_path
                try:
                    os.makedirs(save_path_graficos)
                except:
                    pass
                fig.savefig(save_path_graficos + 'Weights Autocorrelation.png')

        evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy[:, j * len(times):(j + 1) * len(times)], info)
        evoked.times = times

        instantes_index = sgn.find_peaks(np.abs(evoked._data.mean(0)), height=np.abs(evoked._data.mean(0)).max() * 0.4)[
            0]
        if not len(instantes_index): instantes_index = [np.abs(evoked._data.mean(0)).argmax()]
        instantes_de_interes = [i / sr + times[0] for i in instantes_index]  # if i/sr + times[0] < 0]

        fig = evoked.plot_joint(times=instantes_de_interes, show=False,
                                ts_args=dict(unit='False', units=dict(eeg='$w$', grad='fT/cm', mag='fT'),
                                             scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms'),
                                topomap_args=dict(vmin=evoked._data.min(),
                                                  vmax=evoked._data.max(),
                                                  time_unit='ms'))

        fig.suptitle('{}'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))
        fig.set_size_inches(12, 7)
        axs = fig.axes
        axs[0].plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
        axs[0].axvspan(0, axs[0].get_xlim()[1], alpha=0.4, color='grey', label='Unheard stimuli')
        if Autocorrelation_value and times[-1] > 0: axs[0].vlines(decorr_time, axs[0].get_ylim()[0],
                                                                  axs[0].get_ylim()[1], linestyle='dashed', color='red',
                                                                  label='Decorrelation time')
        axs[0].xaxis.label.set_size(13)
        axs[0].yaxis.label.set_size(13)
        axs[0].grid()
        axs[0].legend(fontsize=13, loc='lower left')

        Blues = plt.cm.get_cmap('Blues').reversed()
        cmaps = ['Reds' if evoked._data.mean(0)[i] > 0 else Blues for i in instantes_index]

        for i in range(len(instantes_de_interes)):
            axs[i + 1].clear()
            axs[i + 1].set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)), fontsize=11)
            im = mne.viz.plot_topomap(evoked._data[:, instantes_index[i]], info, axes=axs[i + 1],
                                      show=False, sphere=0.07, cmap=cmaps[i],
                                      vmin=evoked._data[:, instantes_index[i]].min(),
                                      vmax=evoked._data[:, instantes_index[i]].max())
            plt.colorbar(im[0], ax=axs[i + 1], orientation='vertical', shrink=0.8,
                         boundaries=np.linspace(evoked._data[:, instantes_index[i]].min().round(decimals=2),
                                                evoked._data[:, instantes_index[i]].max().round(decimals=2), 100),
                         ticks=[np.linspace(evoked._data[:, instantes_index[i]].min(),
                                            evoked._data[:, instantes_index[i]].max(), 4).round(decimals=2)])

        axs[i + 2].remove()
        axs[i + 4].remove()
        fig.tight_layout()

        if Save_figure_instantes:
            save_path_graficos = Run_graficos_path
            try:
                os.makedirs(save_path_graficos)
            except:
                pass
            fig.savefig(
                save_path_graficos + 'Instantes_interes_{}.svg'.format(Stims_Order[j] if Cant_Estimulos > 1 else stim))

    return returns


def Matriz_corr(Pesos_totales_sujetos_promedio, Pesos_totales_sujetos_todos_canales, sujeto_total, Display, Save,
                Run_graficos_path):
    # Armo df para correlacionar
    Pesos_totales_sujetos_promedio = Pesos_totales_sujetos_promedio[:sujeto_total]
    Pesos_totales_sujetos_promedio.append(
        Pesos_totales_sujetos_todos_canales.transpose().mean(0).mean(1))  # agrego pesos promedio de todos los sujetos
    lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18",
                     "Promedio"]
    Pesos_totales_sujetos_df = pd.DataFrame(Pesos_totales_sujetos_promedio).transpose()
    Pesos_totales_sujetos_df.columns = lista_nombres[:len(Pesos_totales_sujetos_df.columns) - 1] + [lista_nombres[-1]]

    pvals_matrix = Pesos_totales_sujetos_df.corr(method=pearsonr_pval)
    Correlation_matrix = np.array(Pesos_totales_sujetos_df.corr(method='pearson'))
    for i in range(len(Correlation_matrix)):
        Correlation_matrix[i, i] = Correlation_matrix[-1, i]

    Correlation_matrix = pd.DataFrame(Correlation_matrix[:-1, :-1])
    Correlation_matrix.columns = lista_nombres[:len(Correlation_matrix) - 1] + [lista_nombres[-1]]

    if Display:
        plt.ion()
    else:
        plt.ioff()

    mask = np.ones_like(Correlation_matrix)
    mask[np.tril_indices_from(mask)] = False

    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
    fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
    sn.heatmap(abs(Correlation_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
               annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
               cbar=False)

    ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(Correlation_matrix)], rotation='horizontal',
                       fontsize=19)
    ax.set_xticklabels(lista_nombres[:len(Correlation_matrix) - 1] + ['Mean of subjects'], rotation='horizontal',
                       ha='left', fontsize=19)

    sn.despine(right=True, left=True, bottom=True, top=True)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="horizontal")
    cax.yaxis.set_tick_params(labelsize=20)

    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        try:
            os.makedirs(save_path_graficos)
        except:
            pass
        fig.savefig(save_path_graficos + 'Correlation_matrix.png')


def Matriz_std_channel_wise(Pesos_totales_sujetos_todos_canales, Display, Save, Run_graficos_path):
    Pesos_totales_sujetos_todos_canales_average = np.dstack(
        (Pesos_totales_sujetos_todos_canales, Pesos_totales_sujetos_todos_canales.mean(2)))
    Correlation_matrices = np.zeros((Pesos_totales_sujetos_todos_canales_average.shape[0],
                                     Pesos_totales_sujetos_todos_canales_average.shape[2],
                                     Pesos_totales_sujetos_todos_canales_average.shape[2]))
    for channel in range(len(Pesos_totales_sujetos_todos_canales_average)):
        Correlation_matrices[channel] = np.array(
            pd.DataFrame(Pesos_totales_sujetos_todos_canales_average[channel]).corr(method='pearson'))

    # std por sujeto
    std_matrix = Correlation_matrices.std(0)

    for i in range(len(std_matrix)):
        std_matrix[i, i] = std_matrix[-1, i]

    lista_nombres = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Promedio"]
    std_matrix = pd.DataFrame(std_matrix[:-1, :-1])
    std_matrix.columns = lista_nombres[:len(std_matrix) - 1] + [lista_nombres[-1]]

    if Display:
        plt.ion()
    else:
        plt.ioff()

    mask = np.ones_like(std_matrix)
    mask[np.tril_indices_from(mask)] = False

    fig, (ax, cax) = plt.subplots(ncols=2, figsize=(15, 9), gridspec_kw={"width_ratios": [1, 0.05]})
    fig.suptitle('Absolute value of the correlation among subject\'s $w$', fontsize=26)
    sn.heatmap(abs(std_matrix), mask=mask, cmap="coolwarm", fmt='.3', ax=ax,
               annot=True, center=0, xticklabels=True, annot_kws={"size": 19},
               cbar=False)

    ax.set_yticklabels(['Mean of subjects'] + lista_nombres[1:len(std_matrix)], rotation='horizontal', fontsize=19)
    ax.set_xticklabels(lista_nombres[:len(std_matrix) - 1] + ['Mean of subjects'], rotation='horizontal', ha='left',
                       fontsize=19)

    sn.despine(right=True, left=True, bottom=True, top=True)
    fig.colorbar(ax.get_children()[0], cax=cax, orientation="vertical")
    cax.yaxis.set_tick_params(labelsize=20)

    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        try:
            os.makedirs(save_path_graficos)
        except:
            pass
        fig.savefig(save_path_graficos + 'Channelwise_std_matrix.png')


def Cabezas_corr_promedio_scaled(Correlaciones_totales_sujetos, info, Display, Save, Run_graficos_path, title):
    Correlaciones_promedio = Correlaciones_totales_sujetos.mean(0)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig = plt.figure()
    plt.suptitle("Mean {} per channel among subjects".format(title), fontsize=19)
    plt.title('{} = {:.3f} +/- {:.3f}'.format(title, Correlaciones_promedio.mean(), Correlaciones_promedio.std()),
              fontsize=19)
    im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='Greys', vmin=0, vmax=0.41, show=Display, sphere=0.07)
    cb = plt.colorbar(im[0], shrink=0.85, orientation='vertical')
    cb.ax.tick_params(labelsize=23)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + '{}_promedio_scaled.svg'.format(title))
        fig.savefig(save_path_graficos + '{}_promedio_sacled.png'.format(title))


def Plot_instantes_casera(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_figure_instantes,
                          Save_figure_instantes, Run_graficos_path):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0)

    instantes_index = sgn.find_peaks(np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)[50:]),
                                height=np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)).max() * 0.3)[0] + 50

    instantes_de_interes = [i/ sr + times[0] for i in instantes_index if i / sr + times[0] <= 0]

    # Ploteo pesos y cabezas
    if Display_figure_instantes:
        plt.ion()
    else:
        plt.ioff()

    Blues = plt.cm.get_cmap('Blues').reversed()
    cmaps = ['Reds' if Pesos_totales_sujetos_todos_canales_copy.mean(1)[i] > 0 else Blues for i in instantes_index if
             i / sr + times[0] <= 0]

    fig, axs = plt.subplots(figsize=(10, 5), ncols=len(cmaps))
    fig.suptitle('Mean of $w$ among subjects - {} Band'.format(Band))
    for i in range(len(instantes_de_interes)):
        ax = axs[0, i]
        ax.set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)))
        fig.tight_layout()
        im = mne.viz.plot_topomap(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].ravel(), info, axes=ax,
                                  show=False,
                                  sphere=0.07, cmap=cmaps[i],
                                  vmin=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
                                  vmax=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max())
        plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.9,
                     boundaries=np.linspace(
                         Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min().round(decimals=2),
                         Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max().round(decimals=2), 100),
                     ticks=[np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
                                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 4).round(
                         decimals=2)])

    axs[0, -1].remove()
    for ax_row in axs[1:]:
        for ax in ax_row:
            ax.remove()

    ax = fig.add_subplot(3, 1, (2, 3))
    evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy.transpose(), info)
    evoked.times = times

    evoked.plot(show=False, spatial_colors=True, scalings=dict(eeg=1, grad=1, mag=1),
                unit=True, units=dict(eeg='$w$'), axes=ax, zorder='unsorted', selectable=False,
                time_unit='ms')
    ax.plot(times * 1000, Pesos_totales_sujetos_todos_canales_copy.mean(1),
            'k--', label='Mean', zorder=130, linewidth=2)

    ax.axvspan(0, ax.get_xlim()[1], alpha=0.5, color='grey')
    ax.set_title("")
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.grid()
    ax.legend(fontsize=13, loc='upper right')

    fig.tight_layout()

    if Save_figure_instantes:
        save_path_graficos = Run_graficos_path
        try:
            os.makedirs(save_path_graficos)
        except:
            pass
        fig.savefig(save_path_graficos + 'Instantes_interes.png')

    return Pesos_totales_sujetos_todos_canales_copy.mean(1)




def plot_alphas(alphas, correlaciones, best_alpha_overall, lista_Rmse, linea, fino):
    # Plot correlations vs. alpha regularization value
    # cada linea es un canal
    fig = plt.figure(figsize=(10, 5))
    fig.clf()
    plt.subplot(1, 3, 1)
    plt.subplots_adjust(wspace=1)
    plt.plot(alphas, correlaciones, 'k')
    plt.gca().set_xscale('log')
    # en rojo: el maximo de las correlaciones
    # la linea azul marca el mejor alfa

    plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
    plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

    plt.plot(alphas, correlaciones.mean(1), '.r', linewidth=5)
    plt.xlabel('Alfa', fontsize=16)
    plt.ylabel('Correlación - Ridge set', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)

    # Como se ve sola la correlacion maxima para los distintos alfas
    plt.subplot(1, 3, 2)
    plt.plot(alphas, np.array(correlaciones).mean(1), '.r', linewidth=5)
    plt.plot(alphas, np.array(correlaciones).mean(1), '-r', linewidth=linea)

    if fino:
        plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
        plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

    plt.xlabel('Alfa', fontsize=16)
    plt.gca().set_xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    # el RMSE
    plt.subplot(1, 3, 3)
    plt.plot(alphas, np.array(lista_Rmse).min(1), '.r', linewidth=5)
    plt.plot(alphas, np.array(lista_Rmse).min(1), '-r', linewidth=2)

    if fino:
        plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])
        plt.plot([best_alpha_overall, best_alpha_overall], [plt.ylim()[0], plt.ylim()[1]])

    plt.xlabel('Alfa', fontsize=16)
    plt.ylabel('RMSE - Ridge set', fontsize=16)
    plt.gca().set_xscale('log')
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)

    titulo = "El mejor alfa es de: " + str(best_alpha_overall)
    plt.suptitle(titulo, fontsize=18)