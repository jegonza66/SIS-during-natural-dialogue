import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy.signal as sgn
import os
import seaborn as sn
import mne
import Funciones
import librosa
from statannot import add_stat_annotation

def highlight_cell(x, y, ax=None, **kwargs):
    rect = plt.Rectangle((x - .5, y - .5), 1, 1, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


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
        # surviving_channels_names = [channel_names[j] for j in Canales_sobrevivientes]
        # mask = []
        # for j in range(len(channel_names)):
        #     if channel_names[j] in surviving_channels_names:
        #         mask.append(True)
        #     else:
        #         mask.append(False)
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
    # Correl_prom = np.mean(Corr_promedio)

    # mejor_canal_rmse = Rmse_promedio.argmax()
    Rmse_mejor_canal = Rmse_promedio.max()
    # Rmse_prom = np.mean(Rmse_promedio)

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

            # Pitch = subjects_pitch[0][sujeto_total]
            # Pitch_std = subjects_pitch[1][sujeto_total]
            # ax.hlines(Pitch, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='dashed', color='black', label='Speaker\'s Pitch')
            # ax.hlines(Pitch + Pitch_std/2, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='dashed', color='black')
            # ax.hlines(Pitch - Pitch_std / 2, ax.get_xlim()[0], ax.get_xlim()[1], linestyle='dashed', color='black')
            # ax.legend()

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

    # fig.tight_layout()

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
    im = mne.viz.plot_topomap(Correlaciones_promedio, info, cmap='Greys',
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

    good_channels_right = ['B27', 'B28', 'B29', 'B30', 'C4', 'C5', 'C6', 'C7', 'C9', 'C10', 'C13', 'C14']
    good_channels_left = ['D8', 'D9', 'D10', 'D11', 'D7', 'D6', 'D5', 'D4', 'C31', 'C32', 'C27', 'C26']

    # # Plot masked channels
    # mask = [i in good_channels_right + good_channels_left for i in info['ch_names']]
    # fig = plt.figure()
    # plt.title('Masked channels for lateralization comparisson', fontsize=19)
    # im = mne.viz.plot_topomap(np.zeros(info['nchan']), info, show=False, sphere=0.07,
    #                            mask=np.array(mask), mask_params=dict(marker='o', markerfacecolor='g',
    #                                                                  markeredgecolor='k', linewidth=0,
    #                                                                  markersize=4))
    # fig.tight_layout()
    #
    # if Save:
    #     save_path_graficos = Run_graficos_path
    #     os.makedirs(save_path_graficos, exist_ok=True)
    #     fig.savefig(save_path_graficos + 'left_right_channels_{}.svg'.format(title))
    #     fig.savefig(save_path_graficos + 'left_right_channels_{}.png'.format(title))


    mask_right = [i in good_channels_right for i in info['ch_names']]
    mask_left = [i in good_channels_left for i in info['ch_names']]
    corr_right = Correlaciones_promedio[mask_right]
    corr_left = Correlaciones_promedio[mask_left]
    corr = [corr_left, corr_right]

    # fig = plt.figure()
    # plt.boxplot(corr)
    # plt.xticks([1, 2], ['left', 'right'])
    # plt.tick_params(labelsize=15)
    # fig.tight_layout()

    fig = plt.figure()
    data = pd.DataFrame({'Left': corr_left, 'Right': corr_right})
    # ax = sn.boxplot(data=data, width=0.35, boxprops=dict(alpha=.8))
    ax = sn.boxplot(data=data, width=0.35)
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .8))
    sn.swarmplot(data=data, color=".25")
    plt.tick_params(labelsize=15)
    fig.tight_layout()

    if Save:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'left_vs_right_{}.svg'.format(title))
        fig.savefig(save_path_graficos + 'left_vs_right_{}.png'.format(title))

    return Correlaciones_promedio.mean(), Correlaciones_promedio.std()


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

    fig = plt.figure()
    plt.suptitle("Channels passing 5 test per subject - {}".format(title), fontsize=19)
    plt.title('Mean: {:.3f} +/- {:.3f}'.format(Canales_repetidos_sujetos.mean(), Canales_repetidos_sujetos.std()),
              fontsize=19)
    im = mne.viz.plot_topomap(Canales_repetidos_sujetos, info, cmap='Greys',
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


def regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display, Save, Run_graficos_path,
                       Len_Estimulos, stim, title=None, decorrelation_times=None):
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

    for i in range(Cant_Estimulos):
        fig, ax = plt.subplots(figsize=(15, 5))
        fig.suptitle('{}'.format(Stims_Order[i]), fontsize=23)

        if Stims_Order[i] == 'Spectrogram':

            # spectrogram_weights_bands = Pesos_totales_sujetos_todos_canales_copy[:,
            #                       sum(Len_Estimulos[j] for j in range(i)):sum(
            #                           Len_Estimulos[j] for j in range(i + 1))].mean(0)
            # spectrogram_weights_bands = spectrogram_weights_bands.reshape(16, len(times))
            #
            # im = ax.pcolormesh(times * 1000, np.arange(16), spectrogram_weights_bands, cmap='jet',
            #                    vmin=-spectrogram_weights_bands.max(), vmax=spectrogram_weights_bands.max(), shading='auto')
            # ax.set(xlabel='Time (ms)', ylabel='Hz')
            #
            # Bands_center = librosa.mel_frequencies(n_mels=18, fmin=62, fmax=8000)[1:-1]
            # ticks_positions = np.arange(0, 16, 2)
            # ticks_labels = [int(Bands_center[i]) for i in np.arange(0, len(Bands_center), 2)]
            # ax.set_yticks(ticks_positions)
            # ax.set_yticklabels(ticks_labels)
            # ax.xaxis.label.set_size(14)
            # ax.yaxis.label.set_size(14)
            # ax.tick_params(axis='both', labelsize=14)
            #
            # cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            # cbar.set_label('TRF', fontsize=13)
            # cbar.ax.tick_params(labelsize=12)
            #
            # fig.tight_layout()

            spectrogram_weights_chanels = Pesos_totales_sujetos_todos_canales_copy[:,
                                          sum(Len_Estimulos[j] for j in range(i)):sum(
                                              Len_Estimulos[j] for j in range(i + 1))]. \
                reshape(info['nchan'], 16, len(times)).mean(1)

            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=ax)
            ax.plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            if times[-1] > 0: ax.axvspan(0, ax.get_xlim()[1], alpha=0.4, color='grey', label='Unheard stimuli')
            if decorrelation_times and times[-1] > 0:
                ax.vlines(np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
                          color='red',
                          label='Decorrelation time')
                ax.axvspan(np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
                           np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
                           alpha=0.4, color='red', label='Decorrelation time std.')

            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid()
            ax.legend(fontsize=12, loc='lower left')
            fig.tight_layout()

        else:
            evoked = mne.EvokedArray(Pesos_totales_sujetos_todos_canales_copy[:,
                                     sum(Len_Estimulos[j] for j in range(i)):sum(
                                         Len_Estimulos[j] for j in range(i + 1))], info)
            evoked.times = times

            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms',
                        show=False, spatial_colors=True, unit=True, units='W', axes=ax)

            ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)
            if times[-1] > 0: ax.axvspan(0, ax.get_xlim()[1], alpha=0.4, color='grey', label='Unheard stimuli')
            if decorrelation_times and times[-1] > 0:
                ax.vlines(np.mean(decorrelation_times), ax.get_ylim()[0], ax.get_ylim()[1], linestyle='dashed',
                          color='red',
                          label='Decorrelation time')
                ax.axvspan(np.mean(decorrelation_times) - np.std(decorrelation_times) / 2,
                           np.mean(decorrelation_times) + np.std(decorrelation_times) / 2,
                           alpha=0.4, color='red', label='Decorrelation time std.')

            ax.xaxis.label.set_size(14)
            ax.yaxis.label.set_size(14)
            ax.tick_params(axis='both', labelsize=14)
            ax.grid()
            ax.legend(fontsize=12, loc='lower left')

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
                              Save, Run_graficos_path, Len_Estimulos, stim, Band, title=None):
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

            cbar = fig.colorbar(im, ax=axs[1], orientation='horizontal')
            cbar.set_label('TRF', fontsize=13)
            cbar.ax.tick_params(labelsize=12)

            evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
            evoked.times = times
            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=axs[0])
            axs[0].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            axs[0].axis('off')
            axs[0].legend(fontsize=12, loc="lower left")

            # axs[0].plot(times * 1000, spectrogram_weights.mean(0), "k-", label="Mean", zorder=130, linewidth=2)
            # axs[0].axis('off')
            # axs[0].legend(fontsize=12, loc='lower left')

            fig.tight_layout()

        else:
            mean_coefs = Pesos_totales_sujetos_todos_canales_copy[:,
                         sum(Len_Estimulos[j] for j in range(i)):sum(Len_Estimulos[j] for j in range(i + 1))]

            evoked = mne.EvokedArray(mean_coefs, info)
            evoked.times = times

            fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 3]})
            fig.suptitle('{} - {}'.format(Stims_Order[i], Band), fontsize=18)

            im = axs[1].pcolormesh(times * 1000, np.arange(info['nchan']), mean_coefs, cmap='jet',
                                   vmin=-(mean_coefs).max(),
                                   vmax=(mean_coefs).max(), shading='auto')
            axs[1].set(xlabel='Time (ms)', ylabel='Channel')

            axs[1].xaxis.label.set_size(14)
            axs[1].yaxis.label.set_size(14)
            axs[1].tick_params(axis='both', labelsize=14)

            cbar = fig.colorbar(im, ax=axs[1], orientation='horizontal')
            cbar.set_label('TRF', fontsize=13)
            cbar.ax.tick_params(labelsize=12)

            evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                        show=False, spatial_colors=True, unit=False, units='w', axes=axs[0])
            axs[0].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
            axs[0].axis('off')
            axs[0].legend(fontsize=12, loc="lower left")

            fig.tight_layout()

        if Save:
            save_path_graficos = Run_graficos_path
            os.makedirs(save_path_graficos, exist_ok=True)
            if title:
                fig.savefig(save_path_graficos + 'Regression_Weights_matrix_{}_{}.svg'.format(title, Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_marix_{}_{}.png'.format(title, Stims_Order[i]))
            else:
                fig.savefig(save_path_graficos + 'Regression_Weights_matrix_{}.svg'.format(Stims_Order[i]))
                fig.savefig(save_path_graficos + 'Regression_Weights_marix_{}.png'.format(Stims_Order[i]))


def Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, Band, times, sr, Display_figure_instantes,
                          Save_figure_instantes, Run_graficos_path):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0)

    offset = 0
    instantes_index = sgn.find_peaks(np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)[offset:]),
                                height=np.abs(Pesos_totales_sujetos_todos_canales_copy.mean(1)).max() * 0.3)[0] + offset

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
    # fig.suptitle('Mean of $w$ among subjects - {} Band'.format(Band))
    for i in range(len(instantes_de_interes)):
        if len(cmaps)>1:
            ax = axs[i]
        else:
            ax = axs
        ax.set_title('{} ms'.format(int(instantes_de_interes[i] * 1000)), fontsize = 18)
        fig.tight_layout()
        im = mne.viz.plot_topomap(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].ravel(), info, axes=ax,
                                  show=False,
                                  sphere=0.07, cmap=cmaps[i],
                                  vmin=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min().round(3),
                                  vmax=Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max().round(3))
        cbar = plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.4,
                     boundaries=np.linspace(
                         Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
                         Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 100).round(3),
                     ticks=[np.linspace(Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].min(),
                                        Pesos_totales_sujetos_todos_canales_copy[instantes_index[i]].max(), 4).round(3)])
        cbar.ax.tick_params(labelsize=15)

    fig.tight_layout()

    if Save_figure_instantes:
        save_path_graficos = Run_graficos_path
        os.makedirs(save_path_graficos, exist_ok=True)
        fig.savefig(save_path_graficos + 'Instantes_interes.png')
        fig.savefig(save_path_graficos + 'Instantes_interes.svg')

    return Pesos_totales_sujetos_todos_canales_copy.mean(1)


def pearsonr_pval(x, y):
    return stats.pearsonr(x, y)[1]


def Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, Display, Save, Run_graficos_path, title=None):
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
    fig.suptitle('Similarity among subject\'s mTRFs', fontsize=26)
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
        if title:
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix_{}.png'.format(title))
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix_{}.svg'.format(title))
        else:
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix.png')
            fig.savefig(save_path_graficos + 'TRF_correlation_matrix.svg')


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
                Save, Run_graficos_path, Cant_Estimulos, Stims_Order, stim, decorrelation_times=None):
    # Armo pesos promedio por canal de todos los sujetos que por lo menos tuvieron un buen canal
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales.swapaxes(0, 2)
    Pesos_totales_sujetos_todos_canales_copy = Pesos_totales_sujetos_todos_canales_copy.mean(0).transpose()

    # Ploteo pesos y cabezas
    if Display:
        plt.ion()
    else:
        plt.ioff()

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
        if times[-1] > 0: ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Unheard stimuli')
        if decorrelation_times and times[-1] > 0:
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


def decoding_t_lags(Correlaciones_totales_sujetos, times, Display, Save, Run_graficos_path):
    Corr_time_sub = Correlaciones_totales_sujetos.mean(0)
    mean_time_corr = Corr_time_sub.mean(1)
    std_time_corr = Corr_time_sub.std(1)

    if Display:
        plt.ion()
    else:
        plt.ioff()

    fig, ax = plt.subplots()
    plt.plot(times, mean_time_corr)
    plt.fill_between(times, mean_time_corr - std_time_corr/2, mean_time_corr + std_time_corr/2)
    plt.xlabel('Time lag [ms]')
    plt.ylabel('Correlation')
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.tick_params(axis='both', labelsize=15)

    if Save:
        os.makedirs(Run_graficos_path, exist_ok=True)
        fig.savefig(Run_graficos_path + 'Correlation_time_lags.svg')
        fig.savefig(Run_graficos_path + 'Correlation_time_lags.png')


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