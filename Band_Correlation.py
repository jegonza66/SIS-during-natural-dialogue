import Plot
import Run_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


Stims = ['Envelope', 'Pitch', 'Spectrogram', 'Envelope_Pitch_Spectrogram']
Stims = ['Spectrogram']
Alphas = [100]

Bands_low = np.flip(np.arange(2, 9))
# Bands_range = np.arange(0.5, 9, 0.5)
Bands_range = np.arange(0.5, 7, 0.5)

save_path = 'saves/Bands Correlation/'
try:
    os.makedirs(save_path)
except:
    pass

for stim in Stims:
    print("\n\nStim: {}".format(stim))
    for alpha in Alphas:
        print("Alpha: {}".format(alpha))
        try:
            f = open(save_path + '{}_Band_{}_{}_{}_alpha_{}.pkl'.format(stim, Bands_low[-1], Bands_low[0], (Bands_range[1]-Bands_range[0]).round(1), alpha), 'rb')
            Correlations = pickle.load(f)
            f.close()
        except:
            Correlations = np.zeros((len(Bands_low), len(Bands_range)))
            for i, l_freq in enumerate(Bands_low):
                for j, h_freq in enumerate(l_freq + Bands_range):
                    Correlations[i, j], _ = Run_function.run_pipeline(stim=stim, Band=(l_freq, h_freq), alpha=alpha)
                    # print("\rProgress: {}%".format(int((j + 1) * 100 / len(Bands_range))), end='')
                print("\rProgress: {}%".format(int((i + 1) * 100 / len(Bands_low))), end="")

            f = open(save_path + '{}_Band_{}_{}_{}_alpha_{}.pkl'.format(stim, Bands_low[-1], Bands_low[0], (Bands_range[1]-Bands_range[0]).round(1), alpha), 'wb')
            pickle.dump(Correlations, f)
            f.close()

        max_range_percent = 99
        max_corr = np.argwhere(Correlations == Correlations.max())[0]
        max_corrs = np.argwhere(Correlations >= Correlations.max()*(max_range_percent/100))

        plt.ion()
        fig = plt.figure()
        plt.title('Mean correlation values - {}'.format(stim))
        im = plt.imshow(Correlations)
        plt.text(max_corr[1], max_corr[0], Correlations.max().round(2), ha="center", va="center", color="black", size="small")
        ax = plt.gca()
        yticks = np.array(ax.get_yticks(), dtype=int)[1:-1]
        xticks = np.array(ax.get_xticks(), dtype=int)[1:-1]
        plt.yticks(yticks, labels=Bands_low[yticks].round(2))
        plt.xticks(xticks, labels=Bands_range[xticks].round(2))
        plt.colorbar(shrink=0.7)
        plt.ylabel('Low frequency bandpass')
        plt.xlabel('Bandpass width in Hz')
        for i in range(len(max_corrs)):
            Plot.highlight_cell(max_corrs[i][1], max_corrs[i][0], ax=ax, fill=False, alpha=0.4, color='red', linewidth=2, label='{}% Max. Value'.format(max_range_percent))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()

        fig.savefig('gráficos/Bands Correlations/{}_Band_{}_{}_{}_alpha_{}.png'.format(stim, Bands_low[-1], Bands_low[0], (Bands_range[1]-Bands_range[0]).round(1), alpha))
        fig.savefig('gráficos/Bands Correlations/{}_Band_{}_{}_{}_alpha_{}.svg'.format(stim, Bands_low[-1], Bands_low[0], (Bands_range[1]-Bands_range[0]).round(1), alpha))
