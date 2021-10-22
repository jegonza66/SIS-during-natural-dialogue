import Run_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

Alphas = [10, 100, 500, 1000]
Alphas = [100]

Bands_low = np.flip(np.arange(0.1, 12))
Bands_range = np.arange(1, 9, 0.5)

Correlations = np.zeros((len(Bands_low), len(Bands_range)))

save_path = 'saves/Band_search/'
try:
    os.makedirs(save_path)
except:
    pass

for alpha in Alphas:
    for i, l_freq in enumerate(Bands_low):
        for j, h_freq in enumerate(l_freq + Bands_range):
            Correlations[i, j], _ = Run_function.run_pipeline(stim='Envelope', Band=(l_freq, h_freq), alpha=alpha)

        print("\n\nProgress: {}%\n\n".format(int((i + 1) * 100 / len(Bands_low))))

    f = open(save_path + 'alpha_{}.pkl'.format(alpha), 'wb')
    pickle.dump(Correlations, f)
    f.close()

    max_corr = np.argwhere(Correlations == Correlations.max())[0]

    plt.ion()
    fig = plt.figure()
    plt.title('Mean correlation values')
    im = plt.imshow(Correlations)
    plt.text(max_corr[1], max_corr[0], Correlations.max().round(2), ha="center", va="center", color="red", size="small")
    ax = plt.gca()
    yticks = np.array(ax.get_yticks(), dtype=int)[1:-1]
    xticks = np.array(ax.get_xticks(), dtype=int)[1:-1]
    plt.yticks(yticks, labels=Bands_low[yticks].round(2))
    plt.xticks(xticks, labels=Bands_range[xticks].round(2))
    plt.colorbar(shrink=0.95)
    plt.ylabel('Low frequency bandpass')
    plt.xlabel('Bandpass width in Hz')
    plt.tight_layout()

    fig.savefig('gráficos/Bands_Correlations_alpha{}.png'.format(alpha))
