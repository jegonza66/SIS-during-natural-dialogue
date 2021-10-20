import Run_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

Alphas = [10,100,500,1000]
Alphas = [100]

Bands_low = np.arange(0.1, 15, 0.2)
Bands_range = np.flip(np.arange(0.1, 15, 0.2))

Correlations = np.zeros((len(Bands_low), len(Bands_range)))
Errores = np.zeros((len(Bands_low), len(Bands_range)))

save_path = 'saves/Band_search/'
try:
    os.makedirs(save_path)
except:
    pass

for alpha in Alphas:
    for i, l_freq in enumerate(Bands_low):
        print("\rProgress: {}%".format(int((i + 1) * 100 / len(Bands_low))), end='')
        for j, h_freq in enumerate(Bands_low+Bands_range):
            Correlations[i,j], Errores[i,j] = Run_function.run_pipeline(Band=(l_freq,h_freq), alpha=alpha)


    f = open(save_path+'alpha_{}.pkl'.format(alpha), 'wb')
    pickle.dump([Correlations, Errores], f)
    f.close()

    plt.ion()
    fig = plt.figure()
    plt.title('Mean correlation values')
    plt.imshow(Correlations)
    xticks = np.arange(Correlations.shape[0])
    yticks = np.arange(Correlations.shape[1])
    plt.xticks(xticks, labels=Bands_low)
    plt.yticks(yticks, labels=np.sort((Bands_low + Bands_range))[::-1])
    plt.colorbar()
    plt.xlabel('Low frequency bandpass')
    plt.ylabel('Bandpass width in Hz')
    plt.tight_layout()

    fig.savefig('gráficos/Bands_Correlations_alpha{}.png'.format(alpha))
