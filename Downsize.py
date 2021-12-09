import pathlib
import pickle
import numpy as np

folders_path_str = 'saves/Ridge/Fake_it/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax-0.003/'
folders_path = pathlib.Path(folders_path_str)
folders = list(folders_path.glob('*[!Spectrogram]*'))


for folder in folders:
    print('\nFolder: {}'.format(folder))
    files_path = folder
    files = list(files_path.glob('Pesos*.pkl'))

    for i, file in enumerate(files):
        f = open(file, 'rb')
        Data = pickle.load(f)
        f.close()

        Data_low = Data[:,:1000,:,:].mean(0)

        f = open(file, 'wb')
        pickle.dump(Data_low, f)
        f.close()

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(files))), end='')