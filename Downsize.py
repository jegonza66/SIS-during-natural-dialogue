import pathlib
import pickle
import numpy as np

folders_path_str = 'saves/Preprocesed_Data/tmin-0.6_tmax-0.003/Spectrogram'
folders_path = pathlib.Path(folders_path_str)
folders = list(folders_path.glob('*[!info.pkl]*'))


for folder in folders:
    print('\nFolder: {}'.format(folder))
    files_path = folder
    files = list(files_path.glob('*.pkl'))

    for i, file in enumerate(files):
        f = open(file, 'rb')
        Data = pickle.load(f)
        f.close()

        Data_low = []
        for variable in Data:
            variable = np.array(variable, dtype=np.float16)
            Data_low.append(variable)

        f = open(file, 'wb')
        pickle.dump(Data_low, f)
        f.close()

        print("\rProgress: {}%".format(int((i + 1) * 100 / len(files))), end='')