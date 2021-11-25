import pathlib
import pickle
import numpy as np

folders_path_str = 'saves/Ridge/Fake_it/Stims_Normalize_EEG_Standarize/tmin-0.6_tmax-0.003'
folders_path = pathlib.Path(folders_path_str)
folders = list(folders_path.glob('*'))

for folder in folders:
    files_path = folder
    files = list(files_path.glob('Pesos*.pkl'))

    for file in files:
        f = open(file, 'rb')
        Data = pickle.load(f)
        f.close()

        Data = np.array(Data, dtype=np.float16)

        f = open(file, 'wb')
        pickle.dump(Data, f)
        f.close()