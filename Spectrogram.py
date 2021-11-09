from scipy.fft import fftshift
from scipy import signal
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display

s = 21
trial = 1
canal = 1
sr = 16000

ubi = "Datos/wavs/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal)+".wav" 
wav = wavfile.read(ubi)[1]
wav = wav.astype("float")

## PLT
plt.figure()
spectrogram = plt.specgram(wav, Fs=16000, NFFT=2048, noverlap=2048-512, scale='dB')

## SCIPY
f, t, Sxx = signal.spectrogram(wav[:10000], fs=16000, nperseg=2048, noverlap=512)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


## LIBROSA
n_fft = 2048
hop_length = 512
n_mels = 128

S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
S_DB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')