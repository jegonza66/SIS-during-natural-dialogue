# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:43:03 2021

@author: joaco
"""
from scipy.fft import fftshift
from scipy import signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

s = 21
trial = 2
canal = 1

ubi = "Datos/wavs/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal)+".wav" 
wav1 = wavfile.read(ubi)[1]
wav1 = wav1.astype("float")

plt.specgram(wav1[:1000000], Fs=16000)

f, t, Sxx = signal.spectrogram(wav1[:10000], fs=16000, nperseg=256, noverlap=128)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
