# -*- coding: utf-8 -*-
"""
Created on Fri May 28 20:51:04 2021

@author: joaco
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import librosa
from scipy import signal as sgn

def butter_bandpass_filter(data, frecuencia, sampling_freq, order, axis):
    frecuencia /= (sampling_freq/2) 
    b, a = sgn.butter(order, frecuencia, btype='lowpass')
    y = sgn.filtfilt(b, a, data, axis = axis, padlen = None)
    return y

s = 21
trial = 1
canal = 2
channel = canal
largo_envelope = 125 #ventana de 0.5 sgundos
salto_envelope = 125 # avanza de a 125 muestras

ubi = "Datos/wavs/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(canal)+".wav" 
wav1 = wavfile.read(ubi)[1]
wav1 = wav1.astype("float")


# Veo que hace Hilbert
#hilbert te devuelve un array de complejos. SI lo plotes, plotea la parte real. 
#Si tomas módulo, toma el modulo de los completjos.

plt.figure()
plt.plot(wav1)
plt.plot(sgn.hilbert(wav1))
plt.plot(np.imag(sgn.hilbert(wav1)))
plt.plot(np.real(sgn.hilbert(wav1)))
plt.plot(np.abs(sgn.hilbert(wav1)))
hilbert = sgn.hilbert(wav1)



# Hilbert
envelope = np.abs(sgn.hilbert(wav1))
envelope = butter_bandpass_filter(envelope, 25, 16000, 3, 0)
window_size = 125
stride = 125
envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size <= len(envelope)])

plt.figure()
plt.plot(wav1, label = "signal")
plt.plot(np.linspace(0,len(wav1),len(envelope)), envelope, label="envelope")
plt.legend(loc = 'upper right')
plt.grid()
plt.title("Hilbert butter 25 Hz")


### Librosa
salto = salto_envelope 
largo = largo_envelope 
envelope = librosa.feature.rms(wav1,frame_length=largo, hop_length=salto)
envelope = envelope.ravel().flatten()

plt.figure()
plt.plot(wav1, label = "signal")
plt.plot(np.linspace(0,len(wav1),len(envelope)), envelope, label="envelope")
plt.legend(loc = 'upper right')
plt.grid()
plt.title("librosa {}".format(largo_envelope))


#%% Ver espectro
from numpy.fft import fft, fftfreq        

sp = fft(envelope)
freq = fftfreq(envelope.shape[-1],d=1/16000)
plt.figure()
plt.plot(freq,np.abs(sp))
plt.xlim([-10, 6000])
plt.ylim([-10, 3e6])
plt.grid()
plt.title('Sin filtro')
plt.xlabel('Frecuency [Hz]')
plt.ylabel('Amplitud')