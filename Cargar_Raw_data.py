# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 17:09:28 2021

@author: joaco
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import librosa
import librosa.display
import mne
from scipy import signal as sgn
import Processing

##### PARAMETROS #####
s = 21
trial = 1
channel = 2

valores_faltantes_pitch = 0
audio_sr = 16000
sampleStep = 0.01

sr = 128
tmin, tmax = 0.053, 0.3
delays = - np.arange(np.floor(tmin*sr), np.ceil(tmax*sr), dtype=int) 

##### EEG #####
eeg_fname = "Datos/EEG/S"+str(s)+"/s"+str(s)+"-"+str(channel)+"-Trial"+str(trial)+"-Deci-Filter-Trim-ICA-Pruned.set"
eeg = mne.io.read_raw_eeglab(eeg_fname)
eeg_freq = eeg.info.get("sfreq")
eeg.load_data()

##### WAV #####
wav_fname = "Datos/wavs/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(channel)+".wav" 
wav1 = wavfile.read(wav_fname)[1]
wav1 = wav1.astype("float")
samples, sampling_rate = librosa.load(wav_fname)

##### ENVELOPE #####

envelope = np.abs(sgn.hilbert(wav1))
envelope = Processing.butter_filter(envelope, frecuencias = 25, sampling_freq = sr,
                                    btype = 'lowpass', order = 3, axis = 0, ftype = 'NonCausal')
window_size = 125
stride = 125
envelope = np.array([np.mean(envelope[i:i+window_size]) for i in range(0, len(envelope), stride) if i+window_size <= len(envelope)])
envelope = envelope.ravel().flatten()

##### PITCH #####
pitch_fname = "Datos/Pitch/S"+str(s)+"/s"+str(s)+".objects."+ "{:02d}".format(trial)+".channel"+str(channel)+".txt"
read_file = pd.read_csv(pitch_fname)
       
time = np.array(read_file['time'])
pitch = np.array(read_file['pitch'])
intensity = np.array(read_file['intensity'])

pitch[pitch=='--undefined--'] = np.nan
pitch = np.array(pitch, dtype = float)

pitch_der = []
for i in range(len(pitch)-1):
    try: 
        diff = pitch[i+1] - pitch[i]
        pitch_der.append(diff if np.abs(diff)<20 else None)     
    except: pitch_der.append(None)
pitch_der.append(None)
pitch_der = np.array(pitch_der, dtype = float)

if not valores_faltantes_pitch:
    pitch[np.isnan(pitch)] = valores_faltantes_pitch
    pitch_der[np.isnan(pitch_der)] = valores_faltantes_pitch
elif not np.isfinite(valores_faltantes_pitch):
    pitch[np.isnan(pitch)] = float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = float(valores_faltantes_pitch)
elif np.isfinite(valores_faltantes_pitch):
    pitch[np.isnan(pitch)] = np.float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = np.float(valores_faltantes_pitch)
else: print('Invalid missing value for pitch {}'.format(valores_faltantes_pitch)+'\nMust be finite.')
   
pitch = np.array(np.repeat(pitch,audio_sr*sampleStep), dtype = float)
pitch = Processing.subsamplear(pitch,125)
pitch = Processing.matriz_shifteada(pitch,delays)

pitch_der = np.array(np.repeat(pitch_der, audio_sr*sampleStep), dtype = float)
pitch_der = Processing.subsamplear(pitch_der, 125)
pitch_der = Processing.matriz_shifteada(pitch_der,delays)

if not valores_faltantes_pitch:
    pitch[pitch=='--undefined--'] = valores_faltantes_pitch   
elif not np.isfinite(valores_faltantes_pitch):
    pitch[pitch=='--undefined--'] = float(valores_faltantes_pitch)
elif np.isfinite(valores_faltantes_pitch):
    pitch[pitch=='--undefined--'] = float(valores_faltantes_pitch)

pitch = np.array(pitch, dtype = float)

pitch_der = []
for i in range(len(pitch)-1):
    try: 
        diff = pitch[i+1] - pitch[i]
        pitch_der.append(diff if np.abs(diff)<20 else None)     
    except: pitch_der.append(None)
pitch_der.append(None)
pitch_der = np.array(pitch_der)


pitch = np.array(np.repeat(pitch,audio_sr*sampleStep), dtype = float)
pitch = Processing.subsamplear(pitch,125)

pitch_der = np.array(np.repeat(pitch_der,audio_sr*sampleStep), dtype = float)
pitch_der = Processing.subsamplear(pitch_der, 125)

##### PLOT #####
time = np.arange(len(wav1))/16000

plt.ion()
plt.figure()
plt.plot(time, wav1, label = 'Audio signal')
plt.plot(np.linspace(0, time[-1], len(pitch)), pitch, label = 'Pitch [Hz]')
plt.plot(np.linspace(0, time[-1], len(envelope)), envelope, label = 'Envelope')
plt.ylabel('Magnitude')
plt.xlabel('Time [s]')
plt.legend()


#%%
features = np.array(pitch).reshape(len(pitch),1)
nt,ndim = features.shape
dstims = []
for di,d in enumerate(delays):
    dstim = np.zeros((nt, ndim))
    if d<0: ## negative delay
        dstim[:d,:] = features[-d:,:] # The last d elements until the end
    elif d>0:
        dstim[d:,:] = features[:-d,:] # All but the last d elements
    else:
        dstim = features.copy()
    dstims.append(dstim)
dstims = np.hstack(dstims)


pitch = Processing.matriz_shifteada(pitch,delays)
