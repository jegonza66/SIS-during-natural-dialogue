import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import mne
from scipy import signal as sgn
import Processing
from numpy.fft import fft, fftfreq

## PARAMETROS
s = 26
trial = 1
channel = 2

valores_faltantes_pitch = 0
audio_sr = 16000
sampleStep = 0.01

sr = 128
tmin, tmax = 0.053, 0.3
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)

## EEG
Band = 'Theta'
l_freq_eeg, h_freq_eeg = Processing.band_freq(Band)

eeg_fname = "Datos/EEG/S" + str(s) + "/s" + str(s) + "-" + str(channel) + "-Trial" + str(trial) + "-Deci-Filter-Trim-ICA-Pruned.set"
eeg = mne.io.read_raw_eeglab(eeg_fname)
eeg_freq = eeg.info.get("sfreq")
info = eeg.info
eeg.load_data()
eeg = eeg.filter(l_freq=l_freq_eeg, h_freq=h_freq_eeg, phase='minimum')

eeg.plot()
plt.savefig('{}Theta.png'.format(s))

## PSD
psds_welch_mean, freqs_mean = mne.time_frequency.psd_array_welch(eeg._data, sfreq=eeg_freq, fmin=1, fmax=60)

fig, ax = plt.subplots()
evoked = mne.EvokedArray(psds_welch_mean, info)
evoked.times = freqs_mean
evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s',
            show=False, spatial_colors=True, unit=False, units='w', axes=ax)
ax.set_xlabel('Frequency [Hz]')
ax.grid()
# plt.savefig('NO_CAUSAL.png')

## WAV

wav_fname = "Datos/wavs/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
    channel) + ".wav"
wav1 = wavfile.read(wav_fname)[1]
wav1 = wav1.astype("float")
# samples, sampling_rate = librosa.load(wav_fname)

## ENVELOPE

envelope = np.abs(sgn.hilbert(wav1))
envelope = Processing.butter_filter(envelope, frecuencias=25, sampling_freq=16000,
                                    btype='lowpass', order=3, axis=0, ftype='Causal')

window_size = 125
stride = 125
envelope = np.array(
    [np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if i + window_size <= len(envelope)])
envelope = envelope.ravel().flatten()

## PLOT spectre
sp = fft(envelope)
freq = fftfreq(envelope.shape[-1], d=1 / 128)
plt.figure()
plt.plot(freq, np.abs(sp))
plt.xlim([0, 30])
plt.ylim([0, 3e4])
plt.grid()
plt.title('Hilbert + Butter + Prom')
plt.xlabel('Frecuency [Hz]')
plt.ylabel('Amplitud')

## PITCH
pitch_fname = "Datos/Pitch/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(trial) + ".channel" + str(
    channel) + ".txt"
read_file = pd.read_csv(pitch_fname)

time = np.array(read_file['time'])
pitch = np.array(read_file['pitch'])
intensity = np.array(read_file['intensity'])

pitch[pitch == '--undefined--'] = np.nan
pitch = np.array(pitch, dtype=float)

pitch_der = []
for i in range(len(pitch) - 1):
    try:
        diff = pitch[i + 1] - pitch[i]
        pitch_der.append(diff)
    except:
        pitch_der.append(None)
pitch_der.append(None)
pitch_der = np.array(pitch_der, dtype=float)

if not valores_faltantes_pitch:
    pitch[np.isnan(pitch)] = valores_faltantes_pitch
    pitch_der[np.isnan(pitch_der)] = valores_faltantes_pitch
elif not np.isfinite(valores_faltantes_pitch):
    pitch[np.isnan(pitch)] = float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = float(valores_faltantes_pitch)
elif np.isfinite(valores_faltantes_pitch):
    pitch[np.isnan(pitch)] = float(valores_faltantes_pitch)
    pitch_der[np.isnan(pitch_der)] = float(valores_faltantes_pitch)
else:
    print('Invalid missing value for pitch {}'.format(valores_faltantes_pitch) + '\nMust be finite.')

pitch = np.array(np.repeat(pitch, audio_sr * sampleStep), dtype=float)
pitch = Processing.subsamplear(pitch, 125)

pitch_der = np.array(np.repeat(pitch_der, audio_sr * sampleStep), dtype=float)
pitch_der = Processing.subsamplear(pitch_der, 125)

## Normalize features and leave nans for plot

no_nan_pitch = pitch[~np.isnan(pitch)]
no_nan_pitch_der = pitch_der[~np.isnan(pitch_der)]

norm = Processing.normalizar()
norm.normalize_01(envelope)
norm.normalize_01(no_nan_pitch)
norm.normalize_11(no_nan_pitch_der)
no_nan_pitch_der -= no_nan_pitch_der.mean()

pitch[~np.isnan(pitch)] =  no_nan_pitch
pitch_der[~np.isnan(pitch_der)] = no_nan_pitch_der

pitch = np.array(np.repeat(pitch, audio_sr * sampleStep), dtype=float)
pitch = Processing.subsamplear(pitch, 125)

pitch_der = np.array(np.repeat(pitch_der, audio_sr * sampleStep), dtype=float)
pitch_der = Processing.subsamplear(pitch_der, 125)

norm.normalize_11(wav1)
wav1 -= wav1.mean()

## PLOT
time = np.arange(len(wav1)) / 16000



# plt.ion()
# plt.figure()
# plt.plot(time, wav1, label='Audio signal')
plt.plot(np.linspace(0, time[-1], len(envelope)), envelope, label='Envelope Causal')
# plt.plot(np.linspace(0, time[-1], len(pitch)), pitch, label='Pitch [Hz]')
# plt.plot(np.linspace(0, time[-1], len(pitch_der)), pitch_der, label='Pitch derivate [Hz/s]')
plt.ylabel('Magnitude')
plt.xlabel('Time [s]')
plt.xlim([41,43])
plt.title('Scaled audio features')
plt.legend()