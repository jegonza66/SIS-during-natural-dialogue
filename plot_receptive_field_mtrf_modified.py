"""
.. _ex-receptive-field-mtrf:

=========================================
Receptive Field Estimation and Prediction
=========================================

This example reproduces figures from Lalor et al.'s mTRF toolbox in
MATLAB :footcite:`CrosseEtAl2016`. We will show how the
:class:`mne.decoding.ReceptiveField` class
can perform a similar function along with scikit-learn. We will first fit a
linear encoding model using the continuously-varying speech envelope to predict
activity of a 128 channel EEG system. Then, we will take the reverse approach
and try to predict the speech envelope from the EEG (known in the literature
as a decoding model, or simply stimulus reconstruction).

.. _figure 1: https://www.frontiersin.org/articles/10.3389/fnhum.2016.00604/full#F1
.. _figure 2: https://www.frontiersin.org/articles/10.3389/fnhum.2016.00604/full#F2
.. _figure 5: https://www.frontiersin.org/articles/10.3389/fnhum.2016.00604/full#F5
"""  # noqa: E501

# Authors: Chris Holdgraf <choldgraf@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#          Nicolas Barascud <nicolas.barascud@ens.fr>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 3

"""
MNE ENVIRONMENT: C:\ProgramData\Anaconda3\envs\mne\python.exe
"""
"""
import wave

##############################################################################
# Function to interpret wav files and return an int16 array by rudolfbyker in
# https://stackoverflow.com/a/31625891/13155520
def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

spf = wave.open(path+r'\wavs\S21\s21.objects.01.channel2.wav','rb')
nChannels,nFrames = spf.getnchannels(),spf.getnframes()
sampleRate, ampWidth = spf.getframerate(), spf.getsampwidth()
speech = spf.readframes(nFrames*nChannels)
spf.close()
speech = interpret_wav(speech, nFrames, nChannels, ampWidth, True).astype(float)/INTMAX

"""
"""
import scipy.io.wavfile as wavfile
import librosa as libro

speech = wavfile.read(path+r'\wavs\S21\s21.objects.01.channel2.wav')[1]
speech = speech.astype("float")

length, hop = 8000, 125 #int(speech.shape[0]/raw.shape[1])
envelope = libro.feature.rms(speech,frame_length=length, hop_length=hop)#,,
#speech = envelope.ravel().flatten()
"""
"""
import signal_envelope as se 
speech = se.read_wav(path+r'\wavs\S21\s21.objects.01.channel2.wav')[0]
envelope = se.get_frontiers(speech, mode=1)
speech = np.abs(speech[envelope])/ INTMAX
"""


import numpy as np
import matplotlib.pyplot as plt
#from scipy.io import loadmat, wavfile
#from os.path import join

from scipy import signal as sgn
from scipy.io import  wavfile

import mne
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

import pickle

###############################################################################
# Load the data from the publication
# ----------------------------------
#
# First we will load the data collected in :footcite:`CrosseEtAl2016`.
# In this experiment subjects listened to natural speech.
# Raw EEG and the speech stimulus are provided.
# We will load these below, downsampling the data in order to speed up
# computation since we know that our features are primarily low-frequency in
# nature. Then we'll visualize both the EEG and speech envelope.

path = r'Datos/'
decim = 4
INTMAX = np.iinfo(np.int16).max

s = 21
trial=1
canal=1
# Read EEG signals
raw = mne.io.read_raw_eeglab(path+"EEG/S"+str(s)+"/s"+str(s)+"-"+str(canal)+"-Trial"+str(trial)+"-Deci-Filter-Trim-ICA-Pruned.set")#.get_data()
# raw = mne.io.read_raw_eeglab(path+r'\EEG\S21\s21-1-Trial1-Deci-Filter-Trim-ICA-Pruned.set')#.get_data()
raw = raw.load_data().filter(l_freq = 1, h_freq=15).get_data()
# raw[:,:] = (raw[:,:]-raw[:,:].mean(0))/raw[:,:].std()
raw -= raw.min()
raw /= raw.max()

# Read and convert Wav files
speech = wavfile.read(path+r'\wavs\S21\s21.objects.01.channel2.wav')[1]
speech = np.abs(sgn.hilbert(speech))
speech -= speech.min()
speech /= speech.max()

# Downsample Speech to match EEG sample size
# EEG - sr = 512 Hz || Audio - sr = 16 kHz
dwnRate = speech.shape[0]/raw.shape[1]
# agregar filtro pasabajos
# 3rd 223 order Butterworth filter, cut off frequency 25 Hz
speech = mne.filter.resample(speech, down=dwnRate*decim, npad='auto')
raw = mne.filter.resample(raw, down=decim, npad='auto')
sfreq = 512 / decim

# Read in channel positions and create our MNE objects from the raw data
montage = mne.channels.make_standard_montage('biosemi128')
info = mne.create_info(montage.ch_names, sfreq, 'eeg').set_montage(montage)
raw = mne.io.RawArray(raw, info)
n_channels = len(raw.ch_names)

# Cargo todo
sesion = s
procesed_data_path = 'Preprocesed_data/'
f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 1), 'rb')
eeg_sujeto_1, dstims_para_sujeto_1 = pickle.load(f)
f.close()

f = open(procesed_data_path + 'Preprocesed_Data_Sesion{}_Sujeto{}.pkl'.format(sesion, 2), 'rb')
eeg_sujeto_2, dstims_para_sujeto_2 = pickle.load(f)
f.close()

f = open(procesed_data_path + 'info.pkl', 'rb')
info = pickle.load(f)
f.close()

raw = mne.io.RawArray(eeg_sujeto_1.transpose(), info)
speech = dstims_para_sujeto_1
n_channels = len(raw.ch_names)

# Plot a sample of brain and stimulus activity
fig, ax = plt.subplots()
lns = ax.plot(scale(raw[:,3800:5000][0].T), color='k', alpha=.1)
ln1 = ax.plot(scale(speech[3800:5000]), color='r', lw=2)
ax.legend([lns[0], ln1[0]], ['EEG', 'Speech Envelope'], frameon=False)
ax.set(title="Sample activity", xlabel="Time (s)")
ax.set_xticklabels([0,3800,4000,4200,4400,4600,4800,5000])
mne.viz.tight_layout()

#%%
###############################################################################
# Create and fit a receptive field model
# --------------------------------------
#
# We will construct an encoding model to find the linear relationship between
# a time-delayed version of the speech envelope and the EEG signal. This allows
# us to make predictions about the response to new stimuli.

# Define the delays that we will use in the receptive field
tmin, tmax = -.53, 0.

# Initialize the model
rf = ReceptiveField(tmin, tmax, sfreq, feature_names=['envelope'],
                    estimator=1., scoring='corrcoef')
# We'll have (tmax - tmin) * sfreq delays
# and an extra 2 delays since we are inclusive on the beginning / end index
n_delays = int((tmax - tmin) * sfreq) + 2

n_splits = 5
cv = KFold(n_splits)

# Prepare model data (make time the first dimension)
speech = speech.reshape([speech.shape[0],1])
#speech = speech.T
Y, _ = raw[:]  # Outputs for the model
Y = Y.T

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
scores = np.zeros((n_splits, n_channels))
for ii, (train, test) in enumerate(cv.split(speech)):
    print('split %s / %s' % (ii + 1, n_splits))
    rf.fit(speech[train], Y[train])
    scores[ii] = rf.score(speech[test], Y[test])
    # coef_ is shape (n_outputs, n_features, n_delays). we only have 1 feature
    coefs[ii] = rf.coef_[:, 0, :]
times = rf.delays_ / float(rf.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_scores = scores.mean(axis=0)

# Plot mean prediction scores across all channels
fig, ax = plt.subplots()
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mean_scores)
ax.axhline(0, ls='--', color='r')
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
mne.viz.tight_layout()

#%%
###############################################################################
# Investigate model coefficients
# ==============================
# Finally, we will look at how the linear coefficients (sometimes
# referred to as beta values) are distributed across time delays as well as
# across the scalp. We will recreate `figure 1`_ and `figure 2`_ from
# :footcite:`CrosseEtAl2016`.

# Print mean coefficients across all time delays / channels (see Fig 1)
time_plot = -0.28  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8))
max_coef = mean_coefs.max()
ax.pcolormesh(times, ix_chs, mean_coefs, cmap='RdBu_r',
              vmin=-max_coef, vmax=max_coef, shading='gouraud')
ax.axvline(time_plot, ls='--', color='k', lw=2)
ax.set(xlabel='Delay (s)', ylabel='Channel', title="Mean Model\nCoefficients",
       xlim=times[[0, -1]], ylim=[len(ix_chs) - 1, 0],
       xticks=np.arange(tmin, tmax + .2, .2))
plt.setp(ax.get_xticklabels(), rotation=45)
mne.viz.tight_layout()

# Make a topographic map of coefficients for a given delay (see Fig 2C)
ix_plot = np.argmin(np.abs(time_plot - times))
fig, ax = plt.subplots()
mne.viz.plot_topomap(mean_coefs[:, ix_plot], pos=info, axes=ax, show=False,
                     vmin=-max_coef, vmax=max_coef)
ax.set(title="Topomap of model coefficients\nfor delay %s" % time_plot)
mne.viz.tight_layout()
    
#%%
###############################################################################
# Create and fit a stimulus reconstruction model
# ----------------------------------------------
#
# We will now demonstrate another use case for the for the
# :class:`mne.decoding.ReceptiveField` class as we try to predict the stimulus
# activity from the EEG data. This is known in the literature as a decoding, or
# stimulus reconstruction model :footcite:`CrosseEtAl2016`.
# A decoding model aims to find the
# relationship between the speech signal and a time-delayed version of the EEG.
# This can be useful as we exploit all of the available neural data in a
# multivariate context, compared to the encoding case which treats each M/EEG
# channel as an independent feature. Therefore, decoding models might provide a
# better quality of fit (at the expense of not controlling for stimulus
# covariance), especially for low SNR stimuli such as speech.

# We use the same lags as in :footcite:`CrosseEtAl2016`. Negative lags now
# index the relationship
# between the neural response and the speech envelope earlier in time, whereas
# positive lags would index how a unit change in the amplitude of the EEG would
# affect later stimulus activity (obviously this should have an amplitude of
# zero).
tmin, tmax = -.53, 0.

# Initialize the model. Here the features are the EEG data. We also specify
# ``patterns=True`` to compute inverse-transformed coefficients during model
# fitting (cf. next section and :footcite:`HaufeEtAl2014`).
# We'll use a ridge regression estimator with an alpha value similar to
# Crosse et al.
sr = ReceptiveField(tmin, tmax, sfreq, feature_names=raw.ch_names,
                    estimator=1e4, scoring='corrcoef', patterns=True)
# We'll have (tmax - tmin) * sfreq delays
# and an extra 2 delays since we are inclusive on the beginning / end index
n_delays = int((tmax - tmin) * sfreq) + 2

n_splits = 5
cv = KFold(n_splits)

# Iterate through splits, fit the model, and predict/test on held-out data
coefs = np.zeros((n_splits, n_channels, n_delays))
patterns = coefs.copy()
scores = np.zeros((n_splits,))
for ii, (train, test) in enumerate(cv.split(speech)):
    print('split %s / %s' % (ii + 1, n_splits))
    sr.fit(Y[train], speech[train])
    scores[ii] = sr.score(Y[test], speech[test])[0]
    # coef_ is shape (n_outputs, n_features, n_delays). We have 128 features
    coefs[ii] = sr.coef_[0, :, :]
    patterns[ii] = sr.patterns_[0, :, :]
times = sr.delays_ / float(sr.sfreq)

# Average scores and coefficients across CV splits
mean_coefs = coefs.mean(axis=0)
mean_patterns = patterns.mean(axis=0)
mean_scores = scores.mean(axis=0)
max_coef = np.abs(mean_coefs).max()
max_patterns = np.abs(mean_patterns).max()

#%%
###############################################################################
# Visualize stimulus reconstruction
# =================================
#
# To get a sense of our model performance, we can plot the actual and predicted
# stimulus envelopes side by side.
'''
y_pred = sr.predict(Y[test])
time = np.linspace(0, 2., y_pred[sr.valid_samples_][:].shape[0])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time, scale(speech[test][sr.valid_samples_][:]),
        color='grey', lw=2, ls='--')
ax.plot(time, scale(y_pred[sr.valid_samples_][:]), color='r', lw=2)
'''
y_pred = sr.predict(Y[test])
time = np.linspace(.75, 1.1, y_pred[sr.valid_samples_][1136:1666].shape[0])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time, scale(speech[test][sr.valid_samples_][1136:1666]),
        color='grey', lw=2, ls='--')
ax.plot(time, scale(y_pred[sr.valid_samples_][1136:1666]), color='r', lw=2)

ax.legend([lns[0], ln1[0]], ['Envelope', 'Reconstruction'], frameon=False)
ax.set(title="Stimulus reconstruction")
ax.set_xlabel('Time (s)')
mne.viz.tight_layout()

Envelope = scale(speech[test][sr.valid_samples_][:])
Reconstruction = scale(y_pred[sr.valid_samples_][:])
my_rho = np.corrcoef(Envelope.T, Reconstruction.T)
print(my_rho)  # 0.17205899


#%%
###############################################################################
# Investigate model coefficients
# ==============================
#
# Finally, we will look at how the decoding model coefficients are distributed
# across the scalp. We will attempt to recreate `figure 5`_ from
# :footcite:`CrosseEtAl2016`. The
# decoding model weights reflect the channels that contribute most toward
# reconstructing the stimulus signal, but are not directly interpretable in a
# neurophysiological sense. Here we also look at the coefficients obtained
# via an inversion procedure :footcite:`HaufeEtAl2014`, which have a more
# straightforward
# interpretation as their value (and sign) directly relates to the stimulus
# signal's strength (and effect direction).

time_plot = (-.22, -.20)  # To average between two timepoints.
ix_plot = np.arange(np.argmin(np.abs(time_plot[0] - times)),
                    np.argmin(np.abs(time_plot[1] - times)))
fig, ax = plt.subplots(1, 2)
mne.viz.plot_topomap(np.mean(mean_coefs[:, ix_plot], axis=1),
                     pos=info, axes=ax[0], show=False,
                     vmin=-max_coef, vmax=max_coef)
ax[0].set(title="Model coefficients\nbetween delays %s and %s"
          % (time_plot[0], time_plot[1]))

mne.viz.plot_topomap(np.mean(mean_patterns[:, ix_plot], axis=1),
                     pos=info, axes=ax[1],
                     show=False, vmin=-max_patterns, vmax=max_patterns)
ax[1].set(title="Inverse-transformed coefficients\nbetween delays %s and %s"
          % (time_plot[0], time_plot[1]))
mne.viz.tight_layout()

###############################################################################
# References
# ----------
#
# .. footbibliography::
