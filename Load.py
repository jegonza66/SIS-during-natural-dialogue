# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 01:30:36 2021

@author: joaco
"""
import numpy as np
import pandas as pd
import os
import pickle
import mne
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
from praatio import pitch_and_intensity
import Processing


class Trial_channel:

    def __init__(
            self, s=21, trial=1, channel=1, Band='All',
            sr=128, tmin=-0.53, tmax=-0.003, valores_faltantes_pitch=0,
            Causal_filter=True, drop_bads=False
    ):

        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']

        self.Band = Band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(self.Band)
        self.sr = sr
        self.sampleStep = 0.01
        self.audio_sr = 16000
        self.tmin, self.tmax = tmin, tmax
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int)
        self.valores_faltantes_pitch = valores_faltantes_pitch
        self.sex = sex_list[(s - 21) * 2 + channel - 1]
        self.Causal_filter = Causal_filter
        self.drop_bads = drop_bads
        self.bads = ['C7', 'B27', 'B26', 'B25', 'B14', 'B15', 'B10', 'B11', 'B9', 'B8', 'A26', 'A27', 'A25', 'A24',
                     'A13', 'A14', 'A12', 'A11', 'D32', 'D31', 'D24', 'D25', 'D23', 'D22', 'D8', 'D7', 'C30', 'C29']

        self.eeg_fname = "Datos/EEG/S" + str(s) + "/s" + str(s) + "-" + str(channel) + "-Trial" + str(
            trial) + "-Deci-Filter-Trim-ICA-Pruned.set"
        self.wav_fname = "Datos/wavs/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(channel) + ".wav"
        self.pitch_fname = "Datos/Pitch/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(channel) + ".txt"

    def f_eeg(self):
        eeg = mne.io.read_raw_eeglab(self.eeg_fname)
        eeg_freq = eeg.info.get("sfreq")
        eeg.load_data()
        # Hago un lowpass
        if self.Causal_filter:
            eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum')

        else:
            eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg)

        if self.drop_bads: eeg.drop_channels(self.bads)

        # Paso a array
        eeg = eeg.to_data_frame()
        eeg = np.array(eeg)[:, 1:129]  # paso a array y tomo tiro la primer columna de tiempos

        # Subsampleo
        # if self.subsamplear_promediando: eeg = Processing.subsamplear_promediando(eeg, int(eeg_freq/self.sr))
        eeg = Processing.subsamplear(eeg, int(eeg_freq / self.sr))

        return eeg

    def f_info(self):
        # Defino montage e info
        montage = mne.channels.make_standard_montage('biosemi128')
        channel_names = montage.ch_names
        info = mne.create_info(ch_names=channel_names[:], sfreq=self.sr, ch_types='eeg').set_montage(montage)

        if self.drop_bads:
            info['bads'] = self.bads

        return info

    def f_envelope(self):
        wav1 = wavfile.read(self.wav_fname)[1]
        wav1 = wav1.astype("float")

        ### envelope
        envelope = np.abs(sgn.hilbert(wav1))
        envelope = Processing.butter_filter(envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                            btype='lowpass', order=3, axis=0, ftype='NonCausal')
        window_size = 125
        stride = 125
        envelope = np.array([np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if
                             i + window_size <= len(envelope)])
        envelope = envelope.ravel().flatten()

        envelope = Processing.matriz_shifteada(envelope, self.delays)  # armo la matriz shifteada

        return np.array(envelope)

    def f_calculate_pitch(self):
        praatEXE = r"C:\Program Files\Praat\Praat.exe"
        output_folder = "C:/Users/joaco/Desktop/Joac/Facultad/Tesis/Código/Datos/Pitch"

        try:
            os.makedirs(output_folder)
        except:
            pass
        output_path = self.pitch_fname
        if self.sex == 'M':
            minPitch = 50
            maxPitch = 300
        if self.sex == 'F':
            minPitch = 75
            maxPitch = 500
        silenceThreshold = 0.01

        pitch_and_intensity.extractPI(os.path.abspath(self.wav_fname), os.path.abspath(output_path), praatEXE, minPitch,
                                      maxPitch, self.sampleStep, silenceThreshold)
        read_file = pd.read_csv(output_path)

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

        if not self.valores_faltantes_pitch:
            pitch[np.isnan(pitch)] = self.valores_faltantes_pitch
            pitch_der[np.isnan(pitch_der)] = self.valores_faltantes_pitch
        elif not np.isfinite(self.valores_faltantes_pitch):
            pitch[np.isnan(pitch)] = float(self.valores_faltantes_pitch)
            pitch_der[np.isnan(pitch_der)] = float(self.valores_faltantes_pitch)
        elif np.isfinite(self.valores_faltantes_pitch):
            pitch[np.isnan(pitch)] = np.float(self.valores_faltantes_pitch)
            pitch_der[np.isnan(pitch_der)] = np.float(self.valores_faltantes_pitch)
        else:
            print('Invalid missing value for pitch {}'.format(self.valores_faltantes_pitch) + '\nMust be finite.')

        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=float)
        pitch = Processing.subsamplear(pitch, 125)
        pitch = Processing.matriz_shifteada(pitch, self.delays)

        pitch_der = np.array(np.repeat(pitch_der, self.audio_sr * self.sampleStep), dtype=float)
        pitch_der = Processing.subsamplear(pitch_der, 125)
        pitch_der = Processing.matriz_shifteada(pitch_der, self.delays)

        return pitch, pitch_der

    def load_pitch(self):
        read_file = pd.read_csv(self.pitch_fname)

        time = np.array(read_file['time'])
        pitch = np.array(read_file['pitch'])
        intensity = np.array(read_file['intensity'])

        pitch[pitch == '--undefined--'] = np.nan
        pitch = np.array(pitch, dtype=float)

        pitch_der = []
        for i in range(len(pitch) - 1):
            try:
                diff = pitch[i + 1] - pitch[i]
                pitch_der.append(diff if np.abs(diff) < 20 else None)
            except:
                pitch_der.append(None)
        pitch_der.append(None)
        pitch_der = np.array(pitch_der, dtype=float)

        if not self.valores_faltantes_pitch:
            pitch[np.isnan(pitch)] = self.valores_faltantes_pitch
            pitch_der[np.isnan(pitch_der)] = self.valores_faltantes_pitch
        elif not np.isfinite(self.valores_faltantes_pitch):
            pitch[np.isnan(pitch)] = float(self.valores_faltantes_pitch)
            pitch_der[np.isnan(pitch_der)] = float(self.valores_faltantes_pitch)
        elif np.isfinite(self.valores_faltantes_pitch):
            pitch[np.isnan(pitch)] = np.float(self.valores_faltantes_pitch)
            pitch_der[np.isnan(pitch_der)] = np.float(self.valores_faltantes_pitch)
        else:
            print('Invalid missing value for pitch {}'.format(self.valores_faltantes_pitch) + '\nMust be finite.')

        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=float)
        pitch = Processing.subsamplear(pitch, 125)
        pitch = Processing.matriz_shifteada(pitch, self.delays)

        pitch_der = np.array(np.repeat(pitch_der, self.audio_sr * self.sampleStep), dtype=float)
        pitch_der = Processing.subsamplear(pitch_der, 125)
        pitch_der = Processing.matriz_shifteada(pitch_der, self.delays)

        return pitch, pitch_der

    def load_trial(self):
        channel = {}
        channel['eeg'] = self.f_eeg()
        channel['info'] = self.f_info()
        channel['envelope'] = self.f_envelope()
        channel['pitch'], channel['pitch_der'] = self.load_pitch()
        return channel


class Sesion_class:
    def __init__(self, sesion=21, Band='All', sr=128, tmin=-0.53, tmax=-0.003,
                 valores_faltantes_pitch=0, Causal_filter=True,
                 situacion='Escucha', Calculate_pitch=False,
                 procesed_data_path='saves/Preprocesed_Data/', Save_procesed_data=False
                 ):

        self.sesion = sesion
        self.Band = Band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(Band)
        self.sr = sr
        self.tmin, self.tmax = tmin, tmax
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int)
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(Band)
        self.valores_faltantes_pitch = valores_faltantes_pitch
        self.Causal_filter = Causal_filter
        self.situacion = situacion
        self.Calculate_pitch = Calculate_pitch
        self.procesed_data_path = procesed_data_path
        self.Save_procesed_data = Save_procesed_data

    def load_from_raw(self):
        ###### Armo estructura de datos de sujeto ######
        eeg_sujeto_1 = pd.DataFrame()
        envelope_para_sujeto_1 = pd.DataFrame()
        pitch_para_sujeto_1 = pd.DataFrame()
        pitch_der_para_sujeto_1 = pd.DataFrame()

        eeg_sujeto_2 = pd.DataFrame()
        envelope_para_sujeto_2 = pd.DataFrame()
        pitch_para_sujeto_2 = pd.DataFrame()
        pitch_der_para_sujeto_2 = pd.DataFrame()

        run = True
        trial = 1
        while run:
            try:
                Trial_channel_1 = Trial_channel(s=self.sesion, trial=trial, channel=1,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes_pitch=self.valores_faltantes_pitch,
                                                Causal_filter=self.Causal_filter).load_trial()
                Trial_channel_2 = Trial_channel(s=self.sesion, trial=trial, channel=2,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes_pitch=self.valores_faltantes_pitch,
                                                Causal_filter=self.Causal_filter).load_trial()

                ###### Cargo data ######        
                eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1 = \
                Trial_channel_1['eeg'], Trial_channel_2['envelope'], Trial_channel_2['pitch'], Trial_channel_2[
                    'pitch_der']
                if self.Calculate_pitch: pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1 = Trial_channel_2.f_calculate_pitch()
                momentos_sujeto_1_trial = Processing.labeling(self.sesion, trial, canal_hablante=2, sr=self.sr)

                eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2 = \
                Trial_channel_2['eeg'], Trial_channel_1['envelope'], Trial_channel_1['pitch'], Trial_channel_1[
                    'pitch_der']
                if self.Calculate_pitch: pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_1 = Trial_channel_1.f_calculate_pitch()
                momentos_sujeto_2_trial = Processing.labeling(self.sesion, trial, canal_hablante=1, sr=self.sr)

            except:
                run = False

            if run:
                ###### Igualar largos ######
                eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1, momentos_sujeto_1_trial = Processing.igualar_largos(
                    eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1,
                    pitch_der_trial_para_sujeto_1, momentos_sujeto_1_trial)
                eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2, momentos_sujeto_2_trial = Processing.igualar_largos(
                    eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2,
                    pitch_der_trial_para_sujeto_2, momentos_sujeto_2_trial)

                ###### Preprocesamiento ######
                eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1 = Processing.preproc(
                    momentos_sujeto_1_trial, self.delays, self.situacion, eeg_trial_sujeto_1,
                    envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1)
                eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2 = Processing.preproc(
                    momentos_sujeto_2_trial, self.delays, self.situacion, eeg_trial_sujeto_2,
                    envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2)

                # Convierto a DF
                eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1, pitch_der_trial_para_sujeto_1, eeg_trial_sujeto_2, envelope_trial_para_sujeto_2, pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2 = Processing.make_df(
                    eeg_trial_sujeto_1, envelope_trial_para_sujeto_1, pitch_trial_para_sujeto_1,
                    pitch_der_trial_para_sujeto_1, eeg_trial_sujeto_2, envelope_trial_para_sujeto_2,
                    pitch_trial_para_sujeto_2, pitch_der_trial_para_sujeto_2)

                ###### Adjunto a datos de sujeto ######
                if len(eeg_trial_sujeto_1):
                    eeg_sujeto_1 = eeg_sujeto_1.append(eeg_trial_sujeto_1)
                    envelope_para_sujeto_1 = envelope_para_sujeto_1.append(envelope_trial_para_sujeto_1)
                    pitch_para_sujeto_1 = pitch_para_sujeto_1.append(pitch_trial_para_sujeto_1)
                    pitch_der_para_sujeto_1 = pitch_der_para_sujeto_1.append(pitch_der_trial_para_sujeto_1)
                if len(eeg_trial_sujeto_2):
                    eeg_sujeto_2 = eeg_sujeto_2.append(eeg_trial_sujeto_2)
                    envelope_para_sujeto_2 = envelope_para_sujeto_2.append(envelope_trial_para_sujeto_2)
                    pitch_para_sujeto_2 = pitch_para_sujeto_2.append(pitch_trial_para_sujeto_2)
                    pitch_der_para_sujeto_2 = pitch_der_para_sujeto_2.append(pitch_der_trial_para_sujeto_2)

                trial += 1
        info = Trial_channel_1['info']

        # Convierto a array
        eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1, pitch_der_para_sujeto_1, eeg_sujeto_2, envelope_para_sujeto_2, pitch_para_sujeto_2, pitch_der_para_sujeto_2 = Processing.make_array(
            eeg_sujeto_1, envelope_para_sujeto_1, pitch_para_sujeto_1, pitch_der_para_sujeto_1, eeg_sujeto_2,
            envelope_para_sujeto_2, pitch_para_sujeto_2, pitch_der_para_sujeto_2)

        EEG_path = self.procesed_data_path + 'EEG/'
        if self.Causal_filter: EEG_path += 'Causal_'
        EEG_path += 'Sit_{}_Band_{}/'.format(self.situacion, self.Band)
        Envelope_path = self.procesed_data_path + 'Envelope/Sit_{}/'.format(self.situacion)
        Pitch_path = self.procesed_data_path + 'Pitch/Sit_{}_Faltantes_0/'.format(self.situacion)
        Pitch_der_path = self.procesed_data_path + 'Pitch_der/Sit_{}_Faltantes_0/'.format(self.situacion)
        for path in [EEG_path, Envelope_path, Pitch_path, Pitch_der_path]:
            try:
                os.makedirs(path)
            except:
                pass

        f = open(EEG_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([eeg_sujeto_1, eeg_sujeto_2], f)
        f.close()

        f = open(Envelope_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([envelope_para_sujeto_1, envelope_para_sujeto_2], f)
        f.close()

        f = open(Pitch_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([pitch_para_sujeto_1, pitch_para_sujeto_2], f)
        f.close()

        f = open(Pitch_der_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([pitch_der_para_sujeto_1, pitch_der_para_sujeto_2], f)
        f.close()

        f = open(self.procesed_data_path + 'EEG/info.pkl', 'wb')
        pickle.dump(info, f)
        f.close()

        Sujeto_1 = {'EEG': eeg_sujeto_1, 'Envelope': envelope_para_sujeto_2, 'Pitch': pitch_para_sujeto_2,
                    'Pitch_der': pitch_der_para_sujeto_2, 'info': info}
        Sujeto_2 = {'EEG': eeg_sujeto_2, 'Envelope': envelope_para_sujeto_1, 'Pitch': pitch_para_sujeto_1,
                    'Pitch_der': pitch_der_para_sujeto_1, 'info': info}
        Sesion = {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}

        return Sesion

    def load_procesed(self):

        eeg_path = self.procesed_data_path + 'EEG/'

        if self.Causal_filter: eeg_path += 'Causal_'
        eeg_path += 'Sit_{}_Band_{}/'.format(self.situacion, self.Band) + 'Sesion{}.pkl'.format(self.sesion)

        f = open(eeg_path, 'rb')
        eeg_sujeto_1, eeg_sujeto_2 = pickle.load(f)
        f.close()

        f = open(
            self.procesed_data_path + 'Envelope/Sit_{}/'.format(self.situacion) + 'Sesion{}.pkl'.format(self.sesion),
            'rb')
        envelope_para_sujeto_1, envelope_para_sujeto_2 = pickle.load(f)
        f.close()

        f = open(self.procesed_data_path + 'Pitch/Sit_{}_Faltantes_0/'.format(self.situacion) + 'Sesion{}.pkl'.format(
            self.sesion), 'rb')
        pitch_para_sujeto_1, pitch_para_sujeto_2 = pickle.load(f)
        f.close()

        f = open(
            self.procesed_data_path + 'Pitch_der/Sit_{}_Faltantes_0/'.format(self.situacion) + 'Sesion{}.pkl'.format(
                self.sesion), 'rb')
        pitch_der_para_sujeto_1, pitch_der_para_sujeto_2 = pickle.load(f)
        f.close()

        f = open(self.procesed_data_path + 'EEG/info.pkl', 'rb')
        info = pickle.load(f)
        f.close()

        if self.valores_faltantes_pitch == None:
            pitch_para_sujeto_1, pitch_para_sujeto_2 = pitch_para_sujeto_1[pitch_para_sujeto_1 != 0], \
                                                       pitch_para_sujeto_2[pitch_para_sujeto_2 != 0]  # saco 0s
        elif not np.isfinite(self.valores_faltantes_pitch) and self.valores_faltantes_pitch:
            pitch_para_sujeto_1[pitch_para_sujeto_1 == 0], pitch_para_sujeto_2[
                pitch_para_sujeto_2 == 0] = self.valores_faltantes_pitch, self.valores_faltantes_pitch  # cambio 0s
        elif np.isfinite(self.valores_faltantes_pitch) and self.valores_faltantes_pitch:
            pitch_para_sujeto_1[pitch_para_sujeto_1 == 0], pitch_para_sujeto_2[
                pitch_para_sujeto_2 == 0] = self.valores_faltantes_pitch, self.valores_faltantes_pitch  # cambio 0s

        Sujeto_1 = {'EEG': eeg_sujeto_1, 'Envelope': envelope_para_sujeto_2, 'Pitch': pitch_para_sujeto_2,
                    'Pitch_der': pitch_der_para_sujeto_2, 'info': info}
        Sujeto_2 = {'EEG': eeg_sujeto_2, 'Envelope': envelope_para_sujeto_1, 'Pitch': pitch_para_sujeto_1,
                    'Pitch_der': pitch_der_para_sujeto_1, 'info': info}
        Sesion = {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}

        return Sesion


def Load_Data(sesion, Band, sr, tmin, tmax, situacion, procesed_data_path, sujeto_total, Causal_filter=True,
              valores_faltantes_pitch=0, Calculate_pitch=False):
    Sesion_obj = Sesion_class(sesion=sesion, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                              valores_faltantes_pitch=valores_faltantes_pitch, Causal_filter=Causal_filter,
                              situacion=situacion, Calculate_pitch=Calculate_pitch,
                              procesed_data_path=procesed_data_path)

    # Intento cargar de preprocesados si existen
    try:
        Sesion = Sesion_obj.load_procesed()
        # Si falla cargo de raw y guardo
    except:
        Sesion = Sesion_obj.load_from_raw()

    Sujeto_1, Sujeto_2 = Sesion['Sujeto_1'], Sesion['Sujeto_2']

    return Sujeto_1, Sujeto_2


def Estimulos(stim, Sujeto_1, Sujeto_2):
    envelope_para_sujeto_1, pitch_para_sujeto_1, pitch_der_para_sujeto_1 = Sujeto_2['Envelope'], Sujeto_2['Pitch'], \
                                                                           Sujeto_2['Pitch_der']
    envelope_para_sujeto_2, pitch_para_sujeto_2, pitch_der_para_sujeto_2 = Sujeto_1['Envelope'], Sujeto_1['Pitch'], \
                                                                           Sujeto_1['Pitch_der']
    info = Sujeto_1['info']

    if stim == 'Envelope':
        dstims_para_sujeto_1, dstims_para_sujeto_2 = (envelope_para_sujeto_1,), (envelope_para_sujeto_2,)
    elif stim == 'Pitch':
        dstims_para_sujeto_1, dstims_para_sujeto_2 = (pitch_para_sujeto_1,), (pitch_para_sujeto_2,)
    elif stim == 'Pitch_der':
        dstims_para_sujeto_1, dstims_para_sujeto_2 = (pitch_der_para_sujeto_1,), (pitch_der_para_sujeto_2,)
    elif stim == 'Envelope_Pitch':
        dstims_para_sujeto_1, dstims_para_sujeto_2 = (envelope_para_sujeto_1, pitch_para_sujeto_1), (
        envelope_para_sujeto_2, pitch_para_sujeto_2)
    elif stim == 'Envelope_Pitch_Pitch_der':
        dstims_para_sujeto_1, dstims_para_sujeto_2 = (envelope_para_sujeto_1, pitch_para_sujeto_1,
                                                      pitch_der_para_sujeto_1), (
                                                     envelope_para_sujeto_2, pitch_para_sujeto_2,
                                                     pitch_der_para_sujeto_2)
    else:
        print('Invalid sitmulus: {}'.format(stim))

    return dstims_para_sujeto_1, dstims_para_sujeto_2, info


def rename_paths(Stims_preprocess, EEG_preprocess, stim, Band, tmin, tmax, *paths):
    returns = []
    for path in paths:
        path += 'Stim_{}_EEG_Band_{}/'.format(stim, Band)
        returns.append(path)
    return tuple(returns)
