import numpy as np
import pandas as pd
import os
import pickle
import mne
import librosa
import scipy.io.wavfile as wavfile
from scipy import signal as sgn
import platform
from praatio import pitch_and_intensity
import opensmile
import parselmouth
from parselmouth.praat import call
import Processing
import Funciones


class Trial_channel:

    def __init__(
            self, s=21, trial=1, channel=1, Band='All',
            sr=128, tmin=-0.6, tmax=-0.003, valores_faltantes=0,
            Causal_filter_EEG=True, Env_Filter=False
    ):

        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']

        self.Band = Band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(self.Band)
        self.sr = sr
        self.sampleStep = 0.01
        self.audio_sr = 16000
        self.tmin, self.tmax = tmin, tmax
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int)
        self.valores_faltantes = valores_faltantes
        self.sex = sex_list[(s - 21) * 2 + channel - 1]
        self.Causal_filter_EEG = Causal_filter_EEG
        self.Env_Filter = Env_Filter

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
        if self.Band:
            if self.Causal_filter_EEG:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg, phase='minimum')
            else:
                eeg = eeg.filter(l_freq=self.l_freq_eeg, h_freq=self.h_freq_eeg)

        # Paso a array
        eeg = eeg.to_data_frame()
        eeg = np.array(eeg)[:, 1:129]  # paso a array y tiro la primer columna de tiempo

        # Downsample
        eeg = Processing.subsamplear(eeg, int(eeg_freq / self.sr))

        return np.array(eeg)

    def f_info(self):
        # Defino montage e info
        montage = mne.channels.make_standard_montage('biosemi128')
        channel_names = montage.ch_names
        info = mne.create_info(ch_names=channel_names[:], sfreq=self.sr, ch_types='eeg').set_montage(montage)

        return info

    def f_envelope(self):
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        # Envelope
        envelope = np.abs(sgn.hilbert(wav))
        if self.Env_Filter == 'Causal':
            envelope = Processing.butter_filter(envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='Causal')
        elif self.Env_Filter == 'NonCausal':
            envelope = Processing.butter_filter(envelope, frecuencias=25, sampling_freq=self.audio_sr,
                                                btype='lowpass', order=3, axis=0, ftype='NonCausal')
        window_size = 125
        stride = 125
        envelope = np.array([np.mean(envelope[i:i + window_size]) for i in range(0, len(envelope), stride) if
                             i + window_size <= len(envelope)])
        envelope = envelope.ravel().flatten()
        envelope = Processing.matriz_shifteada(envelope, self.delays)  # armo la matriz shifteada
        self.envelope = envelope

        return np.array(envelope)

    def f_calculate_pitch(self):
        if platform.system() == 'Linux':
            praatEXE = 'Praat/praat'
            output_folder = 'Datos/Pitch'
        else:
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

    def load_pitch(self):
        read_file = pd.read_csv(self.pitch_fname)

        # time = np.array(read_file['time'])
        pitch = np.array(read_file['pitch'])
        # intensity = np.array(read_file['intensity'])

        pitch[pitch == '--undefined--'] = np.nan
        pitch = np.array(pitch, dtype=np.float32)

        pitch_der = []
        for i in range(len(pitch) - 1):
            try:
                diff = pitch[i + 1] - pitch[i]
                pitch_der.append(diff)
            except:
                pitch_der.append(None)
        pitch_der.append(None)
        pitch_der = np.array(pitch_der, dtype=np.float32)

        if self.valores_faltantes == None:
            pitch = pitch[~np.isnan(pitch)]
            pitch_der = pitch_der[~np.isnan(pitch_der)]
        elif np.isfinite(self.valores_faltantes):
            pitch[np.isnan(pitch)] = float(self.valores_faltantes)
            pitch_der[np.isnan(pitch_der)] = float(self.valores_faltantes)
        else:
            print('Invalid missing value for pitch {}'.format(self.valores_faltantes) + '\nMust be finite.')

        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=np.float32)
        pitch = Processing.subsamplear(pitch, 125)
        pitch = Processing.matriz_shifteada(pitch, self.delays)

        pitch_der = np.array(np.repeat(pitch_der, self.audio_sr * self.sampleStep), dtype=np.float32)
        pitch_der = Processing.subsamplear(pitch_der, 125)
        pitch_der = Processing.matriz_shifteada(pitch_der, self.delays)

        return np.array(pitch), np.array(pitch_der)

    def f_jitter_shimmer(self):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

        y = smile.process_file(self.wav_fname)
        y.index = y.index.droplevel(0)
        y.index = y.index.map(lambda x: x[0].total_seconds())

        jitter = y['jitterLocal_sma3nz']
        shimmer = y['shimmerLocaldB_sma3nz']

        mcm = Funciones.minimo_comun_multiplo(len(jitter), len(self.envelope))
        jitter = np.repeat(jitter, mcm / len(jitter))
        jitter = Processing.subsamplear(jitter, mcm / len(self.envelope))
        jitter = Processing.matriz_shifteada(jitter, self.delays)

        mcm = Funciones.minimo_comun_multiplo(len(shimmer), len(self.envelope))
        shimmer = np.repeat(shimmer, mcm / len(shimmer))
        shimmer = Processing.subsamplear(shimmer, mcm / len(self.envelope))
        shimmer = Processing.matriz_shifteada(shimmer, self.delays)

        return jitter, shimmer

    # def f_cssp(self):

        # snd = parselmouth.Sound(self.wav_fname)
        # data = []
        # frame_length = 0.2
        # hop_length = 1/128
        # t1s = np.arange(0, snd.duration - frame_length, hop_length)
        # times = zip(t1s, t1s + frame_length)
        #
        # for t1, t2 in times:
        #     powercepstrogram = call(snd.extract_part(t1, t2), "To PowerCepstrogram", 60, 0.0020001, 5000, 50)
        #     cpps = call(powercepstrogram, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0,
        #                 "Exponential decay", "Robust")
        #     data.append(cpps)
        #
        # cssp = np.array(np.repeat(data, self.audio_sr * self.sampleStep))
        # cssp = Processing.subsamplear(cssp, 125)
        # cssp = Processing.matriz_shifteada(cssp, self.delays)

        # return cssp

    def f_spectrogram(self):
        wav = wavfile.read(self.wav_fname)[1]
        wav = wav.astype("float")

        n_fft = 125
        hop_length = 125
        n_mels = 16

        S = librosa.feature.melspectrogram(wav, sr=self.audio_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Shifted matrix row by row
        spec_shift = Processing.matriz_shifteada(S_DB[0], self.delays)
        for i in np.arange(1, len(S_DB)):
            spec_shift = np.hstack((spec_shift, Processing.matriz_shifteada(S_DB[i], self.delays)))

        return np.array(spec_shift)

    def load_trial(self):
        channel = {}
        channel['eeg'] = self.f_eeg()
        channel['info'] = self.f_info()
        channel['envelope'] = self.f_envelope()
        channel['pitch'], channel['pitch_der'] = self.load_pitch()
        channel['spectrogram'] = self.f_spectrogram()
        channel['jitter'], channel['shimmer'] = self.f_jitter_shimmer()
        # channel['cssp'] = self.f_cssp()
        return channel


class Sesion_class:
    def __init__(self, sesion=21, stim='Envelope', Band='All', sr=128, tmin=-0.6, tmax=-0.003,
                 valores_faltantes=0, Causal_filter_EEG=True, Env_Filter=False,
                 situacion='Escucha', Calculate_pitch=False,
                 procesed_data_path='saves/Preprocesed_Data/tmin{}_tmax{}/'.format(-0.6, -0.003)
                 ):

        self.sesion = sesion
        self.stim = stim
        self.Band = Band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(Band)
        self.sr = sr
        self.tmin, self.tmax = tmin, tmax
        self.delays = - np.arange(np.floor(tmin * self.sr), np.ceil(tmax * self.sr), dtype=int)
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(Band)
        self.valores_faltantes = valores_faltantes
        self.Causal_filter_EEG = Causal_filter_EEG
        self.Env_Filter = Env_Filter
        self.situacion = situacion
        self.Calculate_pitch = Calculate_pitch
        self.procesed_data_path = procesed_data_path

    def load_from_raw(self):
        # Armo estructura de datos de sujeto
        Sujeto_1 = {'EEG': pd.DataFrame(), 'Envelope': pd.DataFrame(), 'Pitch': pd.DataFrame(),
                    'Pitch_der': pd.DataFrame(), 'Spectrogram': pd.DataFrame(),
                    'Jitter': pd.DataFrame(), 'Shimmer': pd.DataFrame()}#, 'Cssp': pd.DataFrame()}

        Sujeto_2 = {'EEG': pd.DataFrame(), 'Envelope': pd.DataFrame(), 'Pitch': pd.DataFrame(),
                    'Pitch_der': pd.DataFrame(), 'Spectrogram': pd.DataFrame(),
                    'Jitter': pd.DataFrame(), 'Shimmer': pd.DataFrame()}

        run = True
        trial = 1
        while run:
            try:
                if self.Calculate_pitch:
                    Trial_channel(s=self.sesion, trial=trial, channel=1,
                                  Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                  valores_faltantes=self.valores_faltantes,
                                  Causal_filter_EEG=self.Causal_filter_EEG,
                                  Env_Filter=self.Env_Filter).f_calculate_pitch()
                    Trial_channel(s=self.sesion, trial=trial, channel=2,
                                  Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                  valores_faltantes=self.valores_faltantes,
                                  Causal_filter_EEG=self.Causal_filter_EEG,
                                  Env_Filter=self.Env_Filter).f_calculate_pitch()

                Trial_channel_1 = Trial_channel(s=self.sesion, trial=trial, channel=1,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes=self.valores_faltantes,
                                                Causal_filter_EEG=self.Causal_filter_EEG,
                                                Env_Filter=self.Env_Filter).load_trial()

                Trial_channel_2 = Trial_channel(s=self.sesion, trial=trial, channel=2,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes=self.valores_faltantes,
                                                Causal_filter_EEG=self.Causal_filter_EEG,
                                                Env_Filter=self.Env_Filter).load_trial()

                # Load data to dictionary taking stimuli from speaker and eeg from listener
                Trial_sujeto_1 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()}
                Trial_sujeto_1['eeg'] = Trial_channel_1['eeg']

                Trial_sujeto_2 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}
                Trial_sujeto_2['eeg'] = Trial_channel_2['eeg']

                # Instant labeling of current speaker
                momentos_sujeto_1_trial = Processing.labeling(self.sesion, trial, canal_hablante=2, sr=self.sr)
                momentos_sujeto_2_trial = Processing.labeling(self.sesion, trial, canal_hablante=1, sr=self.sr)

            except:
                run = False

            if run:
                # Match lengths of variables
                momentos_sujeto_1_trial = Funciones.igualar_largos_dict(Trial_sujeto_1, momentos_sujeto_1_trial)
                momentos_sujeto_2_trial = Funciones.igualar_largos_dict(Trial_sujeto_2, momentos_sujeto_2_trial)

                # Preprocesamiento
                Processing.preproc_dict(momentos_escucha=momentos_sujeto_1_trial, delays=self.delays,
                                        situacion=self.situacion, dict=Trial_sujeto_1)
                Processing.preproc_dict(momentos_escucha=momentos_sujeto_2_trial, delays=self.delays,
                                        situacion=self.situacion, dict=Trial_sujeto_2)

                # Convierto a DF
                Funciones.make_df_dict(Trial_sujeto_1)
                Funciones.make_df_dict(Trial_sujeto_2)

                # Adjunto a datos de sujeto
                if len(Trial_sujeto_1['eeg']):
                    Sujeto_1['EEG'] = Sujeto_1['EEG'].append(Trial_sujeto_1['eeg'])
                    Sujeto_1['Envelope'] = Sujeto_1['Envelope'].append((Trial_sujeto_1['envelope']))
                    Sujeto_1['Pitch'] = Sujeto_1['Pitch'].append((Trial_sujeto_1['pitch']))
                    Sujeto_1['Spectrogram'] = Sujeto_1['Spectrogram'].append((Trial_sujeto_1['spectrogram']))
                    Sujeto_1['Jitter'] = Sujeto_1['Jitter'].append((Trial_sujeto_1['jitter']))
                    Sujeto_1['Shimmer'] = Sujeto_1['Shimmer'].append((Trial_sujeto_1['shimmer']))
                    # Sujeto_1['Cssp'] = Sujeto_1['Cssp'].append((Trial_sujeto_1['cssp']))

                if len(Trial_sujeto_2['eeg']):
                    Sujeto_2['EEG'] = Sujeto_2['EEG'].append(Trial_sujeto_2['eeg'])
                    Sujeto_2['Envelope'] = Sujeto_2['Envelope'].append((Trial_sujeto_2['envelope']))
                    Sujeto_2['Pitch'] = Sujeto_2['Pitch'].append((Trial_sujeto_2['pitch']))
                    Sujeto_2['Spectrogram'] = Sujeto_2['Spectrogram'].append((Trial_sujeto_2['spectrogram']))
                    Sujeto_2['Jitter'] = Sujeto_2['Jitter'].append((Trial_sujeto_2['jitter']))
                    Sujeto_2['Shimmer'] = Sujeto_2['Shimmer'].append((Trial_sujeto_2['shimmer']))
                    # Sujeto_2['Cssp'] = Sujeto_2['Cssp'].append((Trial_sujeto_2['cssp']))

                trial += 1
        info = Trial_channel_1['info']

        # Convierto a array
        Funciones.make_array_dict(Sujeto_1)
        Funciones.make_array_dict(Sujeto_2)

        # Define Save Paths
        EEG_path = self.procesed_data_path + 'EEG/'
        if self.Band and self.Causal_filter_EEG: EEG_path += 'Causal_'
        EEG_path += 'Sit_{}_Band_{}/'.format(self.situacion, self.Band)

        Envelope_path = self.procesed_data_path + 'Envelope/Sit_{}/'.format(self.situacion)

        Pitch_path = self.procesed_data_path + 'Pitch/Sit_{}_Faltantes_{}/'.format(self.situacion,
                                                                                   self.valores_faltantes)
        Pitch_der_path = self.procesed_data_path + 'Pitch_der/Sit_{}_Faltantes_{}/'.format(self.situacion,
                                                                                           self.valores_faltantes)
        Spectrogram_path = self.procesed_data_path + 'Spectrogram/Sit_{}/'.format(self.situacion)

        Jitter_path = self.procesed_data_path + 'Jitter/Sit_{}_Faltantes_{}/'.format(self.situacion,
                                                                                   self.valores_faltantes)
        Shimmer_path = self.procesed_data_path + 'Shimmer/Sit_{}_Faltantes_{}/'.format(self.situacion,
                                                                                   self.valores_faltantes)
        # Cssp_path = self.procesed_data_path + 'Cssp/Sit_{}/'.format(self.situacion)

        for path in [EEG_path, Envelope_path, Pitch_path, Pitch_der_path, Spectrogram_path, Jitter_path, Shimmer_path]:
            try:
                os.makedirs(path)
            except:
                pass

        # Save Preprocesed Data
        f = open(EEG_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['EEG'], Sujeto_2['EEG']], f)
        f.close()

        f = open(Envelope_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['Envelope'], Sujeto_2['Envelope']], f)
        f.close()

        f = open(Pitch_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['Pitch'], Sujeto_2['Pitch']], f)
        f.close()

        f = open(Spectrogram_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['Spectrogram'], Sujeto_2['Spectrogram']], f)
        f.close()

        f = open(Jitter_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['Jitter'], Sujeto_2['Jitter']], f)
        f.close()

        f = open(Shimmer_path + 'Sesion{}.pkl'.format(self.sesion), 'wb')
        pickle.dump([Sujeto_1['Shimmer'], Sujeto_2['Shimmer']], f)
        f.close()

        f = open(self.procesed_data_path + 'EEG/info.pkl', 'wb')
        pickle.dump(info, f)
        f.close()

        # Redefine Subjects dictionaries to return only used stimuli
        Sujeto_1_return = {key: Sujeto_1[key] for key in self.stim.split('_')}
        Sujeto_2_return = {key: Sujeto_2[key] for key in self.stim.split('_')}
        Sujeto_1_return['info'] = info
        Sujeto_2_return['info'] = info
        Sujeto_1_return['EEG'] = Sujeto_1['EEG']
        Sujeto_2_return['EEG'] = Sujeto_2['EEG']

        Sesion = {'Sujeto_1': Sujeto_1_return, 'Sujeto_2': Sujeto_2_return}

        return Sesion


    def load_procesed(self):

        EEG_path = self.procesed_data_path + 'EEG/'
        if self.Band and self.Causal_filter_EEG: EEG_path += 'Causal_'
        EEG_path += 'Sit_{}_Band_{}/Sesion{}.pkl'.format(self.situacion, self.Band, self.sesion)

        f = open(EEG_path, 'rb')
        eeg_sujeto_1, eeg_sujeto_2 = pickle.load(f)
        f.close()

        f = open(self.procesed_data_path + 'EEG/info.pkl', 'rb')
        info = pickle.load(f)
        f.close()

        Sujeto_1 = {'EEG': eeg_sujeto_1, 'info': info}
        Sujeto_2 = {'EEG': eeg_sujeto_2, 'info': info}

        for stimuli in self.stim.split('_'):
            if stimuli == 'Envelope':
                Envelope_path = self.procesed_data_path + 'Envelope/'
                if self.Env_Filter: Envelope_path += self.Env_Filter + '_'
                Envelope_path += 'Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion)

                f = open(Envelope_path, 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            if stimuli == 'Pitch':
                f = open(self.procesed_data_path + 'Pitch/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

                if self.valores_faltantes == None:
                    stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
                                                               stimuli_para_sujeto_2[stimuli_para_sujeto_2 != 0]  # saco 0s
                elif self.valores_faltantes:
                    stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
                        stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

            if stimuli == 'Pitch_der':
                f = open(self.procesed_data_path + 'Pitch_der/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

                if self.valores_faltantes == None:
                    stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
                                                                   stimuli_para_sujeto_2[stimuli_para_sujeto_2 != 0]  # saco 0s
                elif self.valores_faltantes:
                    stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
                        stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

            if stimuli == 'Spectrogram':
                f = open(self.procesed_data_path + 'Spectrogram/Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            if stimuli == 'Jitter':
                f = open(self.procesed_data_path + 'Jitter/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

                if self.valores_faltantes == None:
                    stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
                                                               stimuli_para_sujeto_2[stimuli_para_sujeto_2 != 0]  # saco 0s
                elif self.valores_faltantes:
                    stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
                        stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

            if stimuli == 'Shimmer':
                f = open(self.procesed_data_path + 'Shimmer/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

                if self.valores_faltantes == None:
                    stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
                                                               stimuli_para_sujeto_2[stimuli_para_sujeto_2 != 0]  # saco 0s
                elif self.valores_faltantes:
                    stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
                        stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

            Sujeto_1[stimuli] = stimuli_para_sujeto_1
            Sujeto_2[stimuli] = stimuli_para_sujeto_2

        Sesion = {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}

        return Sesion


def Load_Data(sesion, stim, Band, sr, tmin, tmax, procesed_data_path, situacion='Escucha', Causal_filter_EEG=True,
              Env_Filter=False, valores_faltantes=0, Calculate_pitch=False):

    possible_stims = ['Envelope', 'Pitch', 'Spectrogram', 'Shimmer']

    if all(stimulus in possible_stims for stimulus in stim.split('_')):

        Sesion_obj = Sesion_class(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                  valores_faltantes=valores_faltantes, Causal_filter_EEG=Causal_filter_EEG,
                                  Env_Filter=Env_Filter, situacion=situacion, Calculate_pitch=Calculate_pitch,
                                  procesed_data_path=procesed_data_path)

        # Intento cargar de preprocesados si existen
        try:
            Sesion = Sesion_obj.load_procesed()
            # Si falla cargo de raw y guardo
        except:
            Sesion = Sesion_obj.load_from_raw()

        Sujeto_1, Sujeto_2 = Sesion['Sujeto_1'], Sesion['Sujeto_2']

        return Sujeto_1, Sujeto_2

    else:
        print('Invalid stimulus. Please enter valid stimulus. Possible stimulus are:')
        for i in range(len(possible_stims)):
            print(possible_stims[i])


def Estimulos(stim, Sujeto_1, Sujeto_2):
    dfinal_para_sujeto_1 = []
    dfinal_para_sujeto_2 = []

    for stimuli in stim.split('_'):
        dfinal_para_sujeto_1.append(Sujeto_1[stimuli])
        dfinal_para_sujeto_2.append(Sujeto_2[stimuli])

    return dfinal_para_sujeto_1, dfinal_para_sujeto_2