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
import textgrids
import setup
exp_info = setup.exp_info()


class Trial_channel:

    def __init__(
            self, s=21, trial=1, channel=1, Band='All',
            sr=128, tmin=-0.6, tmax=-0.003, valores_faltantes=0,
            Causal_filter_EEG=True, Env_Filter=False, SilenceThreshold=0.03
    ):

        sex_list = ['M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'M']

        self.Band = Band
        self.l_freq_eeg, self.h_freq_eeg = Processing.band_freq(self.Band)
        self.sr = sr
        self.sampleStep = 0.01
        self.SilenceThreshold = SilenceThreshold
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
        self.pitch_fname = "Datos/Pitch_threshold_{}/S".format(SilenceThreshold) + str(s) + "/s" + str(s) + ".objects." \
                           + "{:02d}".format(trial) + ".channel" + str(channel) + ".txt"
        self.phn_fname = "Datos/phonemes/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(channel) + ".aligned_fa.TextGrid"
        self.phn_fname_manual = "Datos/phonemes/S" + str(s) + "/manual/s" + str(s) + "_objects_" + "{:02d}".format(
            trial) + "_channel" + str(channel) + "_aligned_faTAMARA.TextGrid"
        self.phrases_fname = "Datos/phrases/S" + str(s) + "/s" + str(s) + ".objects." + "{:02d}".format(
            trial) + ".channel" + str(
            channel) + ".phrases"

    def f_phonemes(self, envelope):

        # Get trial total length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname)
        # Get phonemes
        phonemes_grid = grid['transcription : phones']
        # Extend first silence time to trial start time
        phonemes_grid[0].xmin = 0.

        # Parse
        labels = []
        times = []
        samples = []

        for ph in phonemes_grid:
            label = ph.text.transcode()
            # Rename silences
            if label == 'sil' or label == 'sp':
                label = ""
            labels.append(label)
            times.append((ph.xmin, ph.xmax))
            samples.append(np.round((ph.xmax - ph.xmin) * self.sr).astype("int"))

        # Extend last silence to trial end time
        labels.append("")
        times.append((ph.xmin, trial_tmax))
        samples.append(np.round((trial_tmax - ph.xmax) * self.sr).astype("int"))

        # Get unique phonemes
        # labels_set = set(labels)
        # unique_labels = (list(labels_set))

        # If use envelope amplitude to make continuous stimuli
        diferencia = np.sum(samples) - len(envelope)
        if diferencia > 0:
            for i in [-i-1 for i in range(len(samples))]:
                if diferencia > samples[i]:
                    diferencia -= samples[i]
                    samples[i] = 0
                # Cuando samples es mayor que diferencia, le resto la diferencia a las samples
                else:
                    samples[i] -= diferencia
                    break
        elif diferencia < 0:
            samples[-1] -= diferencia

        # Make empty df of phonemes
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels)

        # Get samples number instead of time
        phonemes_tgrid = np.repeat(labels, samples)

        for i, phoneme in enumerate(phonemes_tgrid):
            phonemes_df.loc[i, phoneme] = envelope[i]
            # df.loc[i, phoneme] = 1

        return phonemes_df

    def f_Phonemes_manual(self, envelope):

        # Get trial total length
        phrases = pd.read_table(self.phrases_fname, header=None, sep="\t")
        trial_tmax = phrases[1].iloc[-1]

        # Load transcription
        grid = textgrids.TextGrid(self.phn_fname_manual)
        # Get phonemes
        phonemes_grid = grid['transcription : phones']
        # Extend first silence time to trial start time
        phonemes_grid[0].xmin = 0.

        # Parse
        labels = []
        times = []
        samples = []

        for ph in phonemes_grid:
            label = ph.text.transcode()
            label = label.replace(' ', '')
            label = label.replace('º', '')
            label = label.replace('-', '')
            # Rename silences
            if label == 'sil' or label == 'sp' or label == 'sile' or label == 'silsil' \
                    or label == 'SP' or label == 's¡p' or label == 'sils':
                label = ""
            labels.append(label)
            times.append((ph.xmin, ph.xmax))
            samples.append(np.round((ph.xmax - ph.xmin) * self.sr).astype("int"))

        # Extend last silence to trial end time
        labels.append("")
        times.append((ph.xmin, trial_tmax))
        samples.append(np.round((trial_tmax - ph.xmax) * self.sr).astype("int"))

        # If use envelope amplitude to make continuous stimuli
        diferencia = np.sum(samples) - len(envelope)
        if diferencia > 0:
            for i in [-i-1 for i in range(len(samples))]:
                if diferencia > samples[i]:
                    diferencia -= samples[i]
                    samples[i] = 0
                # Cuando samples es mayor que diferencia, le resto la diferencia a las samples
                else:
                    samples[i] -= diferencia
                    break
        elif diferencia < 0:
            samples[-1] -= diferencia

        # Make empty df of phonemes
        phonemes_df = pd.DataFrame(0, index=np.arange(np.sum(samples)), columns=exp_info.ph_labels_man)

        # Get samples number instead of time
        phonemes_tgrid = np.repeat(labels, samples)

        for i, phoneme in enumerate(phonemes_tgrid):
            phonemes_df.loc[i, phoneme] = envelope[i]
            # df.loc[i, phoneme] = 1

        return phonemes_df


    def f_eeg(self):
        eeg = mne.io.read_raw_eeglab(self.eeg_fname)
        eeg_freq = eeg.info.get("sfreq")
        eeg.load_data()

        # Independent sources
        # eeg = mne.preprocessing.compute_current_source_density(eeg)

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
        # envelope_mat = Processing.matriz_shifteada(envelope, self.delays)  # armo la matriz shifteada
        self.envelope = envelope

        return np.array(envelope)

    def f_calculate_pitch(self):
        if platform.system() == 'Linux':
            praatEXE = 'Praat/praat'
            output_folder = 'Datos/Pitch_threshold_{}'.format(self.SilenceThreshold)
        else:
            praatEXE = r"C:\Program Files\Praat\Praat.exe"
            output_folder = "C:/Users/joaco/Desktop/Joac/Facultad/Tesis/Código/Datos/Pitch_threshold_{}".format(
                self.SilenceThreshold)
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
        pitch_and_intensity.extractPI(os.path.abspath(self.wav_fname), os.path.abspath(output_path), praatEXE, minPitch,
                                      maxPitch, self.sampleStep, self.SilenceThreshold)

    def load_pitch(self):
        read_file = pd.read_csv(self.pitch_fname)

        # time = np.array(read_file['time'])
        pitch = np.array(read_file['pitch'])
        # intensity = np.array(read_file['intensity'])

        pitch[pitch == '--undefined--'] = np.nan
        pitch = np.array(pitch, dtype=np.float32)

        if self.valores_faltantes == None:
            pitch = pitch[~np.isnan(pitch)]
        elif np.isfinite(self.valores_faltantes):
            pitch[np.isnan(pitch)] = float(self.valores_faltantes)
        else:
            print('Invalid missing value for pitch {}'.format(self.valores_faltantes) + '\nMust be finite.')

        pitch = np.array(np.repeat(pitch, self.audio_sr * self.sampleStep), dtype=np.float32)
        pitch = Processing.subsamplear(pitch, 125)
        # pitch = Processing.matriz_shifteada(pitch, self.delays)

        # pitch_der = Processing.matriz_shifteada(pitch_der, self.delays)

        return np.array(pitch)

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
        # jitter = Processing.matriz_shifteada(jitter, self.delays)

        mcm = Funciones.minimo_comun_multiplo(len(shimmer), len(self.envelope))
        shimmer = np.repeat(shimmer, mcm / len(shimmer))
        shimmer = Processing.subsamplear(shimmer, mcm / len(self.envelope))
        # shimmer = Processing.matriz_shifteada(shimmer, self.delays)

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
        S_DB = S_DB.transpose()

        # # Shifted matrix row by row
        # spec_shift = Processing.matriz_shifteada(S_DB[0], self.delays)
        # for i in np.arange(1, len(S_DB)):
        #     spec_shift = np.hstack((spec_shift, Processing.matriz_shifteada(S_DB[i], self.delays)))

        return np.array(S_DB)

    def load_trial(self, stims):
        channel = {}
        channel['EEG'] = self.f_eeg()
        channel['info'] = self.f_info()

        if 'Envelope' in stims:
            channel['Envelope'] = self.f_envelope()
        if 'Pitch' in stims:
            channel['Pitch'] = self.load_pitch()
        if 'Spectrogram' in stims:
            channel['Spectrogram'] = self.f_spectrogram()
        if 'Phonemes' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes'] = self.f_phonemes(envelope=self.envelope)
        if 'Phonemes-manual' in stims:
            channel['Envelope'] = self.f_envelope()
            channel['Phonemes-manual'] = self.f_Phonemes_manual(envelope=self.envelope)

        return channel


class Sesion_class:
    def __init__(self, sesion=21, stim='Envelope', Band='All', sr=128, tmin=-0.6, tmax=-0.003,
                 valores_faltantes=0, Causal_filter_EEG=True, Env_Filter=False,
                 situacion='Escucha', Calculate_pitch=False, SilenceThreshold=0.03,
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
        self.SilenceThreshold = SilenceThreshold
        self.procesed_data_path = procesed_data_path
        self.samples_info_path = self.procesed_data_path + 'samples_info/Sit_{}/'.format(self.situacion)
        self.phn_path = "Datos/phonemes/S" + str(self.sesion) + "/"


    def load_from_raw(self):

        # Armo estructura de datos de sujeto
        Sujeto_1 = {'EEG': pd.DataFrame()}
        Sujeto_2 = {'EEG': pd.DataFrame()}

        for stimuli in self.stim.split('_'):
            Sujeto_1[stimuli] = pd.DataFrame()
            Sujeto_2[stimuli] = pd.DataFrame()

        try:
            f = open(self.samples_info_path + f'samples_info_{self.sesion}.pkl', 'rb')
            samples_info = pickle.load(f)
            f.close()
            loaded_samples_info = True
        except:
            loaded_samples_info = False
            samples_info = {'keep_indexes1': [], 'keep_indexes2': [], 'trial_lengths1': [0], 'trial_lengths2': [0]}

        trials = list(set([int(fname.split('.')[2]) for fname in os.listdir(self.phn_path) if os.path.isfile(self.phn_path + f'/{fname}')]))

        for trial in trials:
            try:
                if self.Calculate_pitch:
                    Trial_channel(s=self.sesion, trial=trial, channel=1,
                                  Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                  valores_faltantes=self.valores_faltantes,
                                  Causal_filter_EEG=self.Causal_filter_EEG,
                                  Env_Filter=self.Env_Filter,
                                  SilenceThreshold=self.SilenceThreshold).f_calculate_pitch()
                    Trial_channel(s=self.sesion, trial=trial, channel=2,
                                  Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                  valores_faltantes=self.valores_faltantes,
                                  Causal_filter_EEG=self.Causal_filter_EEG,
                                  Env_Filter=self.Env_Filter,
                                  SilenceThreshold=self.SilenceThreshold).f_calculate_pitch()

                Trial_channel_1 = Trial_channel(s=self.sesion, trial=trial, channel=1,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes=self.valores_faltantes,
                                                Causal_filter_EEG=self.Causal_filter_EEG,
                                                Env_Filter=self.Env_Filter,
                                                SilenceThreshold=self.SilenceThreshold).load_trial(self.stim.split('_'))

                Trial_channel_2 = Trial_channel(s=self.sesion, trial=trial, channel=2,
                                                Band=self.Band, sr=self.sr, tmin=self.tmin, tmax=self.tmax,
                                                valores_faltantes=self.valores_faltantes,
                                                Causal_filter_EEG=self.Causal_filter_EEG,
                                                Env_Filter=self.Env_Filter,
                                                SilenceThreshold=self.SilenceThreshold).load_trial(self.stim.split('_'))

                if self.situacion == 'Habla_Propia' or self.situacion == 'Ambos_Habla':
                    # Load data to dictionary taking stimuli and eeg from speaker
                    Trial_sujeto_1 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}
                    Trial_sujeto_2 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()}

                else:
                    # Load data to dictionary taking stimuli from speaker and eeg from listener
                    Trial_sujeto_1 = {key: Trial_channel_2[key] for key in Trial_channel_2.keys()}
                    Trial_sujeto_1['EEG'] = Trial_channel_1['EEG']

                    Trial_sujeto_2 = {key: Trial_channel_1[key] for key in Trial_channel_1.keys()}
                    Trial_sujeto_2['EEG'] = Trial_channel_2['EEG']

                # Instant labeling of current speaker
                momentos_sujeto_1_trial = Processing.labeling(self.sesion, trial, canal_hablante=2, sr=self.sr)
                momentos_sujeto_2_trial = Processing.labeling(self.sesion, trial, canal_hablante=1, sr=self.sr)

                if loaded_samples_info:
                    _ = Funciones.igualar_largos_dict_sample_data(Trial_sujeto_1, momentos_sujeto_1_trial,
                                                                  minimo_largo=samples_info['trial_lengths1'][trial])
                    _ = Funciones.igualar_largos_dict_sample_data(Trial_sujeto_2, momentos_sujeto_2_trial,
                                                                  minimo_largo=samples_info['trial_lengths2'][trial])

                else:
                    # Match lengths of variables
                    momentos_sujeto_1_trial, minimo_largo1 = Funciones.igualar_largos_dict2(Trial_sujeto_1, momentos_sujeto_1_trial)
                    momentos_sujeto_2_trial, minimo_largo2 = Funciones.igualar_largos_dict2(Trial_sujeto_2, momentos_sujeto_2_trial)

                    samples_info['trial_lengths1'].append(minimo_largo1)
                    samples_info['trial_lengths2'].append(minimo_largo2)

                    # Preprocesamiento
                    keep_indexes1_trial = Processing.preproc_dict2(momentos_escucha=momentos_sujeto_1_trial, delays=self.delays,
                                            situacion=self.situacion)
                    keep_indexes2_trial = Processing.preproc_dict2(momentos_escucha=momentos_sujeto_2_trial, delays=self.delays,
                                            situacion=self.situacion)

                    # Add sum of all previous trials length
                    keep_indexes1_trial += np.sum(samples_info['trial_lengths1'][:-1])
                    keep_indexes2_trial += np.sum(samples_info['trial_lengths2'][:-1])

                    samples_info['keep_indexes1'].append(keep_indexes1_trial)
                    samples_info['keep_indexes2'].append(keep_indexes2_trial)

                # Convierto a DF
                Funciones.make_df_dict(Trial_sujeto_1)
                Funciones.make_df_dict(Trial_sujeto_2)

                # Adjunto a datos de sujeto
                if len(Trial_sujeto_1['EEG']):
                    for key in list(Sujeto_1.keys()):
                        Sujeto_1[key] = Sujeto_1[key].append(Trial_sujeto_1[key])

                if len(Trial_sujeto_2['EEG']):
                    for key in list(Sujeto_2.keys()):
                        Sujeto_2[key] = Sujeto_2[key].append(Trial_sujeto_2[key])

            except:
                pass

        info = Trial_channel_1['info']

        if not loaded_samples_info:
            samples_info['keep_indexes1'] = Funciones.flatten_list(samples_info['keep_indexes1'])
            samples_info['keep_indexes2'] = Funciones.flatten_list(samples_info['keep_indexes2'])
            # Save instants Data
            os.makedirs(self.samples_info_path, exist_ok=True)
            f = open(self.samples_info_path + 'sampels_info_{}.pkl'.format(self.sesion), 'wb')
            pickle.dump(samples_info['keep_indexes1'], f)

        # Dropeo columna de silencio de Phonemes
        if 'Phonemes' in Sujeto_1.keys():
            Sujeto_1['Phonemes'].drop(columns="", inplace=True)
            Sujeto_2['Phonemes'].drop(columns="", inplace=True)
        if 'Phonemes-manual' in Sujeto_1.keys():
            Sujeto_1['Phonemes-manual'].drop(columns="", inplace=True)
            Sujeto_2['Phonemes-manual'].drop(columns="", inplace=True)

        # Convierto a array
        Funciones.make_array_dict(Sujeto_1)
        Funciones.make_array_dict(Sujeto_2)

        if 'Spectrogram' in Sujeto_1.keys():
            Sujeto_1['Spectrogram'] = Sujeto_1['Spectrogram'].transpose()
            Sujeto_2['Spectrogram'] = Sujeto_2['Spectrogram'].transpose()

            # Shifted matrix row by row
            print('Computing shifted matrix for the Spectrogram')
            spec_shift_1 = Processing.matriz_shifteada(Sujeto_1['Spectrogram'][0], self.delays)
            for i in np.arange(1, len(Sujeto_1['Spectrogram'])):
                spec_shift_1 = np.hstack((spec_shift_1, Processing.matriz_shifteada(Sujeto_1['Spectrogram'][i], self.delays)))
            Sujeto_1['Spectrogram'] = spec_shift_1

            spec_shift_2 = Processing.matriz_shifteada(Sujeto_2['Spectrogram'][0], self.delays)
            for i in np.arange(1, len(Sujeto_2['Spectrogram'])):
                spec_shift_2 = np.hstack((spec_shift_2, Processing.matriz_shifteada(Sujeto_2['Spectrogram'][i], self.delays)))
            Sujeto_2['Spectrogram'] = spec_shift_2

        if 'Phonemes' in Sujeto_1.keys():
            Sujeto_1['Phonemes'] = Sujeto_1['Phonemes'].transpose()
            Sujeto_2['Phonemes'] = Sujeto_2['Phonemes'].transpose()

            print('Computing shifted matrix for the Phonemes')
            ph_shift_1 = Processing.matriz_shifteada(Sujeto_1['Phonemes'][0], self.delays)
            for i in np.arange(1, len(Sujeto_1['Phonemes'])):
                ph_shift_1 = np.hstack(
                    (ph_shift_1, Processing.matriz_shifteada(Sujeto_1['Phonemes'][i], self.delays)))
            Sujeto_1['Phonemes'] = ph_shift_1

            ph_shift_2 = Processing.matriz_shifteada(Sujeto_2['Phonemes'][0], self.delays)
            for i in np.arange(1, len(Sujeto_2['Phonemes'])):
                ph_shift_2 = np.hstack(
                    (ph_shift_2, Processing.matriz_shifteada(Sujeto_2['Phonemes'][i], self.delays)))
            Sujeto_2['Phonemes'] = ph_shift_2

        if 'Phonemes-manual' in Sujeto_1.keys():
            Sujeto_1['Phonemes-manual'] = Sujeto_1['Phonemes-manual'].transpose()
            Sujeto_2['Phonemes-manual'] = Sujeto_2['Phonemes-manual'].transpose()

            print('Computing shifted matrix for the Phonemes')
            ph_shift_1 = Processing.matriz_shifteada(Sujeto_1['Phonemes-manual'][0], self.delays)
            for i in np.arange(1, len(Sujeto_1['Phonemes-manual'])):
                ph_shift_1 = np.hstack(
                    (ph_shift_1, Processing.matriz_shifteada(Sujeto_1['Phonemes-manual'][i], self.delays)))
            Sujeto_1['Phonemes-manual'] = ph_shift_1

            ph_shift_2 = Processing.matriz_shifteada(Sujeto_2['Phonemes-manual'][0], self.delays)
            for i in np.arange(1, len(Sujeto_2['Phonemes-manual'])):
                ph_shift_2 = np.hstack(
                    (ph_shift_2, Processing.matriz_shifteada(Sujeto_2['Phonemes-manual'][i], self.delays)))
            Sujeto_2['Phonemes-manual'] = ph_shift_2

        keys = list(Sujeto_1.keys())
        keys.remove('EEG')
        for key in keys:
            if key != 'Spectrogram' and key != 'Phonemes' and key != 'Phonemes-manual':
                Sujeto_1[key] = Processing.matriz_shifteada(Sujeto_1[key], self.delays)
                Sujeto_2[key] = Processing.matriz_shifteada(Sujeto_2[key], self.delays)

        # Keep good time instants
        for key in list(Sujeto_1.keys()):
            Sujeto_1[key] = Sujeto_1[key][samples_info['keep_indexes1'], :]
            Sujeto_2[key] = Sujeto_2[key][samples_info['keep_indexes2'], :]

        # Define Save Paths
        Paths = {}
        Paths['EEG'] = self.procesed_data_path + 'EEG/'
        if self.Band and self.Causal_filter_EEG: Paths['EEG'] += 'Causal_'
        Paths['EEG'] += 'Sit_{}_Band_{}/'.format(self.situacion, self.Band)

        Paths['Envelope'] = self.procesed_data_path + 'Envelope/Sit_{}/'.format(self.situacion)

        Paths['Pitch'] = self.procesed_data_path + 'Pitch_threshold_{}/Sit_{}_Faltantes_{}/'.format(self.SilenceThreshold,
                                                                                                self.situacion,
                                                                                                self.valores_faltantes)

        Paths['Spectrogram'] = self.procesed_data_path + 'Spectrogram/Sit_{}/'.format(self.situacion)
        Paths['Phonemes'] = self.procesed_data_path + 'Phonemes/Sit_{}/'.format(self.situacion)
        Paths['Phonemes-manual'] = self.procesed_data_path + 'Phonemes-manual/Sit_{}/'.format(self.situacion)

        for key in Sujeto_1.keys():
            # Save Preprocesed Data
            os.makedirs(Paths[key], exist_ok=True)
            f = open(Paths[key] + 'Sesion{}.pkl'.format(self.sesion), 'wb')
            pickle.dump([Sujeto_1[key], Sujeto_2[key]], f)
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
                f = open(self.procesed_data_path + 'Pitch_threshold_{}/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.SilenceThreshold, self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

                if self.valores_faltantes == None:
                    stimuli_para_sujeto_1, stimuli_para_sujeto_2 = stimuli_para_sujeto_1[stimuli_para_sujeto_1 != 0], \
                                                                   stimuli_para_sujeto_2[
                                                                       stimuli_para_sujeto_2 != 0]  # saco 0s
                elif self.valores_faltantes:
                    stimuli_para_sujeto_1[stimuli_para_sujeto_1 == 0], stimuli_para_sujeto_2[
                        stimuli_para_sujeto_2 == 0] = self.valores_faltantes, self.valores_faltantes  # cambio 0s

            if stimuli == 'PitchMask':
                f = open(self.procesed_data_path + 'Pitch_mask_threshold_{}/Sit_{}_Faltantes_{}/Sesion{}.pkl' \
                         .format(self.SilenceThreshold, self.situacion, self.valores_faltantes, self.sesion), 'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            if stimuli == 'Spectrogram':
                f = open(
                    self.procesed_data_path + 'Spectrogram/Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion),
                    'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            if stimuli == 'Phonemes':
                f = open(
                    self.procesed_data_path + 'Phonemes/Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion),
                    'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            if stimuli == 'Phonemes-manual':
                f = open(
                    self.procesed_data_path + 'Phonemes-manual/Sit_{}/Sesion{}.pkl'.format(self.situacion, self.sesion),
                    'rb')
                stimuli_para_sujeto_1, stimuli_para_sujeto_2 = pickle.load(f)
                f.close()

            Sujeto_1[stimuli] = stimuli_para_sujeto_1
            Sujeto_2[stimuli] = stimuli_para_sujeto_2

        Sesion = {'Sujeto_1': Sujeto_1, 'Sujeto_2': Sujeto_2}

        return Sesion


def Load_Data(sesion, stim, Band, sr, tmin, tmax, procesed_data_path, situacion='Escucha', Causal_filter_EEG=True,
              Env_Filter=False, valores_faltantes=0, Calculate_pitch=False, SilenceThreshold=0.03):
    possible_stims = ['Envelope', 'Pitch', 'PitchMask', 'Spectrogram', 'Phonemes', 'Phonemes-manual']

    if all(stimulus in possible_stims for stimulus in stim.split('_')):

        Sesion_obj = Sesion_class(sesion=sesion, stim=stim, Band=Band, sr=sr, tmin=tmin, tmax=tmax,
                                  valores_faltantes=valores_faltantes, Causal_filter_EEG=Causal_filter_EEG,
                                  Env_Filter=Env_Filter, situacion=situacion, Calculate_pitch=Calculate_pitch,
                                  SilenceThreshold=SilenceThreshold, procesed_data_path=procesed_data_path)

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