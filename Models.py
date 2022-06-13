from sklearn import linear_model
from mne.decoding import ReceptiveField


class Ridge:
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.model = linear_model.Ridge(self.alpha)
        
    def fit(self, dstims_train_val, eeg_train_val):  
        self.model.fit(dstims_train_val, eeg_train_val)
        self.coefs = self.model.coef_
    
    def predict(self, dstims_test):   
        predicted = self.model.predict(dstims_test)
        return predicted


class mne_mtrf:

    def __init__(self, tmin, tmax, sr, alpha, present_stim_index):
        self.sr = sr
        self.rf = ReceptiveField(tmin, tmax, sr, estimator=alpha, scoring='corrcoef', verbose=False)
        self.present_stim_index = present_stim_index

    def fit(self, dstims_train_val, eeg_train_val):
        stim = dstims_train_val[:, self.present_stim_index]
        stim = stim.reshape([stim.shape[0], 1])

        self.rf.fit(stim, eeg_train_val)
        self.coefs = self.rf.coef_[:, 0, :]

    def predict(self, dstims_test):
        stim = dstims_test[:, self.present_stim_index]
        stim = stim.reshape([stim.shape[0], 1])
        predicted = self.rf.predict(stim)
        return predicted


class mne_mtrf_decoding:

    def __init__(self, tmin, tmax, sr, info, alpha, t_lag):
        self.sr = sr
        self.t_lag = t_lag
        self.rf = ReceptiveField(tmin, tmax, sr, feature_names=info.ch_names, estimator=alpha, scoring='corrcoef',
                                 patterns=True, verbose=False)

    def fit(self, eeg_train_val, dstims_train_val):
        stim = dstims_train_val[:, self.t_lag]
        stim = stim.reshape([stim.shape[0], 1])
        self.rf.fit(eeg_train_val, stim)
        self.coefs = self.rf.coef_[0, :, :]
        self.patterns = self.rf.patterns_[0, :, :]

    def predict(self, eeg_test):
        predicted = self.rf.predict(eeg_test)
        return predicted

#
# class mne_mtrf_decoding_inicial:
#
#     def __init__(self, tmin, tmax, sr, info, alpha):
#         self.sr = sr
#         self.rf = ReceptiveField(tmin, tmax, sr, feature_names=info.ch_names, estimator=alpha, scoring='corrcoef', patterns=True)
#
#     def fit(self, eeg_train_val, dstims_train_val):
#         stim = dstims_train_val[:, 0]
#         stim = stim.reshape([stim.shape[0], 1])
#         self.rf.fit(eeg_train_val, stim)
#         self.coefs = self.rf.coef_[0, :, 1:]
#         self.patterns = self.rf.patterns_[0, :, 1:]
#
#     def predict(self, eeg_test):
#         predicted = self.rf.predict(eeg_test)
#         return predicted
