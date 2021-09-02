<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:20:42 2021

@author: joaco
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

Band = 'Theta'
tmin, tmax = -0.53, 0.3

f = open('saves/Ridge/Final_Correlation/Envelope_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Envelope = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Pitch_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Pitch = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Envelope_Pitch_Pitch_der_EEG_{}(tmin{}_tmax{}).pkl'.format(Band,tmin,tmax), 'rb')
Corr_Envelope_Pitch_Pitch_der = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Pitch_der_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Pitch_der = pickle.load(f)
f.close()

Corr_Envelope = Corr_Envelope.transpose()
Corr_Pitch = Corr_Pitch.transpose()
Corr_Pitch_der = Corr_Pitch_der.transpose()
Corr_Envelope_Pitch_Pitch_der = Corr_Envelope_Pitch_Pitch_der.transpose()

plt.ion()
plt.figure()
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Envelope, '.', color = 'C0')
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Pitch, '.', color = 'C1')
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Pitch_der, '.', color = 'C2')
plt.plot([0,np.max(Corr_Envelope_Pitch_Pitch_der)*1.25], [0,np.max(Corr_Envelope_Pitch_Pitch_der)*1.25], 'k--')
plt.ylabel('Individual model (r)')
plt.xlabel('Full model (r)')
plt.grid()
# plt.legend()
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 19:20:42 2021

@author: joaco
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

Band = 'Theta'
tmin, tmax = -0.53, 0.3

f = open('saves/Ridge/Final_Correlation/Envelope_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Envelope = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Pitch_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Pitch = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Envelope_Pitch_Pitch_der_EEG_{}(tmin{}_tmax{}).pkl'.format(Band,tmin,tmax), 'rb')
Corr_Envelope_Pitch_Pitch_der = pickle.load(f)
f.close()

f = open('saves/Ridge/Final_Correlation/Pitch_der_EEG_{}(tmin{}_tmax{}).pkl'.format(Band, tmin,tmax), 'rb')
Corr_Pitch_der = pickle.load(f)
f.close()

Corr_Envelope = Corr_Envelope.transpose()
Corr_Pitch = Corr_Pitch.transpose()
Corr_Pitch_der = Corr_Pitch_der.transpose()
Corr_Envelope_Pitch_Pitch_der = Corr_Envelope_Pitch_Pitch_der.transpose()

plt.ion()
plt.figure()
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Envelope, '.', color = 'C0')
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Pitch, '.', color = 'C1')
plt.plot(Corr_Envelope_Pitch_Pitch_der, Corr_Pitch_der, '.', color = 'C2')
plt.plot([0,np.max(Corr_Envelope_Pitch_Pitch_der)*1.25], [0,np.max(Corr_Envelope_Pitch_Pitch_der)*1.25], 'k--')
plt.ylabel('Individual model (r)')
plt.xlabel('Full model (r)')
plt.grid()
# plt.legend()
>>>>>>> 84f0a87c261eaf6f64ecdfe637ee7050c0c0dc44
