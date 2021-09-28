import Analysis_Pipeline

# Random permutations
Statistical_test = False
Simulate_random_data = False

# Figures
Display_Ind_Figures = False
Display_Total_Figures = True
Save_Ind_Figures = False
Save_Total_Figures = False

Stims_Order = ['Envelope', 'Pitch', 'Pitch_der', 'Spectrogram', 'Phonemes']
Stims = ['Envelope', 'Pitch', 'Pitch_der', 'Envelope_Pitch_Pitch_der']
Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']

# Define Parameters

# Stimuli and EEG
stim = 'Envelope'
Band = 'Theta'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003

# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Model
# Model = Ridge
alpha = 100

Analysis_Pipeline.Run(stim, Band, tmin, tmax, situacion, Stims_preprocess, EEG_preprocess, alpha, Statistical_test,
                      Simulate_random_data, Display_Ind_Figures, Display_Total_Figures, Save_Ind_Figures, Save_Total_Figures,
                      Stims_Order)