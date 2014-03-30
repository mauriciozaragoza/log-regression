import scipy.io.wavfile
from matplotlib.pyplot import specgram
import numpy as np
#import scikits.talkbox

def h(w, x):
	return 1/(1 + np.exp(np.dot(w,x)))	

# MAIN

W = np.zeros(())

data_folder = 'opihi.cs.uvic.ca/sound/genres/'
genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

sample_rate, X = scipy.io.wavfile.read(data_folder + "classical/classical.00000.wav")
print sample_rate, X.shape
fft_features = abs(scipy.fft(X)[:1000])
print fft_features

# specgram(X, Fs=sample_rate, xextent=(0, 30))
