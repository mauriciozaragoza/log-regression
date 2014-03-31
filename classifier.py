import scipy.io.wavfile
from matplotlib.pyplot import specgram
import numpy as np
#import scikits.talkbox

def exp(yk, w, x):
	v = range(0,6)
	v.remove(yk)
	w = w[v,:]
	return np.sum(np.exp(np.dot(w,x)))

def p(yk, w, x):
	return 1/(1 + exp(yk,w,x))

def likelihood(m, k, y, w, x):
	sum = 0
	for i in range(m):
		for j in range(k):
			sum += y[i][j] * log(p(j,w,x[j])) + ((1 - y[i][j])*log(1 - p(j,w,x[j]))) 

# MAIN

n = 1001
k = 6
m = 600
W = np.zeros(())

data_folder = 'opihi.cs.uvic.ca/sound/genres/'
genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

#sample_rate, X = scipy.io.wavfile.read(data_folder + "classical/classical.00000.wav")
#print sample_rate, X.shape
#fft_features = abs(scipy.fft(X)[:1000])
#print fft_features
# specgram(X, Fs=sample_rate, xextent=(0, 30))

# yk = 1
# w = np.array([[1,1,1],[1,1,2],[1,1,3]])
# x = np.array([1,1,1])

# print w

# print exp(yk, w, x)
