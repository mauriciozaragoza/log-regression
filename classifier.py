import os
import scipy.io.wavfile
from matplotlib.pyplot import specgram
import numpy as np
# import scikits.talkbox

n = 1001
k = 6
m = 600

W = np.zeros((k, n))


data_folder = 'opihi.cs.uvic.ca/sound/genres/'

x_file = "features.npy"
y_file = "classes.npy"

num_samples = 100

genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

def exp(yk, w, x):
	values = range(0,6)
	values.remove(yk)
	w = w[v,:]
	return np.sum(np.exp(np.dot(w,x)))

def p(yk, w, x):
	return 1/(1 + exp(yk,w,x))

def likelihood(m, k, y, w, x):
	sum = 0
	for i in range(m):
		for j in range(k):
			sum += y[i][j] * log(p(j,w,x[i])) + ((1 - y[i][j])*log(1 - p(j,w,x[i]))) 
	return sum

def gradient(k, m, learning_rate, regularization, y, w, x):
	cw = np.copy(w)
	for i in range(k):
		for j in range(n):
			sum = 0
			for l in range(m):
				sum += x[l][j] * (y[l][i] - p(i, w, x[m]))
			cw[i][j] += learning_rate * sum - learning_rate*regularization*w[i][j]
	return cw

def read_files(x_file, y_file):
	if os.path.isfile(x_file):
		X = np.load(x_file)
		Y = np.load(y_file)

		return (X, Y)
	else:
		print "X and Y matrix files not found, computing"

		X = np.empty((num_samples * len(genres), n), dtype=np.float64)
		Y = np.zeros((m, k))

		for i in range(len(genres)):
			genre = genres[i]
			for j in range(num_samples):
				print "reading: " + data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav'
				sample_rate, x = scipy.io.wavfile.read(data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav')
				X[j + i * num_samples] = np.insert(abs(scipy.fft(x)[:n - 1]), [0], 1.0)
				Y[j + i * num_samples, i] = 1.0

		X /= np.max(X)
		X[:, 0] = 1.0

		np.save(x_file, X)
		np.save(y_file, Y)

		return (X, Y)

# MAIN
X, Y = read_files(x_file, y_file)

print X
print Y