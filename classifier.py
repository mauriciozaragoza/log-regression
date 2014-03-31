import os
import scipy.io.wavfile
from matplotlib.pyplot import specgram
import numpy as np
# import scikits.talkbox

genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

data_folder = 'opihi.cs.uvic.ca/sound/genres/'

x_file = "features.npy"
y_file = "classes.npy"

m = 600
n = 1001
k = 6 

num_samples = 100

W = np.zeros((k, n))

def read_files():
	X = np.empty((num_samples * len(genres), n), dtype=np.float64);
	Y = np.zeros((m, k))

	for i in range(len(genres)):
		genre = genres[i]
		for j in range(num_samples):
			print "reading: " + data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav'
			sample_rate, x = scipy.io.wavfile.read(data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav')
			X[i] = np.insert(abs(scipy.fft(x)[:n - 1]), [0], 1.0)
			Y[j + i * num_samples, i] = 1.0

	X /= np.max(X)
	X[:, 0] = 1.0

	np.save(x_file, X)
	np.save(y_file, Y)

	return (X, Y)

if os.path.isfile(x_file):
	X = np.load(x_file)
	Y = np.load(y_file)
else:
	print "X and Y matrix files not found, computing"
	X, Y = read_files();

print X
print Y