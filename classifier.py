import os
import scipy.io.wavfile
from matplotlib.pyplot import specgram
import numpy as np
import sys

def p(w, x):
	k = len(w)
	py = []

	for j in range(k):
		row = 1.0 / (1.0 + np.sum(np.exp(-x.dot(np.delete(w, j, 0).T)), 1))
		py.append(row)

	return np.array(py).T

def cost(x, y, w, regularization):
	m, n, k = dimension(x, y)

	py = p(w, x)

	return (1.0 / m) * (np.sum(-y * np.log(py) - (1.0 - y) * np.log(1.0 - py)) + (regularization / (2 * m)) * np.sum(w * w)) / k

def gradient(learning_rate, regularization, x, y, w):
	m, n, k = dimension(x, y)

	# cw = np.copy(w)

	# Compute full P(Y|X, W) probability matrix with (m, k) dimensions
	py = p(w, x)

	# assert abs(np.sum(np.sum(py, 1) - 1.0)) < 0.5, "Probabilities do not sum to 1, aborting = " + str(abs(np.sum(np.sum(py, 1) - 1.0)))

	# get partial derivatives
	d = (1 / m) * (py - y).T.dot(x) + (regularization / m * w)
	d[:, 0] = 0

	# Compute gradient ascent
	w = w - learning_rate * d

	return w

def classify(x, y, w):
	probabilities = p(w, x)

	max_probabilities = probabilities.argmax(1)

	for i in range(probabilities.shape[0]):
		probabilities[i, :] = 0
		probabilities[i, max_probabilities[i]] = 1

	return probabilities

def confusion(predicted_y, real_y):
	m = len(predicted_y)
	k = len(predicted_y[0])

	c = np.zeros((k, k))

	predicted_y_index = predicted_y.argmax(1)
	real_y_index = real_y.argmax(1)

	for i in range(m):
		c[real_y_index[i], predicted_y_index[i]] += 1

	return c

def accuracy(c):
	return sum([c[i][i] for i in range(6)]) / np.sum(c)

def cross_validation(stop, learning_rate, regularization, x, y):
	m, n, k = dimension(x, y)
	# w = np.zeros((k, n))
	w = np.random.random((k, n))

	x_orig = x
	y_orig = y

	x = x.tolist()
	y = y.tolist()

	for j in range(10):	
		offset = 0
		skip = j * 10
		training = []
		training_y = []
		validation = []
		validation_y = []
		for i in range(6):
			offset = 100 * i
			training += x[offset + skip : offset + skip + 10]
			training_y += y[offset + skip : offset + skip + 10]

			validation += x[offset : offset + skip]
			validation += x[offset + skip + 10 : offset + 100]
			validation_y += y[offset : offset + skip]
			validation_y += y[offset + skip + 10 : offset + 100]

			# print "training with: " + str((offset + skip, offset + skip + 10))
			# print "validating with: " + str((offset, offset + skip)) + " and " + str((offset + skip + 10, offset + 100))

		training = np.array(training)
		training_y = np.array(training_y)
		validation = np.array(validation)
		validation_y = np.array(validation_y)

		w2 = learn(stop, learning_rate, regularization, training, training_y, w)
		predicted_y = classify(validation, validation_y, w2)

		print "cost for cross-validation #" + str(j + 1) + " = " + str(cost(x_orig, y_orig, w2, regularization))
		print "Confusion matrix = "
		c = confusion(predicted_y, validation_y)
		print c
		print "Accuracy = " + str(accuracy(c))

def dist(x,y):   
	return np.sqrt(np.sum((x-y)**2))

def learn(stop, learning_rate, regularization, x, y, w):
	m, n, k = dimension(x, y)

	# print "first cost = "  + str(cost(x, y, w))

	d1 = 1000000
	d2 = 0

	while True:
		w2 = gradient(learning_rate, regularization, x, y, w)
		d2 = dist(w, w2)
		# print d2
		# print "cost = " + str(cost(x, y, w2, regularization))

		if d2 > d1:
			learning_rate /= 2.0
			# print "Gradient ascient is diverging, adjusting learning rate to: " + str(learning_rate)

		if d2 < stop:
			w = w2
			break	
		w = w2	
		d1 = d2
	return w

def read_files(x_file, y_file):
	if os.path.isfile(x_file):
		X = np.loadtxt(x_file)
		Y = np.loadtxt(y_file)

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

		np.savetxt(x_file, X)
		np.savetxt(y_file, Y)

		return (X, Y)

def dimension(x, y):
	return (len(x), len(x[0]), len(y[0]))

# MAIN

data_folder = 'opihi.cs.uvic.ca/sound/genres/'

x_file = ""
y_file = "classes.npy"

num_samples = 100
genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

if len(sys.argv) == 2:
	if sys.argv[1] == "-f":
		x_file = "features.npy"
	elif sys.argv[1] == "-m":
		x_file = "features_mfcc.npy"
	else:
		print "Error: Invalid arguments"
		sys.exit()
	X, Y = read_files(x_file, y_file)
	cross_validation(0.0001, 1, 0.1, X, Y)

