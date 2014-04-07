import os
import scipy.io.wavfile
import scipy.interpolate
import numpy as np
import sys

import warnings
warnings.filterwarnings('error')

def p(w, x):
	# k = len(w)
	# m = len(x)
	# n = len(x[0])
	
	# py = np.zeros((m, k), dtype=np.float64)

	# for j in range(k):
	# 	# row = np.exp(x.dot(w[j])) / (1.0 + np.sum(np.exp(-(x.dot(np.delete(w, j, 0).T))), 1))
	# 	# row = 1.0 / (1.0 + np.sum(np.exp((x.dot(np.delete(w, j, 0).T))), 1))
	# 	# row = np.exp((x.dot(w[j].T))) / (1.0 + np.sum(np.exp((x.dot(np.delete(w, j, 0).T))), 1))
	# 	# assert (x.dot(w[j].T) < 200).all(), str(x.dot(w[j].T))
	# 	# print np.log(x.dot(w[j].T))
		
	# 	row = 1.0 / (1.0 + np.exp(x.dot(w[j].T).clip(-30, 30)))

	# 	py[:, j] = row

	return 1.0 / (1.0 + np.exp(x.dot(w.T).clip(-30, 30)))

	# return py

def cost(x, y, w, regularization):
	m, n, k = dimension(x, y)

	py = p(w, x)

	# py = np.zeros((m, k))
	# assert not (py == 0).any()
	# print py

	a = np.log(py)
	a2 = np.log(1.0 - py)

	return -(1.0 / m) * (np.sum(y * a + (1.0 - y) * a2) + (regularization / (2.0 * m)) * np.sum(w * w))
	
	# return np.sum(y * a + (1.0 - y) * a2)

def gradient(learning_rate, regularization, x, y, w):
	m, n, k = dimension(x, y)

	# Compute full P(Y|X, W) probability matrix with (m, k) dimensions
	py = p(w, x)

	# d = np.zeros((k, n), dtype=np.float64)
	
	w2 = np.copy(w)
	w2[:, 0] = 0

	# d2 = (y - py).T.dot(x) - (regularization * w2)
	
	d2 = (1.0 / m) * (py - y).T.dot(x) - (regularization / m * w2)
	
	# Compute gradient ascent
	w2 = w + learning_rate * d2

	return w2

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
	w = np.zeros((k, n), dtype=np.float64)
	# w = np.random.random((k, n))


	# w2 = learn(stop, learning_rate, regularization, x, y, w)
	# predicted_y = classify(x, y, w2)

	# print "Confusion matrix = "
	# c = confusion(predicted_y, y)
	# print c
	# print "Accuracy = " + str(accuracy(c))


	x_orig = x
	y_orig = y

	x = x.tolist()
	y = y.tolist()
	total_accuracy = 0

	for j in range(10):	
		offset = 0
		skip = j * 10
		training = []
		training_y = []
		validation = []
		validation_y = []
		for i in range(6):
			offset = 100 * i
			validation += x[offset + skip : offset + skip + 10]
			validation_y += y[offset + skip : offset + skip + 10]

			training += x[offset : offset + skip]
			training += x[offset + skip + 10 : offset + 100]
			training_y += y[offset : offset + skip]
			training_y += y[offset + skip + 10 : offset + 100]

			# print "training with: " + str((offset + skip, offset + skip + 10))
			# print "validating with: " + str((offset, offset + skip)) + " and " + str((offset + skip + 10, offset + 100))

		training = np.array(training)
		training_y = np.array(training_y)
		validation = np.array(validation)
		validation_y = np.array(validation_y)

		w2 = learn(stop, learning_rate, regularization, training, training_y, w)
		predicted_y = classify(validation, validation_y, w2)

		# print "cost for cross-validation #" + str(j + 1) + " = " + str(cost(x_orig, y_orig, w2, regularization))
		print "Confusion matrix = "
		c = confusion(predicted_y, validation_y)
		print c
		total_accuracy += accuracy(c) * 100
	print "Total accuracy: " + str(total_accuracy/10)

def dist(x,y):   
	return np.sqrt(np.sum((x-y)**2))

def learn(stop, learning_rate, regularization, x, y, w):
	m, n, k = dimension(x, y)

	# print "first cost = "  + str(cost(x, y, w))

	d1 = 1000000
	d2 = 0

	c1 = 1000000
	c2 = 0

	while True:
		w2 = gradient(learning_rate, regularization, x, y, w)
		# d2 = dist(w, w2)
		c2 = cost(x, y, w2, regularization)

		# print w
		# print w2

		# print d2
		# print "cost = " + str(c2)
		# predicted_y = classify(xc, yc, w2)
		# c = confusion(predicted_y, yc)
		# print "Accuracy = " + str(accuracy(c))

		if c2 > c1:
			learning_rate /= 2.0
			# print "Gradient ascient is diverging, adjusting learning rate to: " + str(learning_rate)

		if abs(c2 - c1) < stop:
			w = w2
			break

		w = w2	
		d1 = d2
		c1 = c2
	return w

def read_files(x_file, y_file):
	if os.path.isfile(x_file):
		X = np.loadtxt(x_file)
		Y = np.loadtxt(y_file)

		# feature scaling
		# X = (X - X.min(0)) / X.max(0)

		X[:, 0] = np.random.random((X.shape[0]))

		for i in range(len(X[0])):
			f = scipy.interpolate.interp1d([X[:, i].min(), X[:, i].max()], [-1.0, 1.0])
			X[:, i] = f(X[:, i])

		# bias feature
		X[:, 0] = 1.0

		return (X, Y)
	else:
		print "X and Y matrix files not found, computing"

		m, n, k = 600, 1001, 6
		num_samples = 100

		X = np.empty((m, n), dtype=np.float64)
		Y = np.zeros((m, k))

		for i in range(len(genres)):
			genre = genres[i]
			for j in range(num_samples):
				print "reading: " + data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav'
				sample_rate, x = scipy.io.wavfile.read(data_folder + genre + '/' + genre + '.000' + ('%02d' % j) + '.wav')
				X[j + i * num_samples] = np.insert(abs(scipy.fft(x)[:n - 1]), [0], 1.0)
				Y[j + i * num_samples, i] = 1.0

		X[:, 0] = np.random.random((m))

		np.savetxt(x_file, X)
		np.savetxt(y_file, Y)

		# np.savetxt(x_file, X)

		return (X, Y)

def dimension(x, y):
	return (len(x), len(x[0]), len(y[0]))

# MAIN
data_folder = 'opihi.cs.uvic.ca/sound/genres/'

x_file = ""
y_file = "classes.npy"

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
