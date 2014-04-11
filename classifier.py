import os
import scipy.io.wavfile
import scipy.interpolate
import numpy as np
import sys

# Computes P(Y|X, W) as a matrix, exponent values are clamped to a range [-30, 30] to prevent overflows
def p(w, x):
	return 1.0 / (1.0 + np.exp(x.dot(w.T).clip(-30, 30)))

# Computes the likelihood of a matrix of weights
def likelihood(x, y, w, regularization):
	m = x.shape[0]

	py = p(w, x)
	return -(1.0 / m) * (np.sum(y * np.log(py) + (1.0 - y) * np.log(1.0 - py)) + (regularization / (2.0 * m)) * np.sum(w * w))

# Computes the next step of gradient ascent
def gradient(learning_rate, regularization, x, y, w):
	# Compute full P(Y|X, W) probability matrix with (m, k) dimensions
	py = p(w, x)
	
	# The weight for the bias feature does not have a regularization penalty
	# So it's weight is temporally set to 0 to remove it
	w2 = np.copy(w)
	w2[:, 0] = 0

	# Computes the gradient for each feature and label
	# yielding a (k, n)-dimensional matrix
	d = (py - y).T.dot(x) - (regularization * w2)
	
	# Compute gradient ascent
	return w + learning_rate * d

# Classifies a matrix of samples given the weight matrix
def classify(x, w):
	# Compute full P(Y|X, W) probability matrix with (m, k) dimensions
	probabilities = p(w, x)

	# Obtain the classes which maximize the probabilities
	max_probabilities = probabilities.argmax(1)

	# Build the new binary Y matrix
	for i in range(probabilities.shape[0]):
		probabilities[i, :] = 0
		probabilities[i, max_probabilities[i]] = 1

	return probabilities

# Compute the confusion matrix given the predicted and the real Y values of a dataset
def confusion(predicted_y, real_y):
	m = len(predicted_y)
	k = len(predicted_y[0])

	c = np.zeros((k, k))

	predicted_y_index = predicted_y.argmax(1)
	real_y_index = real_y.argmax(1)

	# Increment each row-column according to the predicted/real label of each sample
	# The diagonal of this matrix is expected to have larger values than the rest of the entries
	for i in range(m):
		c[real_y_index[i], predicted_y_index[i]] += 1

	return c

# Computes accuracy given a confusion matrix
def accuracy(c):
	return sum([c[i][i] for i in range(6)]) / np.sum(c)

# Trains the logistic-regression classifier using 10-fold cross-validation
# and reports the accuracy for each iteration
def cross_validation(stop, learning_rate, regularization, x, y):
	m, n, k = dimension(x, y)

	# Initial weights are set to 0
	w = np.zeros((k, n), dtype=np.float64)

	x_orig = x
	y_orig = y

	x = x.tolist()
	y = y.tolist()
	total_accuracy = 0

	# For each iteration in the cross-validation
	for j in range(10):	
		offset = 0
		skip = j * 10
		training = []
		training_y = []
		validation = []
		validation_y = []

		# Obtain the training and validation sets according to the current iteration
		for i in range(6):
			offset = 100 * i
			validation += x[offset + skip : offset + skip + 10]
			validation_y += y[offset + skip : offset + skip + 10]

			training += x[offset : offset + skip]
			training += x[offset + skip + 10 : offset + 100]
			training_y += y[offset : offset + skip]
			training_y += y[offset + skip + 10 : offset + 100]

		training = np.array(training)
		training_y = np.array(training_y)
		validation = np.array(validation)
		validation_y = np.array(validation_y)

		# Compute the optimal weights for this iteration
		# optimality is defined by a given learning-rate, regularization and stop constants
		w2 = learn(stop, learning_rate, regularization, training, training_y, w)

		# Classify the validation set with the new weight matrix
		predicted_y = classify(validation, w2)

		# Compute the confusion matrix for this iteration and report accuracy
		c = confusion(predicted_y, validation_y)
		total_accuracy += accuracy(c) * 10

		print "#" + str(j) + " cross-validation confusion matrix: "
		print c
		print "Accuracy: " + str(accuracy(c) * 100)

	print "Total accuracy: " + str(total_accuracy)

# Obtains the 21 most relevant features
def rank(w):
	return np.abs(w).argsort(1)[:, :21]

def learn(stop, learning_rate, regularization, x, y, w):
	m, n, k = dimension(x, y)

	c1 = 1000000
	c2 = 0

	while True:
		# Computes the next matrix of weights according to the next iteration
		# of gradient ascent
		w2 = gradient(learning_rate, regularization, x, y, w)

		# Computes the likelihood of the new weights
		c2 = likelihood(x, y, w2, regularization)

		# If gradient-ascent starts diverging, adjust learning rate accordingly
		if c2 > c1:
			learning_rate /= 2.0

		# If slope is less than a specified slope, stop gradient-ascent
		if abs(c2 - c1) < stop:
			w = w2
			break

		w = w2	
		c1 = c2
	return w

# Read the given X and Y matrix files and return them as 
# numpy arrays
def read_files(x_file, y_file):
	X = np.loadtxt(x_file)
	Y = np.loadtxt(y_file)

	# Initialize the bias column as random numbers to be able to normalize features
	X[:, 0] = np.random.random((X.shape[0]))

	for i in range(len(X[0])):
		f = scipy.interpolate.interp1d([X[:, i].min(), X[:, i].max()], [-1.0, 1.0])
		X[:, i] = f(X[:, i])

	# Set the bias feature to 1.0
	X[:, 0] = 1.0

	return (X, Y)

# Returns the (m, n, k) dimensions given the X and Y matrices
def dimension(x, y):
	return (len(x), len(x[0]), len(y[0]))

# MAIN
data_folder = 'opihi.cs.uvic.ca/sound/genres/'

x_file = ""
y_file = "classes.txt"

genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']

if len(sys.argv) == 2:
	if sys.argv[1] == "-f":

		x_file = "features.txt"
		X, Y = read_files(x_file, y_file)
		cross_validation(0.0001, 0.001, 0.001, X, Y)

	elif sys.argv[1] == "-m":
		
		x_file = "features_mfcc.txt"
		X, Y = read_files(x_file, y_file)
		cross_validation(0.00001, 0.001, 0.01, X, Y)
	
	elif sys.argv[1] == "-r":
		
		x_file = "features-120-fft.txt"
		X, Y = read_files(x_file, y_file)
		cross_validation(0.0001, 0.001, 0.001, X, Y)
		
else:

	print "Usage: classifier.py (-f | -m | -r)"
	print "   -f, Classify using 1000 FFT components"
	print "   -r, Classify using the top-ranked 120 FFT components"
	print "   -m, Classify using MFCC"
	sys.exit()