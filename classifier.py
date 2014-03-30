import numpy as np
import scikits.talkbox

def h(w, x):
	return 1/(1 + np.exp(np.dot(w,x)))	

# MAIN
w = np.array([1,2,3])
x = np.array([2,2,2])

print h(w,x)