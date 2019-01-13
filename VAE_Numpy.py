import numpy as np 
import mnist

mnist.init()

x_train, t_train, x_test, t_test = mnist.load()


"""
load() takes mnist.pkl and returns 4 numpy arrays.

x_train : 60,000x784 numpy array that each row contains flattened version of training images.
t_train : 1x60,000 numpy array that each component is true label of the corresponding training images.
x_test : 10,000x784 numpy array that each row contains flattened version of test images.
t_test : 1x10,000 numpy array that each component is true label of the corresponding test images.

""" 


"""
Architecture

(1x784) x (784x100) -> Relu():(1x100) x ()

(input) -> Dense(Relu, 100) -> 

"""

def approximate_relu(k=10, x):
	return 1.0 / (2 * k) * np.log(1 + np.exp(2*k*x))

def grad__approx_relu(k=10, x):
	return np.exp**(2 k x)/(1 + np.exp^(2 k x))

def gradient_descent(current_location, 
						gradient_function, 
						learning_rate, 
						stopping_threshold):
						
    
EPOCS = 1
TRAINING_SAMPLES = range(x_train.shape[0])
LAYER_1_DIM = 100

# Steps
# Initialize Network.
# Forward Propagate.
# Back Propagate Error.
# Train Network.
# Predict.
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

for itterations in EPOCS:
    for image_index in TRAINING_SAMPLES:
        
        L0_in = np.pad(x_train[image_index], 1,mode="constant")
        
        L1_in = L0
        L1_weights = np.zeros((L1_in.shape[0], LAYER_1_DIM))
        L1_activation = 