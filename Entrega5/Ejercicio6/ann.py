# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def tanh(x):
    return np.tanh(x)

def rand_11(n):
    return np.random.uniform(-1, 1, size=n)


def rmse(predictions, targets):
    """Compute the Root Mean Squared Error
    -predictions, targets: both are np.array from the same size
    """
    return np.sqrt(np.mean((predictions - targets)**2))

def mse(predictions, targets):
    """Compute the Root Mean Squared Error
    -predictions, targets: both are np.array from the same size
    """
    return np.mean((predictions - targets)**2)

def mse_matrix(x, y):
    return np.mean(np.power((x - y),2))

'''
Activation functions:
https://stats.stackexchange.com/questions/101560/tanh-activation-function-vs-sigmoid-activation-function
http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
'''
class NeuralNetwork(object):

    def __init__(self, nin, nhidden, nout, activation=None, list_activations=None):
        self.nin = nin
        self.nhidden = nhidden
        self.nout = nout
        self._init_weights()
        self.error_test = False
        self.error_output = None
        self.activation = sigmoid if activation is None else activation
        self.list_activations = list_activations
        self.enable_gradient_check = False
        self.gradient_error = []


    def _init_weights(self):
        self.weights = [
            np.asmatrix(np.random.randn(self.nin + 1, self.nhidden)),
            np.asmatrix(np.random.randn(self.nhidden + 1, self.nout)),
        ]
        
    def _feedforward(self, inputvector):
        outputs = []
        a = np.matrix(inputvector)

        if self.list_activations:
            for w, actv in zip(self.weights, self.list_activations):
                a = np.insert(a, 0, 1)
                a = actv(a * w)
                outputs.append(a)
        else:
            activation = self.activation
            for w in self.weights:
                a = np.insert(a, 0, 1)
                a = activation(a * w)
                outputs.append(a)
        return outputs

    def feedforward(self, inputvector):
        outputs = self._feedforward(inputvector)
        out = outputs[-1]
        return np.asarray(out).reshape(self.nout)

    def setup_errortest(self, error_func=None):
        self.error_test = True
        self._error_func = error_func
        self.error_output = []
    
    def setup_gradient_check(self, epsilon=10**-4, error_func=euclidean_distances):
        self.enable_gradient_check=True
        self.epsilon=epsilon
        self.gradient_error_func = error_func
        self.gradient_error = []

    def backpropagation(self, training_examples, alpha, iterations):
        """
         - H: (alpha) is the learning rate (e.g.: .05)
         - nhidden: number of hidden units
         - nin: number of input units
         - nout: number of output units
        """
        self._init_weights()

        if self.error_test:
            ef = self._error_func
        
        if self.enable_gradient_check:
            self.gradient_error = []
        
        itern = 0
        for _ in range(iterations):
            for x, t in training_examples:
                itern += 1
                if itern > iterations:
                    break
                deltas = [None, None]
                # Step 1 - propagate the feed forward
                outputs = self._feedforward(x)                
                out = outputs[1]
                # print(f"Feedforward {outputs}")
                
                # calculate values for gradient descent
                if self.enable_gradient_check:
                    plus_epsilon = self._feedforward(x + self.epsilon)[1]
                    plus_epsilon_result = t + self.epsilon
                    jplus = self.j(plus_epsilon, plus_epsilon_result)


                    minus_epsilon = self._feedforward(x - self.epsilon)[1]
                    minus_epsilon_result = t - self.epsilon
                    jminus = self.j(minus_epsilon, minus_epsilon_result)


                    pendiete = (jplus  - jminus)/(2*self.epsilon)
                    # pendiete = list(map(lambda x: (x[0] - x[1])/(2.0*self.epsilon), zip(plus_epsilon,outputs)))
                    # print(f"plus_epsilon: {plus_epsilon[1]}")
                    # print(f"minus_epsilon: {minus_epsilon[1]}")
                    # print(f"Pendiente: {pendiete[1]}")
                     


                # Step - 2 - calculate output units error
                # delta = out * (1 - out) * (tt - out)
                op1 = np.multiply(out, 1 - out)
                op2 = np.matrix(t) - out
                deltas[1] = np.multiply(op1, op2)



                # print ("op1 - %s, op2 - %s" % (str(op1.shape), str(op2.shape)))

                # Step - 3 - calculate hidden units error
                hidden_out = outputs[0] # first item on the list added
                # print ("weights[1] - %s, deltas[1] - %s" % (str(self.weights[1].shape), str(deltas[1].shape)))
                wd = self.weights[1] * deltas[1].transpose()
                # print ("wd - size : %s" % str(wd.shape))

                op1 = np.multiply(hidden_out, 1 - hidden_out)
                # print ("op1 - size: %s" % str(op1.shape))
                deltas[0] = np.multiply(op1, np.delete(wd, 0))

                # print (f"Delta: {deltas}")

                # Step - 4 - adjust weights
                xx = np.matrix(np.insert(x, 0, 1)).transpose()
                # print ("self.weights[0] - %s, xx - %s, deltas[0] - %s" % (str(self.weights[0].shape), str(xx.shape), str(deltas[0].shape)))
                self.weights[0] = self.weights[0] + alpha * xx * deltas[0]

                #hidden_outm = np.matrix(hidden_out).transpose()
                hom = np.insert(hidden_out, 0, 1).transpose()
                # print ("self.weights[1] - %s, hom - %s, deltas[1] - %s" % (str(self.weights[1].shape), str(hom.shape), str(deltas[1].shape)))
                self.weights[1] = self.weights[1] + alpha * hom * deltas[1]

                if self.error_test:
                    p = self.feedforward(x)
                    self.error_output.append((itern, ef(t, p),))
                
                if self.enable_gradient_check:
                    error = self.gradient_error_func(pendiete,deltas[1])
                    self.gradient_error.append((itern, np.asscalar(error)))
                    # self.gradient_error.append((itern, np.asscalar((error[1]))))

    def j(self, y, a):
        return np.power(y - a,2)*(-1/2*len(y))

    def best_alpha(self, training_set, test_data, iterations, alphas):
        results = []
        for e in alphas:
            self.backpropagation(training_set, e, iterations)
            err = self.evaluate(test_data)
            results.append((e, err,))
        return min(results, key=lambda et_err: et_err[1])
            

    def evaluate(self, test_data):
        targets = []
        preds = []
        for x, t in test_data:
            r = self.feedforward(x)
            preds.append(r)
            targets.append(t)
        
        preds = np.array(preds)
        targets = np.array(targets)
        return rmse(preds, targets)

def generate_points(number, func, point_size=1):
    points = []
    for x in range(number):
        p = rand_11(point_size)
        points.append((p, func(p),))
    return points


def predictions(iters, alpha, func, nin=1, nout=1, nhidden=10, activation=None, points=None, gcheck=False):
    """
    Create and execute a NeuralNetwork with 
    - # inputs: ninout
    - # outs  : ninout
    - # hidden: nhidden

    It create 40 points random on space of size: `ninout`.

    Run the NeuralNetowrk(ninout, nhidden, ninout) with `alpha` and
    `iters` number of iterations.

    Returns a pandas.DataFrame object with the predictions.
    """
    if not points:
        points = generate_points(40, func, point_size=nin)

    ann = NeuralNetwork(nin, nhidden, nout, activation=activation)
    ann.setup_errortest(error_func=mse)
    if gcheck:
        ann.setup_gradient_check(epsilon=10**-4)
    ann.backpropagation(points, alpha, iters)

    results = []
    for x, t in points:
        p = ann.feedforward(x)
        if nin > 1:
            results.append((tuple(x), tuple(t), tuple(p)))
        else:
            results.append((x.item(0), t.item(0), p.item(0)))
    values = pd.DataFrame(results, columns = ["x", "y", "prediction"])
    gerr = pd.DataFrame(ann.error_output, columns = ["Iterations", "MSE"])
    gradient_check = pd.DataFrame(ann.gradient_error, columns=['iter', 'ErrorGradiente'])
    return (values, gerr, gradient_check)




