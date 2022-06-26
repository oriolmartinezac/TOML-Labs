import random
import numpy as np


class Network:
    def __init__(self, layers):
        self.last_layer = layers[-1]
        self.num_of_layers = len(layers)
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(1, x) for x in layers[1:]]

    def feedforward(self, X, exc=False):
        X = X.reshape(1, X.shape[0])
        for w, b in zip(self.weights, self.biases):
            z = (w.dot(X.T)).transpose() + b
            X = sigmoid(z)
        return X.reshape((self.last_layer)) if exc else X

    def SGD(self, training_data, batch_size, epoch, eta):
        '''
        Details:    Stochastic Gradient Descent on Training Data
        Parmeters:  Training Data of type list(tuple(list,list)),Batch Size, Epoch,Learning Rate(eta)
        '''
        # print(self.weights)
        n = len(training_data)
        for i in range(epoch):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_batch(mini_batch, eta)

    def update_batch(self, batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for (x, y) in batch:
            x = np.array(x)
            y = np.array(y)
            del_nabla_w, del_nabla_b = self.backpropagate(x.reshape(1, x.shape[0]), y.reshape(1, y.shape[0]))
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, del_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, del_nabla_b)]
        self.weights = [w - (eta / len(batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagate(self, x, y):
        del_nabla_w = [np.zeros(w.shape) for w in self.weights]
        del_nabla_b = [np.zeros(b.shape) for b in self.biases]
        zs = []
        activations = [x]
        activation = x
        for w, b in zip(self.weights, self.biases):
            z = (w.dot(activation.T)).T + b
            activation = sigmoid(z)
            zs.append(z)
            activations.append(activation)
        delta = (activations[-1] - y) * sig_prime(zs[-1])
        del_nabla_b[-1] = delta
        del_nabla_w[-1] = delta.T.dot(activations[-2])
        for n in range(2, self.num_of_layers):
            try:
                delta = delta.dot(self.weights[-n + 1]) * sig_prime(zs[-n])
                del_nabla_b[-n] = delta
                del_nabla_w[-n] = delta.T.dot(activations[-n - 1])
            except Exception as e:
                print(e)
                print('\n', n)
                exit()
        return del_nabla_w, del_nabla_b


def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))


def sig_prime(X):
    return sigmoid(X) * sigmoid(1 - X)