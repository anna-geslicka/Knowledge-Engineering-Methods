import numpy as np


class Perceptron:

    def __init__(self, alpha=0.1, n_iter=10):
        self.errors = None
        self.weights = None
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, x, y):
        self.w = np.zeros(1 + x.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                delta = self.alpha * (target - self.predict(xi))
                self.weights[1:] += delta * xi
                self.weights[0] += delta
                errors += int(delta != 0.0)
            self.errors.append(errors)
        return self

    def net(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return np.where(self.net(x) >= 0.0, 1, -1)
