import numpy as np


class LogisticRegressionGD:

    def __init__(self, alpha=0.05, n_iter=100, random_state=1):
        self.cost = None
        self.w = None
        self.alpha = alpha
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w[1:] += self.alpha * x.T.dot(errors)
            self.w[0] += self.alpha * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost.append(cost)
        return self

    def net_input(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def activation(self, z):  # probability based on net_input
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, 0)