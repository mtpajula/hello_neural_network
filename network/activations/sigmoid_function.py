import numpy as np


class SigmoidFunction:
    def __init__(self):
        self.limit = 0

    def activate(self, value):
        # Sigmoid?
        return 1 / (1 + np.exp(-value))
        # Rectified
        # return max(0.0, value)

    # How Deep Neural Networks Work
    # https://www.youtube.com/watch?v=ILsA4nyG7I0
    # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    def derivate(self, value):
        return value * (1.0 - value)
