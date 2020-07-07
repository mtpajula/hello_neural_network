from random import random


class NeuronsConnection:
    def __init__(self, from_neuron, to_neuron):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random()
