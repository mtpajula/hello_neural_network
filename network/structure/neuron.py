from network.activations.sigmoid_function import SigmoidFunction
from random import random


class Neuron:
    def __init__(self):
        self.backward_synapses = []
        self.forward_synapses = []
        self.activation_function = SigmoidFunction()
        self.value = 0
        self.delta = 0
        self.bias = 0.1  # random()

    # Forward propagation
    def fire(self):
        value = 0
        for synapse in self.backward_synapses:
            value += synapse.weight * synapse.from_neuron.value
        value += self.bias
        self.value = self.activation_function.activate(value)

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, expected):
        return 0.5 * (expected - self.value) ** 2

    def inspect(self):
        print("NEURON, inputs:", end="")
        print(len(self.backward_synapses), end="")
        print(", outputs:", end="")
        print(len(self.forward_synapses), end="")
        print(", value:", end="")
        print(self.value, end="")
        print(", delta:", end="")
        print(self.delta, end="")
        print(", bias:", end="")
        print(self.bias)
        # for s in self.forward_synapses:
        #     print("        OUTPUT SYNAPSE, weight:", s.weight)
