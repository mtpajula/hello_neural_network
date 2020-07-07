from network.structure.neuron import Neuron
from network.structure.neurons_connection import NeuronsConnection


class NeuronsLayer:
    def __init__(self, neurons):
        self.neurons = []
        for i in range(neurons):
            self.neurons.append(Neuron())

    def connect_to_layer(self, neurons_layer):
        for to_neuron in neurons_layer.neurons:
            for from_neuron in self.neurons:
                neurons_connection = NeuronsConnection(from_neuron, to_neuron)
                from_neuron.output_connections.append(neurons_connection)
                to_neuron.input_connections.append(neurons_connection)

    def inspect(self):
        print("LAYER")
        for neuron in self.neurons:
            print("    ", end="")
            neuron.inspect()

    def set_values(self, values):
        for i, value in enumerate(values):
            self.neurons[i].value = value

    def get_values(self):
        values = []
        for neuron in self.neurons:
            values.append(neuron.value)
        return values

    # Forward propagation
    def fire(self):
        for neuron in self.neurons:
            neuron.fire()
