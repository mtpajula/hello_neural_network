from network.structure.neuron import Neuron
from network.structure.synapse import Synapse
import enum


class LayerType(enum.Enum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class NeuronsLayer:
    def __init__(self, neurons, layer_type):
        self.layer_type = layer_type
        self.neurons = []
        for i in range(neurons):
            self.neurons.append(Neuron())

    def connect_to_layer(self, neurons_layer):
        for to_neuron in neurons_layer.neurons:
            for from_neuron in self.neurons:
                synapse = Synapse(from_neuron, to_neuron)
                from_neuron.forward_synapses.append(synapse)
                to_neuron.backward_synapses.append(synapse)

    def inspect(self):
        print("LAYER", self.layer_type)
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
