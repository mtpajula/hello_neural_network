from network.structure.neurons_layer import LayerType
from network.teach.back_propagation import BackPropagation


class Sample:
    def __init__(self, input_values, expected_values):
        self.input_values = input_values
        self.expected_values = expected_values

    def calculate_deltas(self, neural_network):
        for layer in reversed(neural_network.layers):
            if layer.layer_type == LayerType.OUTPUT:
                for i, neuron in enumerate(layer.neurons):
                    BackPropagation.output_layer_delta(neuron, self.expected_values[i])
            else:
                for neuron in layer.neurons:
                    BackPropagation.back_propagated_delta(neuron)

    def calculate_error(self, neural_network):
        error = 0
        # print("")
        for i, neuron in enumerate(neural_network.output_layer.neurons):
            error += neuron.calculate_error(self.expected_values[i])
            # print('\texpected', self.expected_values[i], 'value', neuron.value, 'error', error)
        # print('\tsample error', total_error)
        return error

    def fire(self, neural_network):
        neural_network.input_layer.set_values(self.input_values)
        neural_network.fire()

