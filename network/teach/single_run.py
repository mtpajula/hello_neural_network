

class SingleRun:
    def __init__(self, input_values, expected_values):
        self.input_values = input_values
        self.expected_values = expected_values
        # self.output_values = []
        self.cost_values = []

    def run(self, neural_network):
        neural_network.input_layer.set_values(self.input_values)
        neural_network.run()

        # Calculate deltas
        for i, neuron in enumerate(neural_network.output_layer.neurons):
            neuron.output_error(self.expected_values[i])

        for i, neuron in enumerate(neural_network.hidden_layers[0].neurons):
            neuron.back_propagated_error()

        for i, neuron in enumerate(neural_network.input_layer.neurons):
            neuron.back_propagated_error()

        # update weights
        for i, neuron in enumerate(neural_network.hidden_layers[0].neurons):
            neuron.update_weights()

        for i, neuron in enumerate(neural_network.output_layer.neurons):
            neuron.update_weights()
