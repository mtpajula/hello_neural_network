from network.activations.sigmoid_function import SigmoidFunction


class Neuron:
    def __init__(self):
        self.input_connections = []
        self.output_connections = []
        self.value = 0
        self.activation_function = SigmoidFunction()
        self.delta = 0

    # Forward propagation
    def fire(self):
        value = 0
        for neurons_connection in self.input_connections:
            value += neurons_connection.weight * neurons_connection.from_neuron.value
            # print(neurons_connection.weight, ' * ', neurons_connection.from_neuron.value, end=" ")
        # print('= ', value, end=" ")
        self.value = self.activation_function.activate(value)
        # print('> ', self.value)

    # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    def output_error(self, expected):
        self.delta = (expected - self.value) * self.activation_function.derivate(self.value)

    def back_propagated_error(self):
        self.delta = 0
        for connection in self.output_connections:
            self.delta += (connection.weight * connection.to_neuron.delta) * self.activation_function.derivate(self.value)

    def update_weights(self):
        for connection in self.input_connections:
            # old = connection.weight
            connection.weight += 0.5 * self.delta * connection.from_neuron.value
            # print("old weight", old, "new weight", connection.weight)

    def inspect(self):
        print("NEURON, inputs:", end="")
        print(len(self.input_connections), end="")
        print(", outputs:", end="")
        print(len(self.output_connections), end="")
        print(", value:", end="")
        print(self.value, end="")
        print(", delta:", end="")
        print(self.delta)
        for s in self.output_connections:
            print("        OUTPUT SYNAPSE, weight:", s.weight)
