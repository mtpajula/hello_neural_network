from network.teach.back_propagation import BackPropagation
from network.teach.sample import Sample


class Batch:
    """
    Stochastic gradient method?
    """

    def __init__(self):
        self.samples = []

    def add_sample(self, input_values, expected_values):
        self.samples.append(Sample(input_values, expected_values))

    def train(self, neural_network):
        for sample in self.samples:
            sample.fire(neural_network)
            sample.calculate_deltas(neural_network)
            BackPropagation.update_weights_and_biases(neural_network)

    def calculate_error(self, neural_network):
        error = 0
        for sample in self.samples:
            sample.fire(neural_network)
            error += sample.calculate_error(neural_network)
        return error
