from network.teach.batch import Batch
from network.teach.sample import Sample


class Trainer:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.batches = []

    def test(self, input_values, expected_values):
        sample = Sample(input_values, expected_values)
        sample.fire(self.neural_network)
        print('test sample error', sample.calculate_error(self.neural_network), 'expected', expected_values, 'output', self.neural_network.results())

    def train(self, rounds):
        print('training batches', len(self.batches), 'rounds', rounds)
        d = rounds / 50
        for b, batch in enumerate(self.batches):
            print('batch', b, end=': ')
            for i in range(rounds):
                batch.train(self.neural_network)
                if (i % d) == 0:
                    print('.', end='')
            print(' error: ', batch.calculate_error(self.neural_network))
