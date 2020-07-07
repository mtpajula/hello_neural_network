from network.teach.single_run import SingleRun


class Patch:
    """
    Stochastic gradient method?
    """

    def __init__(self):
        self.single_runs = []
        self.single_runs.append(SingleRun([0.1, 0.0, 0.8, 1.0], [0.0, 1.0]))
        self.single_runs.append(SingleRun([0.2, 0.4, 0.5, 0.9], [0.0, 1.0]))
        self.single_runs.append(SingleRun([0.7, 0.5, 0.3, 0.2], [1.0, 0.0]))
        self.single_runs.append(SingleRun([0.9, 0.8, 0.2, 0.1], [1.0, 0.0]))

    def run(self, neural_network):
        for single_run in self.single_runs:
            single_run.run(neural_network)
        # self.cost(neural_network)