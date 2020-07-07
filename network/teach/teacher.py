from network.teach.patch import Patch


class Teacher:
    def __init__(self, neural_network):
        self.patch = Patch()
        self.neural_network = neural_network

    def teach(self):
        # Run Epoch
        #print("\nteach\n")
        self.patch.run(self.neural_network)
        #print("\n")