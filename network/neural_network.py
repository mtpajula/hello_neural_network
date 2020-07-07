from network.structure.neurons_layer import NeuronsLayer


# But what is a Neural Network? | Deep learning, chapter 1
# https://www.youtube.com/watch?v=aircAruvnKk&t=1024s
class NeuralNetwork:
    def __init__(self):
        self.input_layer = NeuronsLayer(4)
        self.hidden_layers = []
        self.hidden_layers.append(NeuronsLayer(4))
        self.output_layer = NeuronsLayer(2)

    def connect(self):
        self.input_layer.connect_to_layer(self.hidden_layers[0])
        self.hidden_layers[0].connect_to_layer(self.output_layer)

    def inspect(self):
        print("")
        self.input_layer.inspect()
        self.hidden_layers[0].inspect()
        self.output_layer.inspect()
        print("")

    # Forward propagation
    def run(self):
        self.hidden_layers[0].fire()
        self.output_layer.fire()
