from network.structure.neurons_layer import NeuronsLayer, LayerType


# But what is a Neural Network? | Deep learning, chapter 1
# https://www.youtube.com/watch?v=aircAruvnKk
class NeuralNetwork:
    def __init__(self):
        self.input_layer = None
        self.output_layer = None
        self.hidden_layers = []
        self.layers = []

    def add_input_layer(self, neurons):
        self.input_layer = NeuronsLayer(neurons, LayerType.INPUT)

    def add_output_layer(self, neurons):
        self.output_layer = NeuronsLayer(neurons, LayerType.OUTPUT)

    def add_hidden_layer(self, neurons):
        layer = NeuronsLayer(neurons, LayerType.HIDDEN)
        self.hidden_layers.append(layer)

    def construct(self):
        # All layers -list
        self.layers.append(self.input_layer)
        self.layers.extend(self.hidden_layers)
        self.layers.append(self.output_layer)

        # Connect layer neurons with synapses
        for i in range(0, len(self.layers)-1):
            # print(i, self.layers[i].layer_type)
            self.layers[i].connect_to_layer(self.layers[i+1])

    def inspect(self):
        print("")
        for layer in self.layers:
            layer.inspect()
        print("")

    # Forward propagation
    def fire(self):
        for layer in self.hidden_layers:
            layer.fire()
        self.output_layer.fire()

    def results(self):
        return self.output_layer.get_values()

    def set_input(self, input_values):
        self.input_layer.set_values(input_values)

    def run(self, input_values):
        self.set_input(input_values)
        self.fire()
        return self.results()
