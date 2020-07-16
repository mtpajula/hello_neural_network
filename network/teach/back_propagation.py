from network.structure.neurons_layer import LayerType


class BackPropagation:

    # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
    @staticmethod
    def output_layer_delta(neuron, expected):
        neuron.delta = (expected - neuron.value) * neuron.activation_function.derivate(neuron.value)

    @staticmethod
    def back_propagated_delta(neuron):
        neuron.delta = 0
        for synapse in neuron.forward_synapses:
            neuron.delta += (synapse.weight * synapse.to_neuron.delta) * neuron.activation_function.derivate(neuron.value)

    @staticmethod
    def update_weights(neuron):
        for synapse in neuron.backward_synapses:
            # old = synapse.weight
            synapse.weight += 0.5 * neuron.delta * synapse.from_neuron.value
            # print("old weight", old, "new weight", synapse.weight)

    @staticmethod
    def update_bias(neuron):
        neuron.bias += 0.5 * neuron.delta

    @staticmethod
    def update_weights_and_biases(neural_network):
        for layer in neural_network.layers:
            if layer.layer_type != LayerType.INPUT:
                for neuron in layer.neurons:
                    BackPropagation.update_weights(neuron)
                    BackPropagation.update_bias(neuron)
