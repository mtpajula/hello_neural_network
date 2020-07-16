from network.neural_network import NeuralNetwork
from network.teach.batch import Batch
from network.teach.trainer import Trainer

nn = NeuralNetwork()
nn.add_input_layer(4)
nn.add_hidden_layer(4)
nn.add_hidden_layer(4)
nn.add_output_layer(2)

nn.construct()

nn.inspect()

t = Trainer(nn)
b1 = Batch()
b1.add_sample([0.1, 0.0, 0.8, 1.0], [0.0, 1.0])
b1.add_sample([0.2, 0.4, 0.5, 0.9], [0.0, 1.0])
b1.add_sample([0.7, 0.5, 0.3, 0.2], [1.0, 0.0])
b1.add_sample([0.9, 0.8, 0.2, 0.1], [1.0, 0.0])
t.batches.append(b1)

b2 = Batch()
b2.add_sample([0.9, 0.2, 0.2, 1.0], [0.0, 0.0])
b2.add_sample([1.0, 0.3, 0.1, 0.9], [0.0, 0.0])
b2.add_sample([0.1, 1.0, 1.0, 0.1], [0.0, 0.0])
b2.add_sample([0.2, 0.8, 0.9, 0.2], [0.0, 0.0])
t.batches.append(b2)

t.train(1000)
t.train(1000)

t.test([0.0, 0.1, 0.9, 0.7], [0.0, 1.0])
t.test([0.9, 0.1, 0.0, 0.7], [0.0, 0.0])

print('Run: ', nn.run([0.1, 0.1, 0.1, 0.0]))
print('Run: ', nn.run([0.1, 0.8, 0.9, 0.2]))
print('Run: ', nn.run([0.0, 0.1, 0.9, 0.7]))
