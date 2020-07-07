from network.neural_network import NeuralNetwork
from network.teach.teacher import Teacher

nn = NeuralNetwork()
nn.connect()

nn.run()
nn.inspect()

teacher = Teacher(nn)
teacher.teach()
for _ in range(100):
    teacher.teach()


nn.run()
nn.inspect()

nn.input_layer.set_values([0.9, 0.9, 0.1, 0.1])
nn.run()
nn.inspect()
