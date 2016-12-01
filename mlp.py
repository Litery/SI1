import random as rand
import math
import numpy as np


class Neuron:
    def __init__(self, **kwargs):
        self.weights = kwargs.get('weights', None)
        self.outputs = None
        self.__inputs = kwargs.get('inputs', None)
        self.last_output = None
        self.last_delta = None
        self.bias = kwargs.get('bias', 4)
        self.learn_rate = kwargs.get('learn_rate', 1)
        if self.weights is None:
            self.weights = []
            for i in range(len(self.inputs) + 1):
                self.weights.append(rand.random())
        self.delta_weights = np.zeros(len(self.weights))

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        self.__inputs = inputs
        self.reset()

    def reset(self):
        self.last_output = None
        self.last_delta = None
        if self.outputs is not None:
            for o in self.outputs:
                o.reset()

    def show(self):
        for i in self.weights:
            print(i)

    def net(self):
        result = 0
        for (x, w) in zip([1] + self.inputs, self.weights):
            result += x * w
        return result

    def output(self):
        if self.last_output is None:
            self.last_output = 1 / (1 + math.exp(-self.net() * self.bias))
        return self.last_output

    def get_weight(self, neuron):
        return self.weights[self.inputs.index(neuron) + 1]

    def __mul__(self, other):
        return self.output() * other

    def __add__(self, other):
        return self.output() + other

    def delta(self, correct=None):
        if self.last_delta is None:
            result = self.output() * (1 - self.output()) * self.bias
            if self.outputs is None:
                result *= correct - self.output()
            else:
                result *= sum([o.get_weight(self) * o.delta() for o in self.outputs])
            self.last_delta = result
        return self.last_delta

    def teach(self, correct):
        new_deltas = []
        for i in [1] + self.inputs:
            dw = i * self.learn_rate * self.delta(correct)
            new_deltas.append(dw)
        self.delta_weights = [dw + nd for dw, nd in zip(self.delta_weights, new_deltas)]

    def remember_epoch(self, epoch_size=1):
        self.weights = [w + dw/epoch_size for w, dw in zip(self.weights, self.delta_weights)]
        self.delta_weights = np.zeros(len(self.weights))


class Network:
    def __init__(self, depth, width, input_num):
        self.levels = None
        self.input_num = input_num
        for d in range(depth):
            self.add_level(width[d])

    def add_level(self, width):
        level = []
        if self.levels is not None:
            for w in range(width):
                level.append(Neuron(inputs=self.levels[-1]))
            self.levels[-1].outputs = level
        else:
            self.levels = []
            for w in range(width):
                level.append(Neuron(inputs=np.zeros(self.input_num)))

        self.levels.append(level)

    def output(self):
        return [n.output() for n in self.levels[-1]]

    def teach_row(self, input_row, expected):
        for n in self.levels[0]:
            n.inputs = input_row
        for level in self.levels:
            for neuron in level:
                neuron.teach(expected)


hidden = [Neuron(inputs=[0, 0]),
          Neuron(inputs=[0, 0]),
          Neuron(inputs=[0, 0])]

top = Neuron(inputs=hidden)

for n in hidden:
    n.outputs = [top]

for i in range(1000):

    for n in hidden:
        n.inputs = [0, 0]

    print("00 iter " + str(i) + "\t" + str(top.output()))

    top.teach(0)
    for n in hidden:
        n.teach(0)

    for n in hidden:
        n.inputs = [1, 0]

    print("10 iter " + str(i) + "\t" + str(top.output()))

    top.teach(1)
    for n in hidden:
        n.teach(1)

    for n in hidden:
        n.inputs = [1, 1]

    print("11 iter " + str(i) + "\t" + str(top.output()))

    top.teach(0)
    for n in hidden:
        n.teach(0)

    for n in hidden:
        n.inputs = [0, 1]

    print("01 iter " + str(i) + "\t" + str(top.output()))

    top.teach(1)
    for n in hidden:
        n.teach(1)

    top.remember_epoch(4)
    for n in hidden:
        n.remember_epoch(4)

for n in hidden:
    print(n.weights)
print()
print(top.weights)
