import random as rand
import math
import numpy as np


class Neuron:
    def __init__(self, **kwargs):
        self.weights = kwargs.get('weights', [])
        self.inputs = kwargs.get('inputs', None)
        self.outputs = None
        self.last_output = None
        self.bias = kwargs.get('bias', 8)
        self.learn_rate = kwargs.get('learn_rate', 1)
        for i in range(len(self.inputs) + 1):
            self.weights.append(rand.random() * 2 - 1)

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

    def get_weight(self, input):
        return self.weights[self.inputs.index(input)]

    def __mul__(self, other):
        return self.output() * other

    def __add__(self, other):
        return self.output() + other

    def delta(self, correct):
        result = self.output() * (1 - self.output())
        if self.outputs is None:
            result *= correct - self.output()
        else:
            result *= sum([o.get_weight(self) * o.delta(correct) for o in self.outputs])
        return result

    def teach(self, correct):
        new_weights = []
        for w, i in zip(self.weights, [1] + self.inputs):
            nw = w + i * self.learn_rate * self.delta(correct)
            new_weights.append(nw)
        self.weights = new_weights
        self.last_output = None


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

    def teach(self, row):
        pass


hidden = []
for n in range(5):
    hidden.append(Neuron(inputs=[1, 0]))

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
