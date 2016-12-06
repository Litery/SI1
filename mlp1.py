import random as rand
import math
import numpy as np


class Neuron:
    def __init__(self, **kwargs):
        self.weights = kwargs.get('weights', None)
        self.bias = kwargs.get('bias', 2)
        input_num = kwargs.get('input_num', 2)
        if self.weights is None:
            self.weights = []
            for i in range(input_num + 1):
                self.weights.append(rand.random())

    def output(self, input_row):
        result = 0
        for (x, w) in zip([1] + input_row, self.weights):
            result += x * w
        return 1 / (1 + math.exp(-result * self.bias))

    def delta(self, input_row, output_weights, output_deltas, desired_output):
        output = self.output(input_row)
        result = output * (1 - output) * self.bias
        if desired_output is not None:
            result *= desired_output - output
        else:
            result *= sum([w * d for (w, d) in zip(output_weights, output_deltas)])
        return result

    def teach(self, input_row, delta, t_step):
        delta_weights = []
        for i in [1] + input_row:
            dw = i * t_step * delta
            delta_weights.append(dw)
        return delta_weights


class Layer:
    def __init__(self, neuron_num, input_num):
        self.neurons = []
        for i in range(neuron_num):
            self.neurons.append(Neuron(input_num=input_num))
        self.inputs = None

    def output(self):
        return [n.output(self.inputs) for n in self.neurons]

    def weights(self):
        weights = np.array()

    def deltas(self, **kwargs):
        output_weights = kwargs.get('output_weights', None)
        output_deltas = kwargs.get('output_deltas', None)
        desired_output = kwargs.get('desired_output', None)
        return [n.delta(self.inputs, output_weights, output_deltas, desired_output) for n in self.neurons]

    def teach(self):
        pass
