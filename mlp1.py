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
                self.weights.append(rand.random() * 2 - 1)

    def output(self, input_row):
        result = 0
        # result = sum(np.multiply([1] + input_row, self.weights))
        # result = sum([x * w for x, w in zip([1] + input_row, self.weights)])
        for x, w in zip([1] + input_row, self.weights):
            result += x * w
        return 1 / (1 + math.exp(-result * self.bias))

    def delta(self, output_weights, output_deltas, desired_output, output):
        result = output * (1 - output) * self.bias
        if desired_output is not None:
            result *= desired_output - output
        else:
            result *= sum([w * d for (w, d) in zip(output_weights, output_deltas)])
        return result

    def teach(self, input_row, delta, t_step):
        return [i * t_step * delta for i in [1] + input_row]


class Layer:
    def __init__(self, neuron_num, input_num):
        self.neurons = []
        for i in range(neuron_num):
            self.neurons.append(Neuron(input_num=input_num))
        self.inputs = None

    def output(self):
        return [n.output(self.inputs) for n in self.neurons]

    def weights_for_inputs(self):
        return np.array([n.weights[1:] for n in self.neurons]).transpose()

    def mod_weights(self, delta_weights):
        for n, dw in zip(self.neurons, delta_weights):
            n.weights = np.add(n.weights, dw)

    def deltas(self, **kwargs):
        output_weights = kwargs.get('output_weights', None)
        output_deltas = kwargs.get('output_deltas', None)
        desired_output = kwargs.get('desired_output', None)
        calc_output = kwargs.get('calc_output', None)
        if desired_output is None:
            result = [n.delta(weights, output_deltas, None, output) for n, weights, output
                     in zip(self.neurons, output_weights, calc_output)]
        else:
            result = [n.delta(None, None, desired, output) for n, desired, output
                      in zip(self.neurons, desired_output, calc_output)]
        return result

    def teach_row(self, deltas, t_step):
        return [n.teach(self.inputs, delta, t_step) for n, delta in zip(self.neurons, deltas)]


class Network:
    def __init__(self, depth, width, input_num):
        self.layers = [Layer(width[0], input_num)]
        self.input_num = input_num
        for d in range(depth - 1):
            self.layers.append(Layer(width[d + 1], width[d]))

    def output(self, input_row):
        for l in self.layers:
            l.inputs = input_row
            input_row = l.output()
        return input_row

    def full_output(self, input_row):
        self.layers[0].inputs = input_row
        result = [self.layers[0].output()]
        for l in self.layers[1:]:
            l.inputs = result[-1]
            result.append(l.output())
        return result

    def effectiveness(self, test_set):
        result = 0
        for (row, exp) in test_set:
            # result += sum([math.fabs(o - y) for o, y in zip(self.output(row), exp)]) / len(exp)
            my_list = self.output(row)
            max_value = max(my_list)
            max_index = my_list.index(max_value)
            if exp[max_index] == 1:
                result += 1
        # return 1 - result / len(test_set)
        return result / len(test_set)

    def teach_row(self, input_row, desired_output, t_step):
        full_output = self.full_output(input_row)
        weights = self.layers[-1].weights_for_inputs()
        deltas = self.layers[-1].deltas(desired_output=desired_output, calc_output=full_output[-1])
        delta_weights = [self.layers[-1].teach_row(deltas, t_step)]
        for l, out in zip(reversed(self.layers[:-1]), reversed(full_output[:-1])):
            deltas = l.deltas(output_weights=weights, output_deltas=deltas, calc_output=out)
            delta_weights.append(l.teach_row(deltas, t_step))
            weights = l.weights_for_inputs()
        return delta_weights

    def teach_batch(self, t_batch, t_step):
        delta_weights = self.teach_row(t_batch[0][0], t_batch[0][1], t_step)
        for row in t_batch[1:]:
            new_dw = self.teach_row(row[0], row[1], t_step)
            delta_weights = [np.add(dw1, dw2) for dw1, dw2 in zip(delta_weights, new_dw)]
        delta_weights = [np.divide(dw, len(t_batch)) for dw in delta_weights]
        for l, w in zip(self.layers, delta_weights[::-1]):
            l.mod_weights(w)

    def teach(self, teaching_set, validating_set, batch_size, t_step, max_iter=1000):
        for i in range(max_iter):
            rand.shuffle(teaching_set)
            batch_num = 0
            for batch in [teaching_set[i:i + batch_size] for i in range(0, len(teaching_set), batch_size)]:
                self.teach_batch(batch, t_step)
                batch_num += 1
            print(str(i) + ' tset ' + str(self.effectiveness(teaching_set)))
            print(str(i) + ' vset ' + str(self.effectiveness(validating_set)))


def test1():
    l = Layer(4, 2)
    print([n.weights for n in l.neurons])
    print(l.weights_for_inputs())
    l.inputs = [1, 0]
    print(l.deltas(desired_output=1))


def test2():
    n = Network(2, [4, 2], 2)
    for l in n.layers:
        print(l.weights_for_inputs())
    print(n.teach_row([1, 0], [0, 1], 1))


def test3():
    n = Network(2, [4, 2], 2)
    for l in n.layers:
        print(l.weights_for_inputs())
    n.teach_batch([[[1, 0], [0, 1]]], 1)
    print("after")
    for l in n.layers:
        print(l.weights_for_inputs())


def test4():
    t_set = [([0, 0], [0]), ([1, 1], [0]), ([0, 1], [1]), ([1, 0], [1])]
    net = Network(2, [3, 1], 2)
    net.teach(t_set, t_set, 4, 1)
    print(net.output([0, 0]))
    print(net.output([1, 1]))
    print(net.output([0, 1]))
    print(net.output([1, 0]))
