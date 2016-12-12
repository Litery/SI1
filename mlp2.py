import numpy as np
import random as rand


class Layer:
    def __init__(self, neuron_num, input_num):
        self.weights = np.random.rand(neuron_num, input_num + 1)
        self.weights = np.multiply(self.weights, 2)
        self.weights = np.subtract(self.weights, 1)
        self.__inputs = None

    def output(self):
        nets = np.multiply(self.weights, self.inputs)
        nets = np.sum(nets, 1)
        nets *= -1
        nets = np.exp(nets)
        nets += 1
        nets = np.reciprocal(nets)
        return nets

    def delta_output(self, desired_output, output):
        return output * (1 - output) * (desired_output - output)

    def delta_hidden(self, output_weights, output_deltas, output):
        return output * (1 - output) * np.sum(output_weights * output_deltas, 1)

    def weights_for_inputs(self):
        return self.weights[0:, 1:].transpose()

    def teach_row(self, deltas, t_step):
        return self.inputs * deltas.reshape(len(deltas), 1) * t_step

    def mod_weights(self, delta_weights):
        self.weights = self.weights + delta_weights

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        self.__inputs = np.append(1, inputs)


class Network:
    def __init__(self, width, input_num):
        self.layers = [Layer(width[0], input_num)]
        self.input_num = input_num
        for d in range(len(width) - 1):
            self.layers.append(Layer(width[d + 1], width[d]))

    def output(self, input_row):
        for l in self.layers:
            l.inputs = input_row
            input_row = l.output()
        return input_row

    def print(self):
        for l in self.layers:
            print(l.weights.shape)

    def full_output(self, input_row):
        self.layers[0].inputs = input_row
        result = [self.layers[0].output()]
        for l in self.layers[1:]:
            l.inputs = result[-1]
            result.append(l.output())
        return result

    def effectiveness(self, test_set):
        eff = 0
        for (row, exp) in test_set:
            result = max_index(self.output(row))
            if exp[result] == 1:
                eff += 1
        return eff / len(test_set)

    def teach_row(self, input_row, desired_output, t_step):
        full_output = self.full_output(input_row)
        weights = self.layers[-1].weights_for_inputs()
        deltas = self.layers[-1].delta_output(desired_output, full_output[-1])
        delta_weights = [self.layers[-1].teach_row(deltas, t_step)]
        for l, out in zip(reversed(self.layers[:-1]), reversed(full_output[:-1])):
            deltas = l.delta_hidden(weights, deltas, out)
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
            if i % 20 == 0:
                print(str(i) + ' tset: ' + "{0:.3f}".format(self.effectiveness(teaching_set)) +
                      ' vset: ' + "{0:.3f}".format(self.effectiveness(validating_set)))


def max_index(array):
    max = array[0]
    result = 0
    for i in range(len(array)):
        if array[i] > max:
            max = array[i]
            result = i
    return result


def test_d_out():
    l1 = Layer(3, 3)
    l1.inputs = np.array([1, 1, 0])
    print(l1.inputs)
    output = l1.output()
    print(output)
    desired = np.array([1, 0, 0])
    print(desired)
    print(l1.delta_output(desired, output))


def test_wfi():
    l1 = Layer(3, 3)
    print(l1.weights)
    print(l1.weights_for_inputs())


def test_d_hidden():
    l1 = Layer(3, 2)
    l2 = Layer(2, 2)
    l2.inputs = [0, 1]
    print(l2.inputs)
    print(l2.weights)
    out2 = l2.output()
    print(out2)
    print()
    l1.inputs = out2
    print(l1.weights)
    out1 = l1.output()
    print(out1)
    print()
    delta = l1.delta_output([1, 1, 0], out1)
    print(l1.weights_for_inputs())
    print(delta)
    print(l2.delta_hidden(l1.weights_for_inputs(), delta, out2))


def test_teach():
    l1 = Layer(3, 2)
    l1.inputs = [0.5, 0.4]
    out = l1.output()
    deltas = l1.delta_output([0, 0, 0], out)
    print(l1.weights)
    print(deltas)
    delta_weights = l1.teach_row(deltas, 0.6)
    print(delta_weights)
    print()
    print(l1.weights)
    l1.mod_weights(delta_weights)
    print(l1.weights)


def test_net1():
    n = Network([2, 2], 2)
    print(n.output([0, 1]))
    print(n.full_output([0, 1]))
