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

    def mod_weights(self, delta_weights, ratio=1):
        self.weights += delta_weights * ratio

    @property
    def inputs(self):
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        self.__inputs = np.append(1, inputs)


class Network:
    def __init__(self, width, input_num, mse=False, momentum=0.1):
        self.layers = [Layer(width[0], input_num)]
        self.input_num = input_num
        self.mse = mse
        self.momentum = momentum

        for d in range(len(width) - 1):
            self.layers.append(Layer(width[d + 1], width[d]))

    def add_layer(self, width):
        self.layers.append(Layer(width, len(self.layers[-1].weights)))

    def pop_layer(self):
        self.layers = self.layers[:-1]

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
            output = self.output(row)
            if self.mse:
                eff += ((exp - output) ** 2).mean(axis=None)
            else:
                result = max_index(output)
                if exp[result] == 1:
                    eff += 1
        return eff / len(test_set)

    def teach_row(self, input_row, desired_output, t_step, ignore_bottom):
        full_output = self.full_output(input_row)
        weights = self.layers[-1].weights_for_inputs()
        deltas = self.layers[-1].delta_output(desired_output, full_output[-1])
        delta_weights = [self.layers[-1].teach_row(deltas, t_step)]
        for l, out in zip(reversed(self.layers[ignore_bottom:-1]), reversed(full_output[ignore_bottom:-1])):
            deltas = l.delta_hidden(weights, deltas, out)
            delta_weights.append(l.teach_row(deltas, t_step))
            weights = l.weights_for_inputs()
        return delta_weights

    def teach_batch(self, t_batch, t_step, old_delta_weights, ignore_bottom):
        delta_weights = self.teach_row(t_batch[0][0], t_batch[0][1], t_step, ignore_bottom)
        for row in t_batch[1:]:
            new_dw = self.teach_row(row[0], row[1], t_step, ignore_bottom)
            delta_weights = [np.add(dw1, dw2) for dw1, dw2 in zip(delta_weights, new_dw)]
        delta_weights = [np.divide(dw, len(t_batch)) for dw in delta_weights]
        for l, w in zip(self.layers[ignore_bottom:], delta_weights[::-1]):
            l.mod_weights(w)
        for l, w in zip(self.layers[ignore_bottom:], old_delta_weights[::-1]):
            l.mod_weights(w, self.momentum)
        return delta_weights

    def teach(self, teaching_set, validating_set, batch_size, t_step, max_iter=1000, ignore_bottom=0, log_live=False):
        log = []
        for i in range(max_iter):
            rand.shuffle(teaching_set)
            batch_num = 0
            delta_weights = []
            for batch in [teaching_set[i:i + batch_size] for i in range(0, len(teaching_set), batch_size)]:
                delta_weights = self.teach_batch(batch, t_step, delta_weights, ignore_bottom)
                batch_num += 1
            t_set_eff = self.effectiveness(teaching_set)
            v_set_eff = self.effectiveness(validating_set)
            log.append((i, t_set_eff, v_set_eff))
            if log_live and i % 20 == 0:
                print(str(i) + ' t_set: ' + "{0:.3f}".format(t_set_eff) +
                      ' v_set: ' + "{0:.5f}".format(v_set_eff))
        return log, v_set_eff


def max_index(array):
    max = array[0]
    result = 0
    for i in range(len(array)):
        if array[i] > max:
            max = array[i]
            result = i
    return result
