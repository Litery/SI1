import random as rand
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, **kwargs):
        self.weights = kwargs.get("weight_vector", None)
        self.learn_step = kwargs.get("learn_step", 0.5)
        self.draw = kwargs.get("draw", False)
        self.log = kwargs.get("log", False)
        self.adaline = kwargs.get("adaline", False)
        self.theta = kwargs.get("theta", 0)
        self.bipolar = kwargs.get("bipolar", False)
        input_num = kwargs.get("input_num", 2)
        self.teaching_iterations = 0
        if self.weights is None:
            self.weights = []
            for i in range(input_num + 1):
                self.weights.append(rand.random() * 2 - 1)

    def show(self):
        for i in self.weights:
            print(i)

    def net(self, input_vector):
        result = 0
        for (x, w) in zip([1] + list(input_vector), self.weights):
            result += x * w
        return result

    def output(self, input_vector):
        result = self.net(input_vector)
        return 1 if result > self.theta else -1 if self.bipolar else 0

    def output_set(self, input_set):
        return [(x[-1], self.output(x)) for x in input_set]

    def error(self, input_vector, result):
        if self.adaline:
            err = self.error_adaline(input_vector, result)
        else:
            err = result - self.output(input_vector)
        return err

    def error_adaline(self, input_vector, result):
        net = self.net(input_vector)
        out = self.output(input_vector)
        return 0 if out is result else result - net

    def effectiveness(self, input_set):
        eff = 0
        for (x, y) in self.output_set(input_set):
            if x == y:
                eff += 1
        return eff / len(input_set)

    def teach(self, teaching_set, iterations):
        if self.draw:
            plt.hold(True)
            self.print_set(teaching_set)
        if self.log:
            print(self.weights)
        for i in range(iterations):
            correct = 0

            for row in teaching_set:
                input_vector = row[:-1]
                e = self.error(input_vector, row[-1])
                if e is 0:
                    correct += 1
                new_weights = []
                for w, x in zip(self.weights, [1] + list(input_vector)):
                    new_weights.append(w + x * self.learn_step * e)
                self.weights = new_weights
            if self.log:
                print(self.weights)
                print("Effectiveness: " + str(correct / len(teaching_set)))
            if correct / len(teaching_set) > 0.99:
                break
            if self.draw:
                self.draw_function()
        self.teaching_iterations = i
        if self.draw:
            plt.show()

    def draw_function(self):
        [w0, w1, w2] = self.weights
        a = -w1 / w2
        b = -w0 / w2
        x = np.arange(0, 1.1, 0.1)
        plt.plot(x, a * x + b)

    def print_set(self, t_set):
        plt.plot([x for (x, y, z) in t_set if z is 1], [y for (x, y, z) in t_set if z is 1], 'go')
        plt.plot([x for (x, y, z) in t_set if z is not 1], [y for (x, y, z) in t_set if z is not 1], 'ro')


def gen1(bipolar=False):
    while True:
        (x, y) = (rand.random(), rand.random())
        yield (x, y, 1 if x + y > 1 else -1 if bipolar else 0)


def take(n, gen):
    return list(next(gen) for _ in range(n))


def test1(gens=150, max_iter=500, tset_size=90, test_set=None, **kwargs):
    test = []
    for i in range(gens):
        test.append(Perceptron(**kwargs))

    # teaching_set = take(tset_size, gen1())
    avg_iter = 0
    avg_eff = 0
    for perc in test:
        perc.teach(take(tset_size, gen1()), max_iter)
        avg_iter += perc.teaching_iterations
        if test_set is not None:
            avg_eff += perc.effectiveness(test_set)
    avg_iter /= len(test)
    if test_set is not None:
        avg_eff /= len(test)
    return [avg_iter, avg_eff]


def test_step(adaline=False, bipolar=False):
    for i in np.arange(0.1, 1.1, 0.1):
        print("Step: " + str(i) + " " + str(
            test1(learn_step=i, gens=50, max_iter=150, adaline=adaline, bipolar=bipolar)[0]))


def test_tset_size():
    test_set1 = take(1000, gen1())
    for i in range(10, 311, 50):
        print("Tset size: " + str(i))
        results = test1(step=0.5, log=False, gens=100, max_iter=100, tset_size=i, test_set=test_set1, adaline=True)
        print("Avg_iter: " + str(results[0]) + " Avg_eff: " + str(results[1]))


def test_basic():
    p1 = Perceptron(draw=True, log=True, learn_step=0.5, weight_vector=[0, -1, 1], bipolar=True)
    tset = take(90, gen1(True))
    p1.teach(tset, 50)


test_basic()
