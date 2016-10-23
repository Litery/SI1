import random as rand


class Perceptron:

    def __init__(self, input_num):
        self.weights = []
        self.learn_step = 0.1
        for i in range(input_num + 1):
            self.weights.append(rand.random())

    def set_weights(self, weight_vector):
        self.weights = weight_vector

    def show(self):
        for i in self.weights:
            print(i)

    def net(self, input_vector):
        result = 0
        for (x, w) in zip([1] + list(input_vector), self.weights):
            result += x * w
        return result

    def output(self, input_vector):
        return 1 if self.net(input_vector) > 0 else 0

    def output_set(self, input_set):
        return [(x[-1], self.output(x)) for x in input_set]

    def teach(self, teaching_set, iterations):
        for i in range(iterations):
            correct = 0
            for row in teaching_set:
                input = row[:-1]
                e = row[-1] - self.output(input)
                if e == 0:
                    correct += 1
                self.weights = [w + x * self.learn_step * e for w, x in zip(self.weights, [1] + list(input))]
            print(self.weights)
            print("Efectiveness: " + str(correct/len(teaching_set)))
            if correct == len(teaching_set):
                break


def gen1():
    while True:
        (x, y) = (rand.random(), rand.random())
        yield (x, y, 1 if x + y > 1 else 0)


def take(n, gen):
    return list(next(gen) for _ in range(n))


p1 = Perceptron(2)
p1.show()

teaching_set = take(300, gen1())
p1.teach(teaching_set, 50)
