import glob
import os
import random as rand
from PIL import Image
import mlp
import mlp1
import mlp2
import numpy as np


def make_teaching_set(path):
    os.chdir(path)
    t_set = dict()
    for x in range(10):
        t_set[x] = []
    for file in glob.glob(os.path.join(path, '*.png')):
        im = Image.open(file)
        pixels = list(im.getdata())
        pixels = [sum(x)/(255 * len(x)) for x in pixels]
        expected = int(os.path.basename(file)[0])
        t_set[expected].append(pixels)
    for i in range(10):
        rand.shuffle(t_set[i])
    return t_set


def make_teaching_set2(path):
    os.chdir(path)
    t_set = dict()
    for x in range(10):
        t_set[x] = []
    for file in glob.glob(os.path.join(path, '*.png')):
        im = Image.open(file)
        pixels = list(im.getdata())
        pixels = np.array([sum(x)/(255 * len(x)) for x in pixels])
        expected = int(os.path.basename(file)[0])
        t_set[expected].append(pixels)
    for i in range(10):
        rand.shuffle(t_set[i])
    return t_set


def teach_network():
    net = mlp.Network(2, [30, 10], 70)
    # t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    test_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.65 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.65 * len(value)):int(0.9 * len(value))]:
            v_set.append((x, expected))
        for x in value[int(0.9 * len(value)):]:
            test_set.append((x, expected))
    net.teach(t_set, v_set, None, 30, 10)
    print(net.effectiveness(test_set))


def teach_network1(hidden_size, t_step, batch_size):
    net = mlp1.Network(2, [hidden_size, 10], 70)
    # t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    test_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.85 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            v_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            test_set.append((x, expected))
    net.teach(t_set, v_set, batch_size, t_step, 500)
    return net.effectiveness(test_set)


def teach_network2(width, t_step, batch_size):
    net = mlp2.Network(width, 70)
    net.print()
    # t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    t_dict = make_teaching_set2('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    test_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.85 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            v_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            test_set.append((x, expected))
    net.teach(t_set, v_set, batch_size, t_step, 1000)
    return net.effectiveness(test_set)


def test1():
    results = []
    with open('results.txt', 'w', newline="\r\n") as o, open('results_sorted.txt', 'w', newline="\r\n") as o1:
        for batch_size in [50]:
            for h_size in [30]:
                for t_step in [0.6]:
                    result = teach_network1(h_size, t_step, batch_size)
                    results.append(([batch_size, h_size, t_step], result))
                    o.write(str(([batch_size, h_size, t_step], result)) + '\n')
        results = sorted(results, key=lambda result: result[1])
        print(results)
        for result in results:
            o1.write(str(result) + '\n')


def compare():
    n1 = mlp1.Network(2, [2, 2], 2)
    n2 = mlp2.Network([2, 2], 2)
    for l1, l2 in zip(n1.layers, n2.layers):
        w = np.array(l1.get_weights())
        l2.weights = w
    for l1, l2 in zip(n1.layers, n2.layers):
        print(l1.get_weights())
        print(l2.weights)
    print()
    print(n1.full_output([1, 0]))
    print(n2.full_output([1, 0]))
    print('mm')
    print(n1.teach_row([1, 0], [0, 1], 0.6))
    print(n2.teach_row([1, 0], [0, 1], 0.6))


teach_network2([20, 15, 10], 0.6, 100)
