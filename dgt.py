import glob
import os
import random as rand
from PIL import Image
import mlp
import mlp1
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
        pixels = [sum(x)/(255 * len(x)) for x in pixels]
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
    net = mlp1.Network(3, [hidden_size, 15, 10], 70)
    # t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    test_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.7 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.7 * len(value)):int(0.85 * len(value))]:
            v_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            test_set.append((x, expected))
    net.teach(t_set, v_set, batch_size, t_step, 1000)
    return net.effectiveness(test_set)


def test1():
    results = []
    with open('results.txt', 'w', newline="\r\n") as o, open('results_sorted.txt', 'w', newline="\r\n") as o1:
        for batch_size in [50]:
            for h_size in [25]:
                for t_step in [0.4, 0.5, 0.6]:
                    result = teach_network1(h_size, t_step, batch_size)
                    results.append(([batch_size, h_size, t_step], result))
                    o.write(str(([batch_size, h_size, t_step], result)) + '\n')
        results = sorted(results, key=lambda result: result[1])
        print(results)
        for result in results:
            o1.write(str(result) + '\n')


test1()
