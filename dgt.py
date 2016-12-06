import glob
import os

from PIL import Image

import mlp


def make_teaching_set(path):
    os.chdir(path)
    t_set = dict()
    for x in range(10):
        t_set[x] = []
    for file in glob.glob(os.path.join(path, '*.png')):
        im = Image.open(file)
        pixels = list(im.getdata())
        pixels = [1 if x == (255, 255, 255) else 0 for x in pixels]
        expected = int(os.path.basename(file)[0])

        t_set[expected].append(pixels)
    return t_set


def teach_network():
    net = mlp.Network(2, [40, 10], 70)
    t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    t_set = []
    v_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.8 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.8 * len(value)):]:
            v_set.append((x, expected))
    print(t_set)

    net.teach(t_set, v_set, None, 50, 100)


teach_network()
