import glob
import os
import numpy as np

from PIL import Image


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

make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')