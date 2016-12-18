import glob
import os
import random as rand
from PIL import Image
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
        pixels = np.array([1 - sum(x) / (255 * len(x)) for x in pixels])
        expected = int(os.path.basename(file)[0])
        t_set[expected].append(pixels)
    for i in range(10):
        rand.shuffle(t_set[i])
    return t_set


def teach_network(width, t_step, batch_size, momentum, t_iter, log=True):
    net = mlp2.Network(width, 70, momentum=momentum)
    t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    # t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.85 * len(value))]:
            t_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            v_set.append((x, expected))
    net.teach(t_set, v_set, batch_size, t_step, t_iter, log_live=log)
    return net, v_set


def teach_network_mean(width, t_step, batch_size, momentum, t_iter, sample_size):
    result = 0
    for i in range(sample_size):
        net, v_set = teach_network(width, t_step, batch_size, momentum, t_iter)[0]
        result += net.effectiveness(v_set)
    return result / sample_size


def teach_with_auto_mean(sample_size):
    result = 0
    for i in range(sample_size):
        net, v_set = teach_network_with_auto(True)
        result += net.effectiveness(v_set)
    return result / sample_size


def teach_auto_encoder(width, t_step, batch_size, max_iter):
    net = mlp2.Network(width, 70, mse=True, momentum=0.2)
    t_dict = make_teaching_set('/Users/szymon/Downloads/Sieci Neuronowe/')
    # t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    t_set = []
    v_set = []
    for (key, value) in t_dict.items():
        for x in value[:int(0.85 * len(value))]:
            t_set.append((x, x))
        for x in value[int(0.85 * len(value)):]:
            v_set.append((x, x))
    net.teach(t_set, v_set, batch_size, t_step, max_iter=max_iter, log_live=True)
    return net, v_set


def show_image(pixels):
    im = Image.fromarray((1 - np.reshape(pixels, (10, 7))) * 255)
    im = im.resize((70, 100))
    im.show()


def join_images(pixels1, pixels2):
    i1 = (1 - np.reshape(pixels1, (10, 7))) * 255
    i2 = (1 - np.reshape(pixels2, (10, 7))) * 255
    joined = np.hstack((i1, i2))
    im = Image.fromarray(joined)
    return im


def show_images(pixels1, pixels2):
    im = join_images(pixels1, pixels2)
    im = im.resize((140, 100))
    im.format = ".png"
    im.show()


def show_auto_encoder(hidden):
    net, v_set = teach_auto_encoder([hidden, 70], 0.7, 70, 500)
    rand.shuffle(v_set)
    for x in v_set:
        pixels = net.output(x[0])
        full = net.full_output(x[0])
        show_images(x[0], pixels)
        if 'end' == input():
            break
        most_active_index = mlp2.max_index(full[0])
        most_active = net.layers[0].weights[most_active_index]
        # most_active /= max(most_active)
        most_active = most_active.clip(min=0)
        print(most_active)
        print(pixels)
        show_image((most_active[1:] + pixels * 0.5) / 1.5)
        if 'end' == input():
            break


def show_network():
    net, v_set = teach_network([30, 10], 0.6, 60, 0.1, 600)
    rand.shuffle(v_set)
    for x in v_set:
        result = net.output(x[0])
        show_image(x[0])
        if 'end' == input(str(mlp2.max_index(result))):
            break


def save_auto_encoder():
    net, v_set = teach_auto_encoder([30, 70], 0.6, 70, 1000)
    os.chdir('C:/Users/Szon/Documents/Sieci Neuronowe/auto')
    for i in range(len(v_set)):
        pixels = net.output(v_set[i][0])
        im = join_images(v_set[i][0], pixels)
        im = im.resize((140, 100))
        im = im.convert('RGB')
        im.save('{:04d}'.format(i) + '.png', "PNG")


def teach_network_with_auto(log):
    net = mlp2.Network([30, 70], 70, mse=True)
    t_dict = make_teaching_set('C:/Users/Szon/Documents/Sieci Neuronowe')
    ta_set = []
    va_set = []
    t_set = []
    v_set = []
    for (key, value) in t_dict.items():
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected[key] = 1
        for x in value[:int(0.85 * len(value))]:
            ta_set.append((x, x))
            t_set.append((x, expected))
        for x in value[int(0.85 * len(value)):]:
            va_set.append((x, x))
            v_set.append((x, expected))
    print("Teaching AutoEncoder")
    net.print()
    net.teach(ta_set, va_set, 60, 0.8, 500, log_live=log)
    net.pop_layer()
    net.add_layer(10)
    net.mse = False
    print('Teaching top two layers')
    net.print()
    net.teach(t_set, v_set, 60, 0.8, 500, ignore_bottom=1, log_live=log)
    print('Full teaching')
    net.teach(t_set, v_set, 60, 0.6, log_live=log)
    return net, v_set


def test_hyper_par(output, t_iter, sample_size, width_range, batch_range, t_step_range, momentum_range):
    results = []
    for width in width_range:
        for batch_size in batch_range:
            for t_step in t_step_range:
                for momentum in momentum_range:
                    result = teach_network_mean(width, t_step, batch_size, momentum, t_iter, sample_size)
                    results.append(((batch_size, width, t_step, momentum), result))
                    print(results[-1])
                    output.write(str((width, batch_size, t_step, momentum, result)) + '\n')
                    # results = sorted(results, key=lambda result: result[1])


def get_hyper_results(t_iter, sample_size):
    t_step_range = np.append(0.05, np.arange(0.1, 1.3, 0.1))
    momentum_range = np.arange(0.05, 0.5, 0.05)
    width_range = [[10, 10], [15, 10], [20, 10], [25, 10], [30, 10], [35, 10], [40, 10], [45, 10], [50, 10], [60, 10]]
    with open('results_t_set_test.txt', 'w', newline="\r\n") as o1, \
            open('results_momentum_test.txt', 'w', newline="\r\n") as o2, \
            open('results_width_test.txt', 'w', newline="\r\n") as o3:
        # test_hyper_par(o1, t_iter, sample_size, [[30, 10]], [60], t_step_range, [0.1])
        # test_hyper_par(o2, t_iter, sample_size, [[30, 10]], [60], [0.6], momentum_range)
        test_hyper_par(o3, t_iter, sample_size, width_range, [60], [0.6], [0.1])


# get_hyper_results(600, 4)
# print(teach_with_auto_mean(10))
# teach_auto_encoder([30, 70], 0.6, 60, 400)
show_auto_encoder(10)
# teach_network([35, 10], 0.6, 60, 0.1, 500)
# show_auto_encoder()
