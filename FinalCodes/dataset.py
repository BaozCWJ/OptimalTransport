import math
import numpy as np
import csv
import os

'''
notation:
each image is n by n
num = n*n
'''


def Const_Weight(num):
    return np.ones(num) / num


def Random_Weight(num):
    # num = n*n
    v = np.random.random(num)
    return v / sum(v)



def Euclid_Cost(mu_position, nu_position):
    m, n = mu_position.shape[0], nu_position.shape[0]
    ind = np.indices((m, n))
    c = ((mu_position[ind[0]] - nu_position[ind[1]]) ** 2).sum(axis=2)
    return c


def Random_position(num, scalar):
    return scalar * (np.random.rand(num * 2).reshape((num, 2)))


def Random_Cost(m, scalar=1):
    mu_position = Random_position(m, scalar)
    nu_position = Random_position(m, scalar)
    return Euclid_Cost(mu_position, nu_position)


def DOTmark_Weight(num, ImageClass):
    path = os.getcwd()
    # index = np.random.choice(range(1, 10), 2, replace=None)
    index = [1, 2]
    w = []
    for i in index:
        with open(path + '/dataset/' + ImageClass + '/data32_100' + str(i) + '.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                w.append(row)
        csvfile.close()
    w = np.array(w, np.float32).reshape((2, num ** 2))
    v = w[0, :] / sum(w[0, :])
    u = w[1, :] / sum(w[1, :])
    return v, u


def DOTmark_position(start_x, end_x, start_y, end_y, size):
    num = [size, size]
    num_t = num[0] * num[1]
    step_x = (end_x - start_x) / num[0]
    x = np.linspace(start_x + step_x / 2., end_x - step_x / 2., num[0])
    step_y = (end_y - start_y) / num[1]
    y = np.linspace(start_y + step_y / 2., end_y - step_y / 2., num[1])
    xp, yp = np.meshgrid(x, y)  # make grid
    p = np.concatenate((xp.reshape((num_t, 1)), yp.reshape((num_t, 1))), axis=1)
    return p


def DOTmark_Cost(start_x, end_x, start_y, end_y, size):
    mu_position = DOTmark_position(start_x, end_x, start_y, end_y, size)
    nu_position = DOTmark_position(start_x, end_x, start_y, end_y, size)
    return Euclid_Cost(mu_position, nu_position)


def Caffarelli_position(num, x_center, y_center, r, d):
    l = 0
    while l <= num:
        ox = np.random.uniform(-r, r, num)
        oy = np.random.uniform(-r, r, num)
        mask = ox ** 2 + oy ** 2 < r ** 2
        dx, dy = ox[mask], oy[mask]
        dx[dx < 0.] -= d
        dx[dx >= 0.] += d
        n = dx.size
        x = dx + x_center
        y = dy + y_center
        p = np.concatenate((x.reshape(n, 1), y.reshape(n, 1)), axis=1)
        if l == 0:
            pos = p
        else:
            pos = np.concatenate((pos, p), axis=0)
        l = len(pos)
    return pos[0:num, ]


def Caffarelli_Cost(num, x_center, y_center, r, d):
    mu_position = Caffarelli_position(num, x_center, y_center, r, 0)
    nu_position = Caffarelli_position(num, x_center, y_center, r, d)
    return Euclid_Cost(mu_position, nu_position)


def ellipse_position(num, x_center, y_center, r_x, r_y, eps):
    r = np.random.uniform(0, 2. * math.pi, num)
    dx = np.cos(r) + eps / math.sqrt(2.) * np.random.randn(num)
    dy = np.sin(r) + eps / math.sqrt(2.) * np.random.randn(num)
    x = r_x * dx + x_center
    y = r_y * dy + y_center
    p = np.concatenate((x.reshape(num, 1), y.reshape(num, 1)), axis=1)
    return p


def ellipse_Cost(num, x_center, y_center, r_x, r_y, eps):
    mu_position = ellipse_position(num, x_center, y_center, r_x, r_y, eps)
    nu_position = ellipse_position(num, x_center, y_center, r_y, r_x, eps)
    return Euclid_Cost(mu_position, nu_position)
