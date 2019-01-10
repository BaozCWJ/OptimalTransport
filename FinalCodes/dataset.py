import numpy as np
import csv
import os

'''
notation:
each image is n by n
m = n*n
'''

def Read_Image_Pair_from_DOTmark(n, ImageClass):
    path = os.getcwd()
    index = np.random.choice(range(1, 10), 2, replace=None)
    print(index)
    w = []
    for i in index:
        with open(path + '/dataset/' + ImageClass + '/data32_100' + str(i) + '.csv') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                w.append(row)
        csvfile.close()
    w = np.array(w, np.float32).reshape((2, n * n))
    v = w[0, :] / sum(w[0, :])
    u = w[1, :] / sum(w[1, :])

    return v, u


def EuclidCost(m, p):
    C = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            C[i, j] = (i / m - j / m) ** p + (i % m - j % m) ** p
    return C + C.T


def Random_Cost(m):
    # m = n*n
    return np.random.random((m, m))


def Random_Weight(m):
    # m = n*n
    v = np.random.random(m)
    return v / sum(v)



