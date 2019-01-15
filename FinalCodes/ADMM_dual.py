import numpy as np
from dataset import *
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=float, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='random')
parser.add_argument('--iters', type=int, default=15000)
parser.add_argument('--rho', type=float, default=1e3)
parser.add_argument('--alpha', type=float, default=3)
parser.add_argument('--is-tunning', action="store_true", default=False)

args = parser.parse_args()


def init(c):
    m, n = c.shape
    lamda = np.zeros(m)
    eta = np.zeros(n)
    e = c - lamda.reshape((m, 1)) - eta.reshape((1, n))
    d = np.zeros((m, n))

    return lamda, eta, e, d


def update(m, n, mu, nu, c, lamda, eta, e, d, rho, alpha):
    eta_sigma = np.sum(eta)

    lamda = (
                    (mu + np.sum(d, axis=1)) / rho
                    - eta_sigma
                    - np.sum(e, axis=1)
                    + np.sum(c, axis=1)
            ) / n

    lamda_sigma = np.sum(lamda)

    eta = (
                  (nu + np.sum(d, axis=0)) / rho
                  - lamda_sigma
                  - np.sum(e, axis=0)
                  + np.sum(c, axis=0)
          ) / m

    e = d / rho + c - lamda.reshape((m, 1)) - eta.reshape((1, n))
    e = np.maximum(e, 0.)

    d = d + alpha * (c - lamda.reshape((m, 1)) - eta.reshape((1, n)) - e)

    return lamda, eta, e, d


def ADMM_dual(c, mu, nu, iters, rho, alpha, is_tunning=False):
    m, n = c.shape
    lamda, eta, e, d = init(c)
    for j in range(iters):
        lamda, eta, e, d = update(m, n, mu, nu, c, lamda, eta, e, d, rho, alpha)

        if is_tunning and j % 100 == 0:
            pi = -d  # ???
            print('err1=', np.linalg.norm(pi.sum(axis=1) - mu, 1),
                  'err2=', np.linalg.norm(pi.sum(axis=0) - nu, 1),
                  'loss= ', (c * pi).sum())

    print('err1=', np.linalg.norm(pi.sum(axis=1) - mu, 1),
          'err2=', np.linalg.norm(pi.sum(axis=0) - nu, 1),
          'loss= ', (c * pi).sum())


if __name__ == '__main__':
    if args.data == 'DOTmark':
        mu, nu = DOTmark_Weight(args.n, args.image_class)
        c = DOTmark_Cost(0, 1, 0, 1, args.n)
    elif args.data == 'random':
        mu = Random_Weight(int(args.n ** 2))
        nu = Random_Weight(int(args.n ** 2))
        c = Random_Cost(int(args.n ** 2))
    elif args.data == 'Caffa':
        mu = Const_Weight(int(args.n ** 2))
        nu = Const_Weight(int(args.n ** 2))
        c = Caffarelli_Cost(int(args.n ** 2), 0, 0, 1, 2)
    elif args.data == 'ellip':
        mu = Const_Weight(int(args.n ** 2))
        nu = Const_Weight(int(args.n ** 2))
        c = ellipse_Cost(int(args.n ** 2), 0, 0, 0.5, 2, 0.1)

    start = time.time()
    ADMM_dual(c, mu, nu, args.iters, args.rho, args.alpha, is_tunning=args.is_tunning)
    print('time usage=', time.time() - start)