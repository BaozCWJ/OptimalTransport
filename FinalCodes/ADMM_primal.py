import numpy as np
from dataset import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='Caffa')
parser.add_argument('--iters', type=int, default=15000)
parser.add_argument('--rho', type=float, default=1e3)
parser.add_argument('--alpha', type=float, default=1)

args = parser.parse_args()


def init(m, n):
    pi = np.zeros((m, n))
    pi_hat = np.zeros((m, n))
    e = np.zeros((m, n))
    lamda = np.zeros(m)
    eta = np.zeros(n)
    return pi, pi_hat, e, lamda, eta


def update(m, n, mu, nu, c, pi, pi_hat, e, lamda, eta, rho, alpha):
    r = (
            (-e + lamda.reshape((m, 1)) + eta.reshape((1, n)) - c) / rho
            + mu.reshape((m, 1))
            + nu.reshape((1, n))
            + pi_hat
    )
    pi = (
            r
            - ((r.sum(axis=1) - r.sum() / (m + n + 1)) / (n + 1)).reshape((m, 1))
            - ((r.sum(axis=0) - r.sum() / (m + n + 1)) / (m + 1)).reshape((1, n))
    )

    pi_hat = np.maximum(pi + e / rho, 0.)

    lamda = lamda + alpha * rho * (mu - pi.sum(axis=1))

    eta = eta + alpha * rho * (nu - pi.sum(axis=0))

    e = e + alpha * rho * (pi - pi_hat)

    return pi, pi_hat, e, lamda, eta


def ADMM_primal(c, mu, nu, iters, rho, alpha):
    m, n = c.shape
    pi, pi_hat, e, lamda, eta = init(m, n)
    bigrho = rho * 1
    while bigrho >= rho:
        for j in range(iters):
            pi, pi_hat, e, lamda, eta = update(m, n, mu, nu, c, pi, pi_hat, e, lamda, eta, bigrho, alpha)
            if j % 100 == 0:
                print('err1=', np.linalg.norm(pi_hat.sum(axis=1) - mu, 1),
                      'err2=', np.linalg.norm(pi_hat.sum(axis=0) - nu, 1),
                      'loss= ', (c * pi_hat).sum())
        bigrho = bigrho / 10


if __name__ == '__main__':
    if args.data == 'DOTmark':
        mu, nu = DOTmark_Weight(args.n, args.image_class)
        c = DOTmark_Cost(0, 1, 0, 1, args.n)
    elif args.data == 'random':
        mu = Random_Weight(args.n ** 2)
        nu = Random_Weight(args.n ** 2)
        c = Random_Cost(args.n ** 2)
    elif args.data == 'Caffa':
        mu = Const_Weight(args.n ** 2)
        nu = Const_Weight(args.n ** 2)
        c = Caffarelli_Cost(args.n ** 2, 0, 0, 1, 2)
    elif args.data == 'ellip':
        mu = Const_Weight(args.n ** 2)
        nu = Const_Weight(args.n ** 2)
        c = ellipse_Cost(args.n ** 2, 0, 0, 0.5, 2, 0.1)
    ADMM_primal(c, mu, nu, args.iters, args.rho, args.alpha)
