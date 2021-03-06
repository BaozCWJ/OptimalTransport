import numpy as np
from dataset import *
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=float, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='Caffa')
parser.add_argument('--iters', type=int, default=15000)
parser.add_argument('--rho', type=float, default=1e3)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--is-tunning', action="store_true", default=False)
parser.add_argument('--draw', action="store_true", default=False)

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


def ADMM_primal(c, mu, nu, iters, rho, alpha, is_tunning):
    m, n = c.shape
    pi, pi_hat, e, lamda, eta = init(m, n)
    bigrho = rho * 1
    while bigrho >= rho:
        if args.draw:
            plt.figure()
            l = 1
        for j in range(iters):
            pi, pi_hat, e, lamda, eta = update(m, n, mu, nu, c, pi, pi_hat, e, lamda, eta, bigrho, alpha)
            if is_tunning and (j+1) % 1500 == 0:
                print('err1=', np.linalg.norm(pi_hat.sum(axis=1) - mu, 1),
                      'err2=', np.linalg.norm(pi_hat.sum(axis=0) - nu, 1),
                      'loss= ', (c * pi_hat).sum())
                if args.draw:
                    pi_plot = np.zeros_like(pi)
                    for k in range(pi.shape[0]):
                        pi_plot[k, pi[k, :].argsort()[-20:]] = 1

                    plt.subplot(2, 5, l)
                    l += 1
                    plt.imshow(pi_plot)
                    plt.title('iteration=' + str(j+1))
        if args.draw:
            plt.show()
        bigrho = bigrho / 10

    print('err1=', np.linalg.norm(pi_hat.sum(axis=1) - mu, 1),
          'err2=', np.linalg.norm(pi_hat.sum(axis=0) - nu, 1),
          'loss=', (c * pi_hat).sum())


if __name__ == '__main__':
    if args.data == 'DOTmark':
        mu, nu = DOTmark_Weight(args.n, args.image_class, )
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
    ADMM_primal(c, mu, nu, args.iters, args.rho, args.alpha, is_tunning=args.is_tunning)
    print('time usage=', time.time() - start)


