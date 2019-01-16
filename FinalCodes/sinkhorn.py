import argparse
import time
from dataset import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=float, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='DOTmark')
parser.add_argument('--iters', type=int, default=3000)
parser.add_argument('--eps', type=float, default=4e-4)
parser.add_argument('--eps-iters', type=int, default=1)
parser.add_argument('--is-tunning', action="store_true", default=False)

args = parser.parse_args()


def Sinkhorn(c, a, b, iters, eps, eps_iters, is_tunning):
    # a,b 是边缘分布
    m = len(c)  # m = n*n
    v = np.ones(m)
    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        plt.figure()
        l = 1
        for j in range(iters):
            u = a / np.dot(K, v)
            v = b / np.dot(K.T, u)

            pi = np.diag(u).dot(K).dot(np.diag(v))
            if is_tunning and (j+1) % 300 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
                      'loss=', (c * pi).sum(),
                      'loss with entropy=', (c * pi + eps * pi * np.log(pi)).sum())

                pi_plot = np.zeros_like(pi)
                for k in range(pi.shape[0]):
                    pi_plot[k, pi[k, :].argsort()[-20:]] = 1

                plt.subplot(2, 5, l)
                l += 1
                plt.imshow(pi_plot)
                plt.title('iteration=' + str(j+1))

        eps = eps / 10
        plt.show()

    print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
          'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
          'loss=', (c * pi).sum(),
          'loss with entropy=', (c * pi + eps * pi * np.log(pi)).sum())


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
    Sinkhorn(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters, is_tunning=args.is_tunning)
    print('time usage=', time.time() - start)