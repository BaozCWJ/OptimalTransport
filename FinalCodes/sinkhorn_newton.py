import argparse
import time
from dataset import *
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=float, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='DOTmark')
parser.add_argument('--iters', type=int, default=30)
parser.add_argument('--eps', type=float, default=4e-4)
parser.add_argument('--eps-iters', type=int, default=1)
parser.add_argument('--is-tunning', action="store_true", default=False)
parser.add_argument('--draw', action="store_true", default=False)

args = parser.parse_args()


def Sinkhorn_Newton(c, a, b, iters, eps, eps_iters, is_tunning):
    m, _ = c.shape

    for _ in range(eps_iters):
        K = np.exp(- c / eps)
        if args.draw:
            plt.figure()
            l = 1

        for j in range(iters):
            a_ = K.dot(np.ones(m))
            b_ = K.T.dot(np.ones(m))

            y = eps * np.concatenate((a_ - a, b_ - b))
            A = np.vstack((np.hstack((np.diag(a_), K)), np.hstack((K.T, np.diag(b_)))))
            x = np.linalg.solve(A, y)

            K = np.diag(np.exp(-x[:m] / eps)).dot(K).dot(np.diag(np.exp(-x[m:] / eps)))

            if is_tunning and (j + 1) % 3 == 0:
                print('err1=', np.linalg.norm(K.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(K.sum(axis=0) - b, 1),
                      'loss=', (c * K).sum(),
                      'loss with entropy=', (c * K + eps * K * np.log(K)).sum())

                if args.draw:
                    pi_plot = np.zeros_like(K)
                    for k in range(K.shape[0]):
                        pi_plot[k, K[k, :].argsort()[-20:]] = 1

                    plt.subplot(2, 5, l)
                    l += 1
                    plt.imshow(pi_plot)
                    plt.title('iteration=' + str(j + 1))

        eps /= 10
        if args.draw:
            plt.show()

    print('err1=', np.linalg.norm(K.sum(axis=1) - a, 1),
          'err2=', np.linalg.norm(K.sum(axis=0) - b, 1),
          'loss=', (c * K).sum())


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
    Sinkhorn_Newton(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters, is_tunning=args.is_tunning)
    print('time usage=', time.time() - start)