import argparse
from dataset import *

parser = argparse.ArgumentParser()

parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='Caffa')
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps-iters', type=int, default=1)
parser.add_argument('--is-tunning', action="store_true", default=False)

args = parser.parse_args()


def min_e(z, eps):
    return - eps * np.log(np.exp(-z / eps).sum())


def min_row(A, eps):
    _, n = A.shape
    ret = [min_e(A[i, :], eps) for i in range(n)]
    return np.asarray(ret)


def min_col(A, eps):
    m, _ = A.shape
    ret = [min_e(A[:, j], eps) for j in range(m)]
    return np.asarray(ret)


def Sinkhorn_Logdomain(c, a, b, iters, eps, eps_iters, is_tunning=False):
    m, n = c.shape

    f = np.ones((m, 1))
    g = np.ones((n, 1))
    for _ in range(eps_iters):
        K = np.exp(- c / eps)
        for i in range(iters):
            S = c - f - g.T
            f = min_row(S, eps) - f.squeeze() + eps * np.log(a)
            f = f.reshape((m, 1))
            S = c - f - g.T
            g = min_col(S, eps) - g.squeeze() + eps * np.log(b)
            g = g.reshape((n, 1))

            pi = np.diag(np.exp(-f.squeeze() / eps)).dot(K).dot(np.diag(np.exp(-g.squeeze() / eps)))

            if is_tunning and i % 1 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
                      'loss=', (c * pi).sum(),
                      'loss with entropy=',  (c * pi + eps * pi * np.log(pi)).sum())

        eps /= 10

    print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
          'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
          'loss=', (c * pi).sum(),
          'loss with entropy=', (c * pi + eps * pi * np.log(pi)).sum())


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

    Sinkhorn_Logdomain(c, mu, nu, iters=args.iters, eps=args.eps,
                       eps_iters=args.eps_iters, is_tunning=args.is_tunning)
