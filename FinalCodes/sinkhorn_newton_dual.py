import argparse
from dataset import *
import time

parser = argparse.ArgumentParser()

parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=float, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='DOTmark')
parser.add_argument('--iters', type=int, default=200)
parser.add_argument('--eps', type=float, default=10)
parser.add_argument('--eps-iters', type=int, default=1)
parser.add_argument('--is-tunning', action="store_true", default=False)

args = parser.parse_args()


def my_cg(A, b, x=None):
    # A is a function
    n = len(b)
    if not x:
        x = np.ones(n)
    r = A(x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2 * n):
        Ap = A(p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            if args.is_tunning:
                print('Itr:', i)
                pass
            break
        p = beta * p - r
    return x


def Sinkhorn_Newton_Dual(c, a, b, iters, eps, eps_iters, is_tunning):
    m, _ = c.shape

    f = np.zeros(m)
    g = np.zeros(m)
    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        for i in range(iters):
            a_ = np.exp(-f / eps) * (K.dot(np.exp(-g / eps)))
            b_ = np.exp(-g / eps) * (K.T.dot(np.exp(-f / eps)))

            y = eps * np.concatenate((a_ - a, b_ - b))
            A = lambda x: np.concatenate((a_*x[:m] + np.exp(-f/eps)*K.dot(np.exp(-g/eps)*x[m:]),
                             np.exp(-g/eps)*K.T.dot(np.exp(-f/eps)*x[:m]) + b_*x[m:]))
            x = my_cg(A, y)

            f += x[:m]
            g += x[m:]

            pi = np.diag(np.exp(-f / eps)).dot(K).dot(np.diag(np.exp(-g / eps)))

            if is_tunning and (i + 1) % 10 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
                      'loss=', (c * pi).sum(),
                      'loss with entropy=', (c * pi + eps * pi * np.log(pi)).sum())
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
    Sinkhorn_Newton_Dual(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters,
                         is_tunning=args.is_tunning)
    print('time usage=', time.time() - start)
