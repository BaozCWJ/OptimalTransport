import argparse
from dataset import *
from scipy.sparse.linalg import cg

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--p', type=float, default=2.)
parser.add_argument('--cost', type=str, choices=['euclid', 'euclidv2', 'random'], default='euclid')
parser.add_argument('--iters', type=int, default=100)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps-iters', type=int, default=1)

args = parser.parse_args()


def my_cg(A, b, x=None):
    # A is a function
    n = len(b)
    if not x:
        x = np.ones(n)
    r = A(x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = A(p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            print('Itr:', i)
            break
        p = beta * p - r
    return x


def Sinkhorn_Newton_Dual(c, a, b, iters, eps, eps_iters):
    # a,b 是边缘分布
    m, _ = c.shape  # m = n*n

    f = np.zeros(m)
    g = np.zeros(m)
    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        for i in range(iters):
            a_ = np.exp(-f/eps) * (K.dot(np.exp(-g/eps)))
            b_ = np.exp(-g/eps) * (K.T.dot(np.exp(-f/eps)))

            # 共轭梯度求逆
            y = eps * np.concatenate((a_-a, b_-b))
            A = lambda x: np.concatenate((a_*x[:m] + np.exp(-f/eps)*K.dot(np.exp(-g/eps)*x[m:]),
                             np.exp(-g/eps)*K.T.dot(np.exp(-f/eps)*x[:m]) + b_*x[m:]))
            x = my_cg(A, y)

            f += x[:m]
            g += x[m:]

            pi = np.diag(np.exp(-f / eps)).dot(K).dot(np.diag(np.exp(-g / eps)))
            if (i+1) % 10 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a/m, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b/m, 1),
                      'real_loss=', (c * pi).sum(),
                      'loss=',  (c * pi + eps * pi * np.log(pi)).sum())

        eps /= 10


if __name__ == '__main__':
    mu, nu = Read_Image_Pair_from_DOTmark(args.n, args.image_class)
    if args.cost == 'euclid':
        c = EuclidCost(args.n ** 2, args.p)
    elif args.cost == 'euclidv2':
        c = EuclidCostv2(args.n ** 2, args.p)
    elif args.cost == 'random':
        c = Random_Cost(args.n ** 2)

    Sinkhorn_Newton_Dual(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)