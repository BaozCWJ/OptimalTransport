import argparse
from dataset import *

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--p', type=float, default=2.)
parser.add_argument('--cost', type=str, choices=['euclid', 'random'], default='euclid')
parser.add_argument('--iters', type=int)
parser.add_argument('--eps', type=float,)
parser.add_argument('--eps-iters', type=int)

args = parser.parse_args()


def Sinkhorn(c, a, b, iters, eps, eps_iters):
    # a,b 是边缘分布
    m, _ = c.shape  # m= n*n
    v = np.ones(m)

    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        for i in range(iters):
            u = a / np.matmul(K, v)
            v = b / np.matmul(K.T, u)
            pi = (u * K).T * v  # np.diag(u) 与 K 与 np.diag(v) 矩阵乘， pi是得到的联合分布

            if (i+1) % 50 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
                      'loss=', (c * pi).sum())

        eps = eps / 10  #


if __name__ == '__main__':
    mu, nu = Read_Image_Pair_from_DOTmark(args.n, args.image_class)
    if args.cost == 'euclid':
        c = EuclidCost(args.n ** 2, args.p)
    elif args.cost == 'random':
        c = Random_Cost(args.n ** 2)

    Sinkhorn(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)
