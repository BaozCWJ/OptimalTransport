import argparse
from dataset import *

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--p', type=float, default=2.)
parser.add_argument('--cost', type=str, choices=['euclid', 'euclidv2', 'random'], default='euclid')
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps-iters', type=int, default=1)

args = parser.parse_args()


def Sinkhorn(c, a, b, iters, eps, eps_iters):
    # a,b 是边缘分布
    m, _ = c.shape  # m= n*n
    v = np.ones(m) / m
    u = u_hat = np.ones(m) / m
    a *= m
    b *= m
    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        for i in range(iters):
            u = a / np.matmul(K, v)
            v = b / np.matmul(K.T, u)

            # v_hat = b / np.matmul(K.T, u_hat)
            # u_hat = a / np.matmul(K, v_hat)
            # u_hat=np.copy(u)
            # v_hat = np.copy(v)
            # v = b / np.matmul(K.T, u_hat)
            # #u_hat = (u + a / np.matmul(K, v_hat)) / 2
            # u = a / np.matmul(K, v_hat)

            pi = np.diag(u).dot(K).dot(np.diag(v)) / m
            # pi_hat = np.diag(u_hat).dot(K).dot(np.diag(v_hat)) / m
            # pi = (pi + pi_hat) / 2

            #pi = np.diag(u + u_hat).dot(K).dot(np.diag(v + v_hat)) / (4 * m)

            # print(u[:10])
            # print(v[:10])

            if (i+1) % 10 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a/m, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b/m, 1),
                      'real_loss=', (c * pi).sum(),
                      'loss=',  (c * pi + eps * pi * np.log(pi)).sum())

        eps = eps / 10  #


if __name__ == '__main__':
    mu, nu = Read_Image_Pair_from_DOTmark(args.n, args.image_class)
    if args.cost == 'euclid':
        c = EuclidCost(args.n ** 2, args.p)
    elif args.cost == 'euclidv2':
        c = EuclidCostv2(args.n ** 2, args.p)
    elif args.cost == 'random':
        c = Random_Cost(args.n ** 2)

    Sinkhorn(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)
