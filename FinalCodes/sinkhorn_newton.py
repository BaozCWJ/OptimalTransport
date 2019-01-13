import argparse
from dataset import *
from scipy.sparse.linalg import cg

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--p', type=float, default=2.)
parser.add_argument('--cost', type=str, choices=['euclid', 'euclidv2', 'random'], default='euclid')
parser.add_argument('--iters', type=int, default=10)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps-iters', type=int, default=1)

args = parser.parse_args()


def Sinkhorn_Newton(c, a, b, iters, eps, eps_iters):
    # a,b 是边缘分布
    m, _ = c.shape  # m = n*n

    for _ in range(eps_iters):
        K = np.exp(- c / eps)

        for i in range(iters):
            a_ = K.dot(np.ones(m))
            b_ = K.T.dot(np.ones(m))

            # 共轭梯度求逆
            y = eps * np.concatenate((a_-a, b_-b))
            #A = lambda x: np.concatenate((a_*x[:m] + K.dot(x[m:]), K.T.dot(x[:m]) + b_*x[m:]))
            A = np.vstack((np.hstack((np.diag(a_), K)), np.hstack((K.T, np.diag(b_)))))
            x, info = cg(A, y)
            # scipy的cg报错信息
            if info > 0:
                print('cg not converge!')
            elif info < 0:
                print('illegal input!')

            K = np.diag(np.exp(-x[:m]/eps)).dot(K).dot(np.diag(np.exp(-x[m:]/eps)))

            if (i + 1) % 10 == 0:
                print('err1=', np.linalg.norm(K.sum(axis=1) - a / m, 1),
                      'err2=', np.linalg.norm(K.sum(axis=0) - b / m, 1),
                      'real_loss=', (c * K).sum(),
                      'loss=', (c * K + eps * K * np.log(K)).sum())

        eps /= 10


if __name__ == '__main__':
    mu, nu = Read_Image_Pair_from_DOTmark(args.n, args.image_class)
    if args.cost == 'euclid':
        c = EuclidCost(args.n ** 2, args.p)
    elif args.cost == 'euclidv2':
        c = EuclidCostv2(args.n ** 2, args.p)
    elif args.cost == 'random':
        c = Random_Cost(args.n ** 2)

    Sinkhorn_Newton(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)