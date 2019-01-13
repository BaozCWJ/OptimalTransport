import argparse
from dataset import *
from scipy.sparse.linalg import cg

parser = argparse.ArgumentParser()

parser.add_argument('--image-class', type=str, default='Shapes')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='Caffa')
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

    Sinkhorn_Newton(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)