import argparse
from dataset import *

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark','random','Caffa','ellip'], default='Caffa')
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('--eps', type=float, default=1)
parser.add_argument('--eps-iters', type=int, default=1)

args = parser.parse_args()


def Sinkhorn(c, a, b, iters, eps, eps_iters):
    # a,b 是边缘分布
    m= len(c)  # m= n*n
    v = np.ones(m)
    for _ in range(eps_iters):
        K = np.exp(- c / eps)
        for i in range(iters):
            #if i%2==0:
            u = a / np.dot(K, v)
            v = b / np.dot(K.T, u)
            #else:
            #    v = b / np.dot(K.T, u)
            #    u = a / np.dot(K, v)
            pi = np.diag(u).dot(K).dot(np.diag(v))
            if (i) % 100 == 0:
                print('err1=', np.linalg.norm(pi.sum(axis=1) - a, 1),
                      'err2=', np.linalg.norm(pi.sum(axis=0) - b, 1),
                      'loss=', (c * pi).sum())

        eps = eps / 10  #


if __name__ == '__main__':
    if args.data == 'DOTmark':
        mu, nu = DOTmark_Weight(args.n, args.image_class)
        c = DOTmark_Cost(0,1,0,1,args.n)
    elif args.data == 'random':
        mu = Random_Weight(args.n ** 2)
        nu = Random_Weight(args.n ** 2)
        c = Random_Cost(args.n ** 2)
    elif args.data == 'Caffa':
        mu = Const_Weight(args.n ** 2)
        nu = Const_Weight(args.n ** 2)
        c = Caffarelli_Cost(args.n ** 2,0,0,1,2)
    elif args.data=='ellip':
        mu = Const_Weight(args.n ** 2)
        nu = Const_Weight(args.n ** 2)
        c = ellipse_Cost(args.n ** 2,0,0,0.5,2,0.1)

    Sinkhorn(c, mu, nu, iters=args.iters, eps=args.eps, eps_iters=args.eps_iters)
