import argparse
from dataset import *
import mosek

parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark', 'random', 'Caffa', 'ellip'], default='random')
args = parser.parse_args()


def mosek_set_model(mu, nu, c, task):
    m, n = c.shape

    inf = 0.
    task.appendvars(m * n)
    task.appendcons(m + n)

    task.putvarboundlist(range(m * n),
                         [mosek.boundkey.lo] * (m * n),
                         [0.] * (m * n), [inf] * (m * n)
                         )

    for i in range(m):
        task.putarow(i, range(i * n, (i + 1) * n), [1.] * n)
    task.putconboundlist(range(0, m), [mosek.boundkey.fx] * m, mu, mu)

    for i in range(n):
        task.putarow(i + m, range(i, i + m * n, n), [1.] * m)
    task.putconboundlist(range(m, m + n), [mosek.boundkey.fx] * n, nu, nu)

    task.putclist(range(m * n), c.reshape(m * n))

    task.putobjsense(mosek.objsense.minimize)


def solve_mosek(mu, nu, c, mtd=None, sol=None, log=None):
    m, n = c.shape

    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, log)

        with env.Task() as task:
            task.set_Stream(mosek.streamtype.log, log)

            task.putintparam(mosek.iparam.optimizer, mtd)

            mosek_set_model(mu, nu, c, task)

            task.optimize()

            xx = [0.] * (m * n)
            task.getxx(sol, xx)

            pi = np.array(xx).reshape(m, n)
            print('err1=', np.linalg.norm(pi.sum(axis=1) - mu, 1),
                  'err2=', np.linalg.norm(pi.sum(axis=0) - nu, 1))
            print('loss=', (c * pi).sum())
    return pi


def solve_mosek_primal_simplex(mu, nu, c):
    return solve_mosek(mu, nu, c,
                       mtd=mosek.optimizertype.primal_simplex, sol=mosek.soltype.bas)


def solve_mosek_dual_simplex(mu, nu, c):
    return solve_mosek(mu, nu, c,
                       mtd=mosek.optimizertype.dual_simplex, sol=mosek.soltype.bas)


def solve_mosek_interior_point(mu, nu, c):
    return solve_mosek(mu, nu, c,
                       mtd=mosek.optimizertype.intpnt, sol=mosek.soltype.itr)


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

    solve_mosek_primal_simplex(mu, nu, c)
