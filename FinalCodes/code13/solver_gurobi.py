import argparse
from dataset import *
from gurobipy import *


parser = argparse.ArgumentParser()
# 记得补上default值
parser.add_argument('--image-class', type=str, default='ClassicImages')
parser.add_argument('--n', type=int, default=32)
parser.add_argument('--data', type=str, choices=['DOTmark','random','Caffa','ellip'], default='Caffa')
args = parser.parse_args()



def gurobi_set_model(mu, nu, c, M):
    m, n = c.shape

    pi = M.addVars(m, n, lb=0., ub=GRB.INFINITY)

    # LinExpr is much faster than tuplelist.prod or quicksum
    M.addConstrs(LinExpr([(1., pi[i, j]) for j in range(n)]) == mu[i] for i in range(m))
    M.addConstrs(LinExpr([(1., pi[i, j]) for i in range(m)]) == nu[j] for j in range(n))

    M.setObjective(LinExpr([(c[i, j], pi[i, j]) for i in range(m) for j in range(n)]))

    return pi


def solve_gurobi(mu, nu, c,mtd=-1):
    m, n = c.shape
    M = Model("OT")

    M.setParam(GRB.Param.Method, mtd)

    s = gurobi_set_model(mu, nu, c, M)

    M.optimize()

    sx = M.getAttr("x", s)
    pi = np.array([sx[i, j] for i in range(m) for j in range(n)]).reshape(m, n)
    print('err1=', np.linalg.norm(pi.sum(axis=1) - mu, 1),
          'err2=', np.linalg.norm(pi.sum(axis=0) - nu, 1))
    print('loss=', (c * pi).sum())
    return pi


def solve_gurobi_primal_simplex(mu, nu, c):
    return solve_gurobi(mu, nu, c,mtd=0)

def solve_gurobi_dual_simplex(mu, nu, c):
    return solve_gurobi(mu, nu, c,mtd=1)

def solve_gurobi_barrier(mu, nu, c):
    return solve_gurobi(mu, nu, c,mtd=2)


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
    solve_gurobi_primal_simplex(mu, nu,c)
