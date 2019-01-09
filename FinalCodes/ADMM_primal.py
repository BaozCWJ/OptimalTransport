import numpy as np

def init(m, n):
    pi = np.zeros((m, n))
    pi_hat = np.zeros((m, n))
    e = np.zeros((m, n))
    lamda = np.zeros(m)
    eta = np.zeros(n)
    return pi, pi_hat, e, lamda, eta

def update(m,n,mu, nu, c, pi, pi_hat, e, lamda, eta, rho, alpha):
    r = (
          (e+ lamda.reshape((m, 1))+ eta.reshape((1, n))- c) / rho
        + mu.reshape((m, 1))
        + nu.reshape((1, n))
        + pi_hat
    )
    pi = (
            r
            - ((r.sum(axis=1) - r.sum() / (m + n + 1)) / (n + 1)).reshape((m, 1))
            - ((r.sum(axis=0) - r.sum() / (m + n + 1)) / (m + 1)).reshape((1, n))
    )

    pi_hat = np.maximum(pi - e / rho, 0.)

    lamda = lamda + alpha * rho * (mu - pi.sum(axis=1))

    eta = eta + alpha * rho * (nu - pi.sum(axis=0))

    e = e + alpha * rho * (pi_hat - pi)

    return pi, pi_hat, e, lamda, eta


def ADMM_primal(c,mu,nu,its, rho, alpha):
    m, n = c.shape
    pi, pi_hat, e, lamda, eta =init(m,n)
    bigrho=rho*1e5
    while bigrho>=rho:
        for j in range(its):
            pi, pi_hat, e, lamda, eta = update(m, n, mu, nu,c, pi, pi_hat, e, lamda, eta, bigrho, alpha)
            if j%50==0:
                print('err1=',np.linalg.norm(pi_hat.sum(axis=1) - mu, 1),'err2=',np.linalg.norm(pi_hat.sum(axis=0) - nu, 1))
                print('loss',(c * pi_hat).sum())
        bigrho=bigrho/10