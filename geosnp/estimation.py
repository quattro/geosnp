__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
__all__ = ["est_loc"]

import math
import numpy
import numpy.linalg as linalg

from scipy import stats
from scipy import optimize as opt

MAJOR = 0
HETERO = 1
MINOR = 2


def est_loc(snp_matrix, k=2, max_iter=10):
    # n people, l snps
    n, l = snp_matrix.shape

    X, Y, Z, chunk_size = _get_variables(snp_matrix, k)

    def _fij(i, yj):
        quad = math.exp(Z[i].T.dot(yj))
        return quad / (1.0 + quad)

    # likelihood function
    def _f(yj, j):
        ll = 0.0
        for i in range(n):
            gij = snp_matrix[i, j]
            print Z[i].T.dot(yj), Z[i], yj
            ll += gij * math.log(1 + math.exp(-Z[i].T.dot(yj))) + (2 - gij) * math.log(1 + math.exp(Z[i].T.dot(yj)))

        return -ll

    # gradient of the likelihood
    def _grad(yj, j):
        grad = numpy.zeros(chunk_size)
        for i in range(n):
            fij = _fij(i, yj)
            gij = snp_matrix[i, j]
            grad += (gij * (1 - fij) - (2 - gij) * fij) * Z[i]

        return grad

    # hessian of the likelihood
    def _hess(yj, j):
        hess = numpy.zeros((chunk_size, chunk_size))
        for i in range(n):
            fij = _fij(i, yj)
            hess -= 2 * fij * (1 - fij) * numpy.outer(Z[i], Z[i])

        return hess

    out = None
    for iter in range(max_iter):
        # maximize likelihood wrt Q, A, B for fixed X
        # we can do each 'j' individually due to linearity in 'i'
        for j in range(l):
            out = opt.minimize(_f, Y[j], method="Newton-CG", jac=_grad, hess=_hess, args=(j,))
            Y[j] = out.x

        # maximize likelihood wrt X for fixed Q, A, B
        # ...

    return out


def _get_variables(snp_matrix, k=2):
    """ This function takes the snp matrix along with an initial estimate X
    and computes the extended variable formulations Z and Y.
    """
    # n people, l snps
    n, l = snp_matrix.shape

    # random initialization of the location matrix
    X = stats.norm.rvs(size=[n, k], loc=[10]*k)

    chunk_size = k**2 + k + 1

    # Get the MLEs for each Gaussian
    mu = numpy.ones((l, 2, k))
    sigma = numpy.ones((l, 2, k, k))
    p = numpy.ones((l, 2))
    for j in range(l):
        xs = [X[i] for i in range(n) if snp_matrix[i, j] == MAJOR]
        flen = float(len(xs))
        p[j, 0] = flen / n
        p[j, 1] = 1 - flen / n
        mu[j, 0] = sum(xs) / flen
        sigma[j, 0] = sum([numpy.outer((x - mu[j, 0]), (x - mu[j, 0])) for x in xs]) / flen

        xs = [X[i] for i in range(n) if snp_matrix[i, j] == MINOR]
        flen = float(len(xs))
        mu[j, 1] = sum(xs) / flen
        sigma[j, 1] = sum([numpy.outer((x - mu[j, 1]), (x - mu[j, 1])) for x in xs]) / flen

    # the extended X = [vec(XX^T), X, 1] formulation
    Z = numpy.ones((n, chunk_size))
    for i in range(n):
        Z[i] = numpy.concatenate((numpy.outer(X[i], X[i]).flatten('f'), X[i], [1.0]))

    # the extended Y = [vec(Q), A, B] formulation
    Y = numpy.ones((l, chunk_size))
    for j in range(l):
        inv_sigma_j0 = linalg.inv(sigma[j, 0])
        inv_sigma_j1 = linalg.inv(sigma[j, 1])
        qj = inv_sigma_j0 - inv_sigma_j1
        aj = (inv_sigma_j1.dot(mu[j, 1])) - (inv_sigma_j0.dot(mu[j, 0]))
        bj = (mu[j, 0].T.dot(inv_sigma_j0).dot(mu[j, 0])) - (mu[j, 1].T.dot(inv_sigma_j1).dot(mu[j, 1]))
        bj += numpy.log(math.sqrt(linalg.det(inv_sigma_j1) / linalg.det(inv_sigma_j0)) + (p[j, 0] / p[j, 1]))
        Y[j] = numpy.concatenate((qj.flatten('f'), aj, [bj]))

    return X, Y, Z, chunk_size
