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

    # random initialization of the location matrix
    X = stats.norm.rvs(size=[n, k])

    chunk_size = k**2 + k + 1
    Z, Y = _get_variables(snp_matrix, X, chunk_size, k)

    # do this to get the part of the flattened vector we are interested in
    def _chunk(i, j):
        jprime = j * chunk_size
        end = (j + 1) * chunk_size
        return jprime, end

    def _fij(i, j, y):
        jprime, end = _chunk(i, j)
        return 1.0 / (1.0 + numpy.exp(Z[i].T.dot(y[jprime:end])))

    # likelihood function
    def _f(y):
        ll = 0.0
        for i in range(n):
            for j in range(l):
                jprime, end = _chunk(i, j)
                gij = snp_matrix[i, j]
                yj = y[jprime:end]
                ll += gij * math.log(1 + math.exp(-Z[i].T.dot(yj))) + (2 - gij) * math.log(1 + math.exp(Z[i].T.dot(yj)))
        return -ll

    # gradient of the likelihood
    def _grad(y, *args):
        grad = numpy.zeros(n)
        for i in range(n):
            for j in range(l):
                fij = _fij(i, j, y)
                gij = snp_matrix[i, j]
                grad[j] += (gij * (1 - fij) - (2 - gij) * fij) * Z[i]
        return grad

    # hessian of the likelihood
    def _hess(y, *args):
        hess = numpy.zeros((n, l))
        for i in range(n):
            for j in range(l):
                fij = _fij(i, j, y)
                hess += 2 * fij * (1 - fij) * numpy.outer(Z[i], Z[i])
        return hess

    out = None
    for iter in range(max_iter):
        # maximize likelihood wrt Q, A, B for fixed X
        # ...
        out = opt.fmin_ncg(_f, Y, fprime=_grad, fhess=_hess)

        # maximize likelihood wrt X for fixed Q, A, B
        # ...
        print out

    return out


def _get_variables(snp_matrix, X, chunk_size, k=2):
    """ This function takes the snp matrix along with an initial estimate X
    and computes the extended variable formulations Z and Y.
    """
    # n people, l snps
    n, l = snp_matrix.shape

    # Get the MLEs for each Gaussian
    mu = numpy.ones((l, 2, k))
    sigma = numpy.ones((l, 2, k, k))
    for j in range(l):
        snp = MAJOR
        xs = [X[i] for i in range(n) if snp_matrix[i, j] == snp or snp_matrix[i, j] == HETERO]
        flen = float(len(xs))
        mu[j, 0] = sum(xs) / flen
        sigma[j, 0] = sum([numpy.outer((x - mu[j, 0]), (x - mu[j, 0])) for x in xs]) / flen

        snp = MINOR
        xs = [X[i] for i in range(n) if snp_matrix[i, j] == snp or snp_matrix[i, j] == HETERO]
        flen = float(len(xs))
        mu[j, 1] = sum(xs) / flen
        sigma[j, 1] = sum([numpy.outer((x - mu[j, 1]), (x - mu[j, 1])) for x in xs]) / flen

    # These should be provided [I'd imagine] or directly estimated;
    # otherwise, we would only have local estimate for MLEs
    # but for testing this is fine...
    alpha = numpy.ones(k)
    p = numpy.random.dirichlet(alpha, size=l)

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

    return Z, Y
