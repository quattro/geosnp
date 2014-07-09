__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
__all__ = ["est_loc"]

import logging
import math
import sys

import numpy
import numpy.linalg as linalg
import geosnp

from scipy import stats
from scipy import optimize as opt

SKIP_MISSING = -1

def est_loc(population, X=None, Y=None, k=2, max_iter=10, epsilon=1e-3):

    snp_matrix = population.genotype_matrix

    est_coef = Y is None
    est_loc = X is None
    max_iter = 1 if X is not None or Y is not None else max_iter

    # n people, l snps
    n, l = snp_matrix.shape
    chunk_size = k + 2
    zi = numpy.ones(chunk_size)

    # for constraints
    if est_loc:
        X = stats.norm.rvs(size=[n, k])
        for i in range(n):
            # normalize so ||X|| = 1
            X[i] *= (1.0 / linalg.norm(X[i]))

    if est_coef:
        Y = numpy.zeros((l, chunk_size))

    # define a bunch of functions for optimization
    def _gij(i, j):
        if snp_matrix[i, j] == geosnp.HOMO_MINOR:
            return 2
        elif snp_matrix[i, j] == geosnp.HOMO_MAJOR:
            return 0
        elif snp_matrix[i, j] == geosnp.HETERO:
            return 1
        else:
            return SKIP_MISSING

    # negative log-likelihood function (NLL)
    # use this for Y
    def _nlly(yj, j, grad, hess):
        ll = 0.0
        q, a, b = yj[0], yj[1:k + 1], yj[-1]
        for i in range(n):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue
            xi = X[i]
            qnf = (q * sum(xi**2.0)) + a.dot(xi) + b
            ll -= gij * math.log(1 + math.exp(-qnf)) + (2 - gij) * math.log(1 + math.exp(qnf))

        # return NLL in order to minimize
        return -ll

    # negative log-likelihood function (NLL)
    # use this for x
    def _nllx(xi, i, grad, hess):
        ll = 0.0
        zi[0] = sum(xi**2.0)
        zi[1:k + 1] = xi
        for j in range(l):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue
            qnf = zi.T.dot(Y[j])
            ll -= gij * math.log(1 + math.exp(-qnf)) + (2 - gij) * math.log(1 + math.exp(qnf))

            # return NLL in order to minimize
        return -ll

    # gradient of the NLL
    # use this for Y
    y_grad = numpy.zeros(chunk_size)
    def _grady(yj, j, grad, hess):
        grad.fill(0.0)
        q, a, b = yj[0], yj[1:k + 1], yj[-1]
        for i in range(n):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue

            xi = X[i]
            xi2 = sum(xi**2.0)
            qnf = (q * xi2) + a.dot(xi) + b
            fij = qnf / (1.0 + math.exp(qnf))
            zi[0] = xi2
            zi[1:k + 1] = xi
            grad += ((gij * (1.0 - fij)) + ((gij - 2.0) * fij)) * zi

        # flip for NLL
        return -grad

    # gradient of the NLL
    # use this for X
    x_grad = numpy.zeros(k)
    def _gradx(xi, i, grad, hess):
        grad.fill(0.0)
        for j in range(l):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue

            yj = Y[j]
            qj, aj, bj = yj[0], yj[1:k + 1], yj[-1]
            qnf = (qj * sum(xi**2.0)) + aj.dot(xi) + bj
            fij = qnf / (1.0 + math.exp(qnf))
            grad += ((gij * (1.0 - fij)) + ((gij - 2.0) * fij)) * (2.0 * sum(qj * xi) + aj)

        # flip for NLL
        return -grad

    # hessian of the NLL
    # use this for Y
    y_hess = numpy.zeros((chunk_size, chunk_size))
    def _hessy(yj, j, grad, hess):
        hess.fill(0.0)
        for i in range(n):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue

            xi = X[i]
            q, a, b = yj[0], yj[1:k + 1], yj[-1]
            xi2 = sum(xi**2.0)
            qnf = (q * xi2) + a.dot(xi) + b
            fij = qnf / (1.0 + math.exp(qnf))
            zi[0] = xi2
            zi[1:k + 1] = xi
            hess -= 2.0 * fij * (1.0 - fij) * numpy.outer(zi, zi)

        # flip for NLL
        return -hess

    # hessian of the NLL
    # use this for X
    x_hess = numpy.zeros((k, k))
    def _hessx(xi, i, grad, hess):
        hess.fill(0.0)
        for j in range(l):
            gij = _gij(i, j)
            if gij == SKIP_MISSING:
                continue

            yj = Y[j]
            qj, aj, bj = yj[0], yj[1:k + 1], yj[-1]
            qnf = (qj * sum(xi**2.0)) + aj.dot(xi) + bj
            fij = qnf / (1.0 + math.exp(qnf))
            term = 2 * sum(qj * xi) + aj
            hess -= 2.0 * fij * (1.0 - fij) * numpy.outer(term, term) + (gij - fij)*(2.0 * qj)

        # flip for NLL
        return -hess

    logging.info('Beginning optimization.')
    nll = lnll = sys.maxint
    for iter_num in range(1, max_iter + 1):
        # maximize likelihood wrt Q, A, B for fixed X
        # we can do each 'j' individually due to linearity in 'i'
        nll = 0.0
        if est_coef:
            for j in range(l):
                out = opt.minimize(_nlly, Y[j], method="trust-ncg", jac=_grady, hess=_hessy,
                                   args=(j, y_grad, y_hess), options={'gtol': 1e-3})
                Y[j] = out.x
                nll += out.fun

            logging.info("Iteration {0} NLL wrt Y: {1}".format(iter_num, nll))

        # maximize likelihood wrt X for fixed Q, A, B
        # this is not necessarily concave, so we may need to resort to
        # other methods like grid-search if we can find good regions
        # first test with CG.
        nll = 0.0
        if est_loc:
            for i in range(n):
                out = opt.minimize(_nllx, X[i], method="trust-ncg", jac=_gradx, hess=_hessx,
                                   args=(i, x_grad, x_hess), options={'gtol': 1e-3})
                X[i] = out.x
                nll += out.fun

            logging.info("Iteration {0} NLL wrt X: {1}".format(iter_num, nll))

        if math.fabs(nll - lnll) < epsilon:
            break
        lnll = nll


    return X, Y
