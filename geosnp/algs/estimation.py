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

    if est_coef:
        Y = numpy.ones((l, chunk_size))

    # negative log-likelihood function (NLL)
    # use this for Y
    def _nlly(yj, j):
        ll = 0.0
        q, a, b = yj[0], yj[1:k + 1], yj[-1]
        for i in range(n):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue
            xi = X[i]
            qnf = (q * sum(xi**2.0)) + a.dot(xi) + b
            r = numpy.logaddexp2(-qnf, 0)
            r2 = numpy.logaddexp2(qnf, 0)
            ll -= (gij * r) + ((2.0 - gij) * (r2))

        # return NLL in order to minimize
        return -ll

    # negative log-likelihood function (NLL)
    # use this for x
    def _nllx(xi, i):
        ll = 0.0
        zi[0] = sum(xi**2.0)
        zi[1:k + 1] = xi
        for j in range(l):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue
            qnf = zi.T.dot(Y[j])
            r = numpy.logaddexp2(-qnf, 0)
            r2 = numpy.logaddexp2(qnf, 0)
            ll -= (gij * r) + ((2.0 - gij) * (r2))

        # return NLL in order to minimize
        return -ll

    # gradient of the NLL
    # use this for Y
    def _grady(yj, j):
        grad = numpy.zeros(chunk_size)
        q, a, b = yj[0], yj[1:k + 1], yj[-1]
        for i in range(n):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue

            xi = X[i]
            xi2 = sum(xi**2.0)
            qnf = (q * xi2) + a.dot(xi) + b
            fij = qnf / (1.0 + math.exp(qnf))
            zi[0] = xi2
            zi[1:k + 1] = xi
            grad += ((gij * (1.0 - fij)) - ((2.0 - gij) * fij)) * zi

        # flip for NLL
        return -grad

    # gradient of the NLL
    # use this for X
    def _gradx(xi, i):
        grad = numpy.zeros(k)
        for j in range(l):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue

            yj = Y[j]
            qj, aj, bj = yj[0], yj[1:k + 1], yj[-1]
            qnf = (qj * sum(xi**2.0)) + aj.dot(xi) + bj
            fij = qnf / (1.0 + math.exp(qnf))
            grad += ((gij * (1.0 - fij)) + ((2.0 - gij) * fij)) * (2.0 * sum(qj * xi) + aj)

        # flip for NLL
        return -grad

    # hessian of the NLL
    # use this for Y
    def _hessy(yj, j):
        hess = numpy.zeros((chunk_size, chunk_size))
        for i in range(n):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue

            xi = X[i]
            q, a, b = yj[0], yj[1:k + 1], yj[-1]
            xi2 = sum(xi**2.0)
            qnf = (q * xi2) + a.dot(xi) + b
            fij = qnf / (1.0 + math.exp(qnf))
            zi[0] = xi2
            zi[1:k + 1] = xi
            hess -= fij * (1.0 - fij) * numpy.outer(zi, zi)

        # flip for NLL
        hess = -2.0 * hess
        if linalg.det(hess) < 1e-20:
            hess += numpy.eye(chunk_size)

        return hess

    # hessian of the NLL
    # use this for X
    def _hessx(xi, i):
        hess = numpy.zeros((k, k))
        for j in range(l):
            gij = snp_matrix[i, j]
            if gij == geosnp.MISSING:
                continue

            yj = Y[j]
            qj, aj, bj = yj[0], yj[1:k + 1], yj[-1]
            qnf = (qj * sum(xi**2.0)) + aj.dot(xi) + bj
            fij = qnf / (1.0 + math.exp(qnf))
            term = 2.0 * sum(qj * xi) + aj
            hess -= fij * (1.0 - fij) * numpy.outer(term, term) + (gij - 2.0 * fij) * (2.0 * qj)

        # flip for NLL
        hess = -2.0 * hess

        # as NLL wrt X is not necessarily convex, we may need to alter the
        # hessian so that newton method still converges to local optimum
        # algorithm 6.3 from Numerical Opt Nocedal,Wright 1999
        beta = linalg.norm(hess, 'fro')
        tau = 0 if min(numpy.diag(hess)) > 0 else beta
        eye = numpy.eye(k)
        while True:
            hess = hess + (tau * eye)
            # test for Positive Definiteness
            if min(linalg.eigvals(hess)) > 0:
                break
            else:
                tau = max(2 * tau, beta / 2)

        return hess

    logging.info('Beginning optimization.')
    nll = lnll = sys.maxint
    for iter_num in range(1, max_iter + 1):
        # maximize likelihood wrt Q, A, B for fixed X
        nll = 0.0
        if est_coef:
            for j in range(l):
                out = opt.minimize(_nlly, Y[j], method="newton-cg", jac=_grady, hess=_hessy,
                                   args=(j,))
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
                out = opt.minimize(_nllx, X[i], method="newton-cg", jac=_gradx, hess=_hessx,
                                   args=(i,))
                X[i] = out.x
                nll += out.fun

            logging.info("Iteration {0} NLL wrt X: {1}".format(iter_num, nll))

        if math.fabs(nll - lnll) < epsilon:
            break
        lnll = nll

    return X, Y
