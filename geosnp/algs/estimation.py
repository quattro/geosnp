__author__ = "Nicholas Mancuso (nick.mancuso@gmail.com)"
__all__ = ["est_loc"]

import math
import numpy
import numpy.linalg as linalg

from scipy import stats
from scipy import optimize as opt
import geosnp


def est_loc(population, k=2, max_iter=10):

    snp_matrix = population.genotype_matrix

    # n people, l snps
    n, l = snp_matrix.shape

    # for constraints
    flat_eye = numpy.eye(k).flat
    lagmult = 1.0

    X, Y, Z, valid, chunk_size = _get_variables(snp_matrix, k)

    # define a bunch of functions for optimization
    def _fij(i, yj):
        qnf = math.exp(Z[i].T.dot(yj))
        return qnf / (1.0 + qnf)

    def _gij(i, j):
        if snp_matrix[i, j] == geosnp.HOMO_MINOR:
            return 2
        elif snp_matrix[i, j] == geosnp.HOMO_MAJOR:
            return 0
        elif snp_matrix[i, j] == geosnp.HETERO:
            return 1
        else:
            raise ValueError("Missing SNP information!")

    # negative log-likelihood function (NLL)
    # use this for Y
    def _nlly(yj, j):
        ll = 0.0
        # add the constraint of Q = qI
        constraint = lagmult * (yj[:k**2].dot(flat_eye) - k*yj[0])
        for i in range(n):
            gij = _gij(i, j)
            qnf = Z[i].T.dot(yj)
            ll -= gij * math.log(1 + math.exp(-qnf)) + (2 - gij) * math.log(1 + math.exp(qnf))

        # return NLL in order to minimize
        return -ll - constraint

    # negative log-likelihood function (NLL)
    # use this for x
    def _nllx(xi, i):
        ll = 0.0
        zi = numpy.concatenate((numpy.outer(xi, xi).flat, xi, [1.0]))
        for j in range(l):
            if not valid[j]:
                continue
            gij = _gij(i, j)
            qnf = zi.T.dot(Y[j])
            ll -= gij * math.log(1 + math.exp(-qnf)) + (2 - gij) * math.log(1 + math.exp(qnf))

            # return NLL in order to minimize
        return -ll

    # gradient of the NLL
    # use this for Y
    def _grady(yj, j):
        grad = numpy.zeros(chunk_size)
        for i in range(n):
            fij = _fij(i, yj)
            gij = _gij(i, j)
            grad += ((gij * (1.0 - fij)) + ((gij - 2.0) * fij)) * Z[i]

        # flip for NLL
        return -grad

    # gradient of the NLL
    # use this for X
    def _gradx(xi, i):
        grad = numpy.zeros(k)
        for j in range(l):
            if not valid[j]:
                continue
            fij = _fij(i, Y[j])
            gij = _gij(i, j)
            qj, aj = Y[j][:k**2].reshape((k, k)), Y[j][k**2:k**2 + 1]
            grad += ((gij * (1.0 - fij)) + ((gij - 2.0) * fij)) * (2.0 * qj.dot(xi) + aj)

        # flip for NLL
        return -grad

    # hessian of the NLL
    # use this for Y
    def _hessy(yj, j):
        hess = numpy.zeros((chunk_size, chunk_size))
        for i in range(n):
            fij = _fij(i, yj)
            hess -= 2.0 * fij * (1.0 - fij) * numpy.outer(Z[i], Z[i])

        # flip for NLL
        return -hess

    # hessian of the NLL
    # use this for X
    def _hessx(xi, i):
        hess = numpy.zeros((k, k))
        for j in range(l):
            if not valid[j]:
                continue
            fij = _fij(i, Y[j])
            gij = _gij(i, j)
            qj, aj = Y[j][:k**2].reshape((k, k)), Y[j][k**2:k**2 + 1]
            term = 2 * qj.dot(xi) + aj
            hess -= 2.0 * fij * (1.0 - fij) * numpy.outer(term, term) + (gij - fij)*(2.0 * qj)

        # flip for NLL
        return -hess

    print 'beginning optimization'
    for iter_num in range(max_iter):
        # maximize likelihood wrt Q, A, B for fixed X
        # we can do each 'j' individually due to linearity in 'i'
        nll = 0.0
        for j in range(l):
            if not valid[j]:
                continue
            out = opt.minimize(_nlly, Y[j], method="trust-ncg", jac=_grady, hess=_hessy, args=(j,),
                               options={'gtol': 1e-3})
            Y[j] = out.x
            nll += out.fun

        print "Iteration {0} NLL wrt Y: {1}".format(iter_num, nll)

        # maximize likelihood wrt X for fixed Q, A, B
        # this is not necessarily concave, so we may need to resort to
        # other methods like grid-search if we can find good regions
        # first test with CG.
        nll = 0.0
        for i in range(n):
            xi = Z[i][k**2:k**2 + k]
            out = opt.minimize(_nllx, xi, method="trust-ncg", jac=_gradx, hess=_hessx, args=(i,),
                               options={'gtol': 1e-3})
            X[i] = out.x
            Z[i] = numpy.concatenate((numpy.outer(out.x, out.x).flat, out.x, [1.0]))
            nll += out.fun

        print "Iteration {0} NLL wrt X: {1}".format(iter_num, nll)

    return X, Y


def _get_variables(snp_matrix, X, k=2):
    """ This function takes the snp matrix along with an initial estimate X
    and computes the extended variable formulations Z and Y.
    """
    # n people, l snps
    n, l = snp_matrix.shape

    # random initialization of the location matrix
    X = stats.norm.rvs(size=[n, k])

    # normalize so ||X[i]|| = 1, for each i
    for i in range(n):
        X[i] *= (1.0 / linalg.norm(X[i]))

    chunk_size = k**2 + k + 1

    # Get the MLE parameters for each Gaussian
    valid = numpy.ones(l)
    mu = numpy.ones((l, 2, k))
    sigma = numpy.ones((l, 2, k, k))
    p = numpy.ones((l, 2))
    for j in range(l):
        xs = [X[i] for i in range(n) if snp_matrix[i, j] == geosnp.HOMO_MAJOR]
        flen = float(len(xs))
        p[j, 0] = flen / n
        if not xs:
            valid[j] = 0
            continue
        mu[j, 0] = sum(xs) / flen
        sigma[j, 0] = sum([numpy.outer((x - mu[j, 0]), (x - mu[j, 0])) for x in xs]) / flen

        xs = [X[i] for i in range(n) if snp_matrix[i, j] == geosnp.HOMO_MINOR]
        flen = float(len(xs))
        p[j, 1] = 1.0 - p[j, 0]
        if not xs:
            valid[j] = 0
            continue
        mu[j, 1] = sum(xs) / flen
        sigma[j, 1] = sum([numpy.outer((x - mu[j, 1]), (x - mu[j, 1])) for x in xs]) / flen

    # the extended Z = [vec(XX^T), X, 1] formulation
    Z = numpy.ones((n, chunk_size))
    for i in range(n):
        Z[i] = numpy.concatenate((numpy.outer(X[i], X[i]).flat, X[i], [1.0]))

    # the extended Y = [vec(Q), A, B] formulation
    Y = numpy.ones((l, chunk_size))
    for j in range(l):
        if not valid[j]:
            continue
        sigmaj0 = sigma[j, 0]
        sigmaj1 = sigma[j, 1]
        inv_sigma_j0, det_sigma_j0 = _pinv_pdet(sigmaj0)
        inv_sigma_j1, det_sigma_j1 = _pinv_pdet(sigmaj0)
        inv_sigma_j1 = linalg.pinv(sigmaj1)
        muj0 = mu[j, 0]
        muj1 = mu[j, 1]

        qj = -0.5 * (inv_sigma_j0 - inv_sigma_j1)
        aj = inv_sigma_j1.dot(muj1) - inv_sigma_j0.dot(muj0)
        bj = muj0.T.dot(inv_sigma_j0).dot(muj0) - muj1.T.dot(inv_sigma_j1).dot(muj1)
        bj -= math.log(det_sigma_j1 / det_sigma_j0)
        bj -= 2.0 * math.log(p[j, 0] / p[j, 1])
        bj *= -0.5
        Y[j] = numpy.concatenate((qj.flat, aj, [bj]))

    return X, Y, Z, valid, chunk_size

def _pinv_pdet(a, rcond=1e-15):
    a = a.conjugate()
    u, s, vt = linalg.svd(a, 0)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * numpy.maximum.reduce(s)
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.;
    pinv = linalg.dot(numpy.transpose(vt), numpy.multiply(s[:, numpy.newaxis], numpy.transpose(u)))
    pdet = numpy.product([d for d in numpy.diag(s) if d != 0.0])

    return pinv, pdet
