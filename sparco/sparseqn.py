"""
Calls cython quasinewton algorithm with batches
"""
# TODO needs documentation

import numpy as np
import quasinewton as qn

class Objective(object):

    def __init__(self, phi, X, mask=None):
        self.phi = phi
        self.X = X
        self.mask = mask
        self.T = X.shape[-1]
        self.C, self.N, self.P = phi.shape
        self.alen = self.T+self.P-1
        self.xhat = np.empty((self.C, self.T))
        self.dx = np.empty_like(self.xhat)
        self.indx = 0

    def objective(self, x, df):
        """Return objective and modify derivative"""
        self.xhat.fill(0.)
        a = x.reshape(self.N, self.alen)        
        deriv = df.reshape((self.N, self.alen))
        deriv.fill(0.)        

        for b in range(self.P):
            self.xhat += np.dot(self.phi[:,:,b], a[:,b:b+self.T])

        # self.dx[:] = self.xhat - self.X[self.indx]
        self.dx[:] = self.xhat - self.X
        fx = 0.5 * (self.dx**2).sum()

        for b in range(self.P):
            deriv[:,b:b+self.T] += np.dot(self.phi[:,:,b].T, self.dx)

        if self.mask is not None:
            deriv *= self.mask

        return fx

# TODO why is the warm start called `Sin`
# TODO document `past`
# TODO remove debug
# TODO redo docstring for mask
def sparseqn_batch(phi, X, lam=1., maxit=25,
                   positive=False, Sin=None, debug=False,
                   delta=0.01, past=6, mask=None):
    """Use quasinewton method to infer coefficients.

    Parameters
    ----------
    phi : 3d array
      Basis
    X : 2d array
      Patch
    lam : float
      L1 penalty
    maxit : int
      Maximum number of quasi-newton iterations
    positive : bool
      If True, only allow positive coefficients
    Sin : 2d array
      Starting value for coefficients; if None, zeros are used.
    debug : bool
      Print debugging information
    delta : int
      ?????
    past : int
      ?????
    mask : 2d array
      An array of dims of coefficients used to mask derivative.

    Returns
    -------
    2d array
      Coefficients for this dataset and this basis.
    """
    defaults = {
        'lam': 1.,
        'maxit': 25,
        'positive': False,
        'Sin': None,
        'debug': False,
        'delta': 0.01,
        'past': 6,
        'mask': None
        }
    C, N, P = phi.shape
    npats = X.shape[0]
    T = X.shape[-1]
    alen = T + P - 1
    A = Sin if Sin else np.zeros((N,alen))
    lam = lam * np.ones(N*alen)

    # don't regularized coefficients that are masked out
    if mask is not None: lam *= mask.flatten()

    # instantiate objective class
    obj = Objective(phi, X, mask)

    q = qn.owlbfgs(obj, N*alen, lam=lam,
                   debug=debug, maxit=maxit, delta=delta,
                   past=past, pos=positive)

    A = q.run(A.flatten()).reshape(N, alen)
    return A
