"""
some random tools, slow code.
"""

import collections
import imp
import os
import time
import types

import numpy as np
import scipy.signal as signal

import sparco.mpi as mpi

# TODO docstrings for all methods

###################################
########### OBJECTIVE
###################################
# TODO give more generic names, move

def obj(x, a, phi):
  xhat = compute_xhat(phi, a)
  dx = compute_dx(x, xhat=xhat)
  E = compute_E(dx)
  dphi = compute_dphi(dx, a)
  return xhat, dx, E, dphi

def compute_dx(x, a=None, phi=None, xhat=None):
  return x - (xhat if xhat != None else compute_xhat(phi, a))

def compute_xhat(phi, a):
  p = phi.shape[2]; t = a.shape[1] - p + 1
  return reduce(np.add, (np.dot(phi[:,:,i], a[:,i:i+t]) for i in range(p)))

def compute_E(dx):
  return 0.5 * np.linalg.norm(dx)**2

def compute_dphi(dx, a):
  t = dx.shape[1]; p = a.shape[1] - t + 1
  return np.dstack(tuple(np.dot(dx, a[:,i:i+t].T) for i in range(p)))
  # return np.array(tuple(np.dot(dx, a[:,i:i+t].T) for i in range(p)))

def compute_angle(phi1, phi2):
  dot = np.sum(phi1*phi2) / (np.linalg.norm(phi1) * np.linalg.norm(phi2))
  angle = np.arccos(dot) * 180 / np.pi
  return 0 if np.isnan(angle) else angle

def compute_proposed_phi(phi, dphi, eta):
  newphi = phi - eta * dphi
  return newphi / vnorm(newphi)


###################################
########### MATRIX/VECTOR OPERATIONS
###################################
# TODO Needed or can be done with numpy calls?

def vnorm(phi):
    """
    Norm of each basis element as a [1, N, 1] matrix
    """
    return np.sqrt((phi**2).sum(axis=0).sum(axis=1))[np.newaxis,:,np.newaxis]

def norm(phi):
    """
    Norm of each basis element as a N-vector
    """
    return np.sqrt((phi**2).sum(axis=0).sum(axis=1))

def blur(phi, window=.2):
    """
    Gaussian blur of basis functions
    """
    C, N, P = phi.shape
    w = np.int(window * min(C,P))
    g = signal.gaussian(w, 1)
    philp = np.empty_like(phi)
    for i in range(N):
        philp[:,i] = signal.sepfir2d(phi[:,i], g, g)
    return philp

def center(arr, maxshift=None):
  """
  Shift each basis function to its center of mass by a maximum
  amount of cmax.

  Optionally, smooth basis functions.

  Center of mass is defined using sum of squares.
  """
  for n in range(self.N):
    s = np.sum(arr[:,n]**2, axis=0)
    total = np.sum(s)
    if total == 0.: continue
    m = int(np.round(np.sum(np.arange(self.P) * s)/total))
    shift = self.P/2 - m
    if maxshift:
      shift = np.sign(shift) * min(abs(shift), self.cmax)
    arr[:,n] = np.roll(arr[:,n], shift, axis=1)
    if shift > 0:
      arr[:,n,0:shift] = 0.
    elif shift < 0:
      arr[:,n,shift:] = 0.
    else:
      continue
    arr[:,n] /= np.linalg.norm(arr[:,n])
    print 'Shifting %d by %d' % (n, shift)

def smooth(phi):
  a = 1
  b = [0.25, .5, 0.25]
  for n in range(self.N):
    phi[:,n] = scipy.signal.lfilter(b, a, phi[:,n], axis=1)

###################################
########### OTHER
###################################

# TODO complete this terminating decorator
# def terminate_after(seconds):
#   def inner_decorator(orig):
#       @functools.wraps(orig)
#       def wrapper(*args, **kwargs):
#         return test_func(*args, **kwargs)
#       return wrapper
#     return actualDecorator

###################################
########### DEPRECATED
###################################

# TODO remove once it has been taken from the experimental learners
def attributesFromDict(d):
    "Automatically initialize instance variables, Python Cookbook 6.18"
    self = d.pop('self')
    for n, v in d.iteritems():
        setattr(self, n, v)
