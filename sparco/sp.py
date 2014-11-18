"""
A convolution based sparsenet of an array of time series

Key:
 phi - basis [channel, neuron, coefficients]
 A   - coefficients
 X   - patches of data

Note:
1. The kernels are convolution kernels.
"""

# TODO needs docstrings for almost all functions
# TODO needs implementation of basis centering

import functools
import os
import time

import numpy as np
import pfacets

import mpi
import sparco
import sparco.sparseqn
import sparco.sptools as sptools

###################################
########### UTILITY
###################################

def config_key(config, sn=None):
  config = (config.__dict__ if isinstance(config, Spikenet)
      else pfacets.merge(Spikenet.defaults, config))
  tup = (config['num_iterations'],
      config['inference_settings']['lam'],
      config['inference_settings']['maxit'])
  return "niter_{0}_lam_{1}_maxit_{2}".format(*tup)

###################################
########### MPI NODE
###################################

# TODO finish the Spikenet docstring with reference to inference/learning specs

class Spikenet(object):
  """Carry out convolutional sparse-coding on an MPI node.

  This class implements convolutional sparse coding on any MPI node.
  Configuration is implemented as keyword arguments to the `__init__` method.
  Execution can be terminated either after a fixed time (using
  `run_time_limit`) or number of iterations (using `num_iterations`). The code
  implementing inference and learning is provided by a function and class,
  respectively.

  Parameters
  ----------
  sampler : Sampler or other class with `get_patches`
    The data provider. Must respond to `get_patches` with a 3d numpy array.
  patches_per_iteration : int (optional)
    Number of patches to process in a single iteration.
  starting_iteration : int (optional)
    Starting value of the iteration counter. Useful when resuming a run.
  num_iterations : int (optional)
    Total number of iterations to carry out before terminating.
  run_time_limit : number (optional)
    Execution will terminate if this amount of time is exceeded.
  dictionary_size : int (optional)
    The number of basis functions to generate.
  convolution_time_length : int (optional)
    The length of each basis function in the time dimension.
  phi : 3d array (optional)
    A starting set of basis functions.
  inference_function : function
    A function that satisfies the inference function specification.
  inference_settings : dict
    Keyword arguments for inference_function.
  eta : float
    A starting value for the basis adjustment constant.
  learner_class : class
    A class implementing the learner class specification.
  eta_up_factor : float
    Factor by which eta will be adjusted if basis angle is below target.
  eta_down_factor : float
    Factor by which eta will be adjusted if basis angle is above target.
  target_angle : number
    Target angle between successive states of phi 
  max_angle : number
    Maximum tolerable angle between successive states of phi. Angles above
    this will cause the rejection of the most recently calculated phi.
  update_coefficient_statistics_interval : int
    The number of iterations between updates to coefficient statistics
    (l0/l1/l2 norm, variance).
  basis_centering_interval : int or None
    The number of iterations between centerings of the basis.
  basis_centering_max_shift : number
    The maximum shift for each basis function during centering.
  """
  # todos for above docstring
  # TODO dimension specification for `phi`
  # TODO must provide reference to specification for inference function
  # TODO must provide reference to specification for learning function
  # TODO better name for eta than adjustment constant
  # TODO eta_up/down_factor and other params are is specific to angle chasing learner

  ########### INITIALIZATION

  defaults = {
      'sampler_settings': None,
      'patches_per_iteration': mpi.procs,
      'starting_iteration': 0,
      'num_iterations': 100,
      'run_time_limit': float("inf"),
      'num_channels': None,
      'patch_length': 128,
      'dictionary_size': 100,
      'convolution_time_length': 64,
      'phi': None,
      'inference_function': sparco.sparseqn.sparseqn_batch,
      'inference_settings': {
        'lam': 0,
        'maxit': 15,
        'debug': False,
        'positive': False,
        'delta': 0.0001,
        'past': 6
        },
      'eta': .00001,
      'learner_class': sparco.learn.AngleChasingLearner,
      'eta_up_factor': 1.01,
      'eta_down_factor': .99,
      'target_angle': 1.,
      'max_angle': 2.,
      'update_coefficient_statistics_interval': 100,
      'basis_centering_interval': None,
      'basis_centering_max_shift': None,
      }

  def __init__(self, **kwargs):
    """Configure the Spikenet."""

    pfacets.set_attributes_from_dicts(self, Spikenet.defaults, kwargs)

    self.patches_per_core = self.patches_per_iteration / mpi.procs
    pfacets.mixin(self, self.learner_class)
    self.a_variance_cumulative = np.zeros(self.dictionary_size)
    self.run_time = 0
    self.last_time = time.time()
    self.starting_iteration = mpi.bcast_obj(self.starting_iteration)

    C, N, P = (self.num_channels, self.dictionary_size, self.convolution_time_length)
    T = self.patch_length
    patch_nodebuf_dims = { 'a': (N, P+T-1), 'x': (C, T), 'xhat': (C,T),
        'dx': (C,T), 'dphi': (C,N,P), 'E': (1,), 'a_l0_norm': (N,),
        'a_l1_norm': (N,), 'a_l2_norm': (N,), 'a_variance': (N,) }
    nodebuf_dims = { k: (self.patches_per_core,) + v
        for (k,v) in patch_nodebuf_dims.items() }
    rootbuf_dims = pfacets.merge(patch_nodebuf_dims,
        {'x': (self.patches_per_iteration,) + patch_nodebuf_dims['x']})
    self.nodebufs = pfacets.data(**{k: np.zeros(v) for (k,v) in nodebuf_dims.items()})
    self.rootbufs = pfacets.data(**{k: np.zeros(v) for (k,v) in rootbuf_dims.items()})
    self.nodebufs.mean = pfacets.data(**{k: np.zeros(v)
      for (k,v) in patch_nodebuf_dims.items()})
    self.initialize_phi(C,N,P)

  def initialize_phi(self, *dims):
    """Allocate buffer for phi and broadcast from the root."""
    self.phi = np.empty(dims) if self.phi is None else self.phi
    mpi.bcast(self.phi)

  ########### LEARNING
  # methods here draw on methods provided by a learner mixin

  # TODO use a decorator for time termination
  def run(self):
    """ Learn basis by alternative online minimization."""
    for self.t in range(self.starting_iteration, self.num_iterations+1):
      if not self.within_time_limit(): return
      self.iteration() 

  # TODO temp until decorator solution
  def within_time_limit(self):
    """Check if net has been running for longer than configured limit.

    Returns
    -------
    bool
      True if run time is greater than `run_time_limit`
    """
    self.run_time = mpi.bcast_obj(self.run_time)
    return self.run_time < self.run_time_limit

  def iteration(self):
    mpi.bcast(self.phi)
    mpi.scatter(self.rootbufs.x, self.nodebufs.x)
    self.infer_coefficients()
    self.learn_basis()
    if self.t > 0 and self.t % self.update_coefficient_statistics_interval == 0:
      self.update_coefficient_statistics()

  def infer_coefficients(self):
    for i in range(self.patches_per_core):
      self.nodebufs.a[i] = self.inference_function(self.phi,
          self.nodebufs.x[i], **self.inference_settings)

  # more parallel, higher bandwidth requirement
  def learn_basis(self):
    self.compute_patch_objectives(self.nodebufs)
    self.average_patch_objectives(self.nodebufs)
    self.reduce_to_mean('E')
    self.reduce_to_mean('dphi')

  def compute_patch_objectives(self, bufset):
    for i in range(bufset.x.shape[0]):
      res = sptools.obj(bufset.x[i], bufset.a[i], self.phi)
      bufset.xhat[i], bufset.dx[i] = res[0], res[1]
      bufset.E[i], bufset.dphi[i] = res[2], res[3]

  def average_patch_objectives(self, bufset):
    bufset.mean.dphi = np.mean(bufset.dphi, axis=0)
    bufset.mean.E = np.mean(bufset.E, axis=0)

  ########### COEFFICIENT STATISTICS

  # TODO see if I can get the normalized norms in a single call
  def update_coefficient_statistics(self):
    for i in range(self.patches_per_core):

      l0_norm = functools.partial(np.linalg.norm, ord=0)
      self.nodebufs.a_l0_norm[i] = np.apply_along_axis(l0_norm, 1, self.nodebufs.a[i])
      self.nodebufs.a_l0_norm[i] /= self.nodebufs.a[i].shape[1]

      l1_norm = functools.partial(np.linalg.norm, ord=1)
      self.nodebufs.a_l1_norm[i] = np.apply_along_axis(l1_norm, 1, self.nodebufs.a[i])
      self.nodebufs.a_l1_norm[i] /= np.max(self.nodebufs.a_l1_norm[i])

      l2_norm = functools.partial(np.linalg.norm, ord=2)
      self.nodebufs.a_l2_norm[i] = np.apply_along_axis(l2_norm, 1, self.nodebufs.a[i])
      self.nodebufs.a_l2_norm[i] /= np.max(self.nodebufs.a_l2_norm[i])

      self.nodebufs.a_variance[i] = np.apply_along_axis(np.var, 1, self.nodebufs.a[i])
      self.nodebufs.a_variance[i] /= np.max(self.nodebufs.a_variance[i])

    for stat in ['a_l0_norm', 'a_l1_norm', 'a_l1_norm', 'a_variance']:
      setattr(self.nodebufs.mean, stat, np.mean(getattr(self.nodebufs, stat), axis=0))
      self.reduce_to_mean(stat)
      # mpi.gather(getattr(self.nodebufs.mean, stat), getattr(self.rootbufs, stat))

  ########### MEAN

  def reduce_to_mean(self, x):
    mpi.reduce(getattr(self.nodebufs.mean, x), getattr(self.rootbufs, x), op=mpi.SUM)

###################################
########### MPI ROOT
###################################

class RootSpikenet(Spikenet):
  """Extends `Spikenet` with special code to be run only on the MPI root.

  The special functionality includes the allocations of large memory buffers
  for gathering the parallel results, the averaging of parallel results, and
  the updating of eta and phi.
  """

  def __init__(self, **kwargs):
    """Configure the RootSpikenet.

    Configuration differs slightly from what is performed on a node due to the
    override of certain methods called in `Spikenet#__init__`.

    Parameters
    ----------
    see `Spikenet#__init__`
    """
    Spikenet.__init__(self, **kwargs)
    self.sampler = sparco.Sampler(**self.sampler_settings)

  def initialize_phi(self, *dims):
    self.phi = np.random.randn(*dims) if self.phi is None else self.phi
    self.phi /= sptools.vnorm(self.phi)
    super(RootSpikenet, self).initialize_phi(*dims)

  def run(self):
    super(RootSpikenet, self).run()
    self.sampler.close_files()

  # TODO temp until decorator solution
  def within_time_limit(self):
    """Check if net has been running for longer than configured limit.

    Returns
    -------
    bool
      True if run time is greater than `run_time_limit`
    """
    now = time.time()
    self.run_time += now - self.last_time
    self.last_time = now
    return super(RootSpikenet, self).within_time_limit()

  def iteration(self):
    self.load_patches()
    super(RootSpikenet, self).iteration()

  def load_patches(self):
    self.rootbufs.x = self.sampler.get_patches(self.patches_per_iteration)

  def infer_coefficients(self):
    super(RootSpikenet, self).infer_coefficients()

  def learn_basis(self):
    super(RootSpikenet, self).learn_basis()
    self.rootbufs.dphi /= mpi.procs
    self.rootbufs.E /= mpi.procs
    self.update_eta_and_phi()

  def update_eta_and_phi(self):
    self.proposed_phi = sptools.compute_proposed_phi(self.phi,
        self.rootbufs.dphi, self.eta)
    self.phi_angle = sptools.compute_angle(self.phi, self.proposed_phi)
    self.update_phi()
    self.update_eta()

  def update_coefficient_statistics(self):
    super(RootSpikenet, self).update_coefficient_statistics()
    for stat in ['a_l0_norm', 'a_l1_norm', 'a_l2_norm', 'a_variance']:
      setattr(self.rootbufs, stat, getattr(self.rootbufs, stat) / mpi.procs)
    self.a_variance_cumulative += self.rootbufs.a_variance
    self.basis_sort_order = np.argsort(self.a_variance_cumulative)[::-1]

  def reduce_to_mean(self, x):
    super(RootSpikenet, self).reduce_to_mean(x)
    setattr(self.rootbufs, x, getattr(self.rootbufs, x) / mpi.procs)
