import copy
import json
import logging
import os
import pfacets

import h5py
import numpy as np

import sparco.mpi as mpi
import traceutil.tracer

# TODO need docstrings

class Tracer(traceutil.tracer.Tracer):
  """Tracer for sparco.sp.RootSpikenet.

    Parameters
    ----------
    snapshot_interval : int
      Number of iterations between successive writings of basis snapshots to disk.
    dump_keys : list
      Keys from the Spikenet configuration that are dumped as JSON.
  """

  defaults = {
      'snapshot_interval': 100,
      }

  def __init__(self, **kwargs):
    """Configure the Tracer for a RootSpikenet."""
    kwargs = pfacets.merge(Tracer.defaults, kwargs)
    super(Tracer, self).__init__(**kwargs)

  def dump_state(self, path=None):
    special = {
        'phi': lambda x: 'starting value' if x is not None else 'random start',
        'inference_function': lambda x: x.__name__,
        'learner_class': lambda x: x.__name__,
        }
    exclude = ['last_time', 'run_time']
    d = self.target.__dict__
    state = {}
    for k in d.keys():
      if k in special.keys():
        special[k](d[k])
      elif not k in exclude and type(d[k]) in [int, str, list, float]:
        state[k] = d[k]
      else:
        continue
    with open(path, 'w') as f:
      f.write(json.dumps(state, sort_keys=True, indent=4, separators=(',', ': ')))

  def write_snapshot(self):
    """Write an hdf5 snapshot of the Spikenet's current basis to disk."""
    self.write_basis_h5()

  # TODO figure out the sort order and stats here
  def write_basis_h5(self):
    """Write an hdf5 file representing the current basis to disk.

    Contains datasets for the basis itself, a sort order (based on variance),
    variance, and l0/l1/l2 norms."""
    path = os.path.join(self.output_path, '{0}.h5'.format(self.target.t))
    h5 = h5py.File(path, 'w')
    h5.create_dataset('phi', data=self.target.phi)
    h5.create_dataset('order', data=self.target.basis_sort_order)
    h5.create_dataset('variance', data=self.target.rootbufs.a_variance)
    h5.create_dataset('l0', data=self.target.rootbufs.a_l0_norm)
    h5.create_dataset('l1', data=self.target.rootbufs.a_l1_norm)
    h5.create_dataset('l2', data=self.target.rootbufs.a_l2_norm)
    h5.close()

  ###################################
  ########### CUSTOM DECORATORS
  ###################################

  # this stuff has to be implemneted here because I can't currently decorate
  # __init__ (since the tracer is applied AFTER object initialization)
  def t_run(tracer, orig, self, *args, **kwargs):
    tracer.dump_state(os.path.join(tracer.output_path, 'config.json'))
    return orig(self, *args, **kwargs)

  def t_iteration(tracer, orig, self, *args, **kwargs):
    logging.info('Iteration #{0} @ {1}'.format(self.t, self.run_time))
    ret = orig(self, *args, **kwargs)
    if (self.t > 0 and tracer.snapshot_interval
        and self.t % tracer.snapshot_interval == 0):
      tracer.write_snapshot()
    return ret

  def t_within_time_limit(tracer, orig, self, *args, **kwargs):
    ret = orig(self, *args, **kwargs)
    logging.info("run time is {0} on rank {1}".format(self.run_time, mpi.rank))
    return ret

  wrappers = {
      'run': [t_run],
      'iteration': [t_iteration],
      'within_time_limit': [t_within_time_limit]
      }
