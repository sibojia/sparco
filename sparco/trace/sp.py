import logging
import os
import pfacets

import h5py

import traceutil.tracer

# TODO need docstrings

class Tracer(traceutil.tracer.Tracer):
  """Tracer for sparco.sp.RootSpikenet.

    Parameters
    ----------
    snapshot_interval : int
      Number of iterations between successive writings of basis snapshots to disk.
  """

  defaults = {
      'snapshot_interval': 100
      }

  def __init__(self, **kwargs):
    """Configure the Tracer for a RootSpikenet."""
    kwargs = pfacets.merge(Tracer.defaults, kwargs)
    super(Tracer, self).__init__(**kwargs)

  def write_snapshot(self):
    """Write an hdf5 and/or graphical snapshot of the Spikenet's current basis to disk."""
    self.write_basis()
    self.symlink_basis()
    # if self.create_plots:
    #   self.write_plots()

  # TODO figure out the sort order and stats here
  def write_basis(self):
    """Write an hdf5 file representing the current basis to disk.

    Contains datasets for the basis itself, a sort order (based on variance),
    variance, and l0/l1/l2 norms."""
    self.basis_path = os.path.join(self.output_path, '{0}.h5'.format(self.target.t))
    h5 = h5py.File(self.basis_path, 'w')
    h5.create_dataset('phi', data=self.target.phi)
    h5.create_dataset('order', data=self.target.basis_sort_order)
    h5.create_dataset('variance', data=self.target.rootbufs.mean.a_variance)
    h5.create_dataset('l0', data=self.target.rootbufs.mean.a_l0_norm)
    h5.create_dataset('l1', data=self.target.rootbufs.mean.a_l1_norm)
    h5.create_dataset('l2', data=self.target.rootbufs.mean.a_l2_norm)
    h5.close()

  def symlink_basis(self):
    """Create a symlink to the most recently generated basis in the output directory."""
    linkf = os.path.join(self.output_path, 'basis.h5')
    try:
      os.remove(linkf)
    except:
      pass
    os.symlink(self.basis_path, linkf)

  def write_plots(self):
    pass  # TODO

  ###################################
  ########### CUSTOM DECORATORS
  ###################################

  # this stuff has to be implemneted here because I can't currently decorate
  # __init__ (since the tracer is applied AFTER object initialization)
  def t_run(tracer, orig, self, *args, **kwargs):
    tracer.dump_state(os.path.join(tracer.output_path, 'config.txt'))
    return orig(self, *args, **kwargs)

  def t_iteration(tracer, orig, self, *args, **kwargs):
    logging.info('Iteration #{0} @ {1}'.format(self.t, self.run_time))
    ret = orig(self, *args, **kwargs)
    if (self.t > 0 and tracer.snapshot_interval
        and self.t % tracer.snapshot_interval == 0):
      tracer.write_snapshot()
    return ret

  wrappers = {
      'run': [t_run],
      'iteration': [t_iteration]
      }
