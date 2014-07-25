import logging
import os
import pfacets
import pfacets.matplotlib

import h5py
import numpy as np
import matplotlib.pyplot as plt

import traceutil.tracer

# TODO need docstrings
# TODO better describe the keyword arguments passed inside plot

class Tracer(traceutil.tracer.Tracer):
  """Tracer for sparco.sp.RootSpikenet.

    Parameters
    ----------
    create_plots : bool
      Generate a grid of grids in png format
    snapshot_interval : int
      Number of iterations between successive writings of basis snapshots to disk.
    plot_settings : dict
      Contains settings for plots. There are two keys: `image`, which consists
      of keyword arguments for `matplotlib.pyplot.imshow`, and `axis`, which
      consists of keyword arguments for `matplotlib.pyplot.subplot`. All
      parameters in both dictionaries are applied to each subplot in the output
      grid.
  """

  defaults = {
      'create_plots': True,
      'snapshot_interval': 100,
      'plot_settings': {
        'image': {
          'cmap': plt.cm.jet,
          'interpolation': 'nearest',
          'aspect': 'equal',
          'origin': 'upper',
          },
        'axis': {
          'xticks': [],
          'yticks': [],
          'frame_on': False
          }
        }
      }

  def __init__(self, **kwargs):
    """Configure the Tracer for a RootSpikenet."""
    kwargs = pfacets.merge(Tracer.defaults, kwargs)
    super(Tracer, self).__init__(**kwargs)

  def write_snapshot(self):
    """Write an hdf5 and/or graphical snapshot of the Spikenet's current basis to disk."""
    self.write_basis_h5()
    if self.create_plots:
      self.write_basis_plots()

  # TODO figure out the sort order and stats here
  def write_basis_h5(self):
    """Write an hdf5 file representing the current basis to disk.

    Contains datasets for the basis itself, a sort order (based on variance),
    variance, and l0/l1/l2 norms."""
    path = os.path.join(self.output_path, '{0}.h5'.format(self.target.t))
    h5 = h5py.File(path, 'w')
    h5.create_dataset('phi', data=self.target.phi)
    h5.create_dataset('order', data=self.target.basis_sort_order)
    h5.create_dataset('variance', data=self.target.rootbufs.mean.a_variance)
    h5.create_dataset('l0', data=self.target.rootbufs.mean.a_l0_norm)
    h5.create_dataset('l1', data=self.target.rootbufs.mean.a_l1_norm)
    h5.create_dataset('l2', data=self.target.rootbufs.mean.a_l2_norm)
    h5.close()
    pfacets.symlink_f(path, os.path.join(self.output_path, 'basis.h5'))

  # TODO look into the old functionality of representing statistics on the plots
  #   with rectangles in the corners
  # TODO need to close this figure
  def write_basis_plots(self):
    fig = pfacets.matplotlib.grid_of_grids(np.transpose(self.target.phi, (1,0,2)),
        ax_kwargs=self.plot_settings['axis'], im_kwargs=self.plot_settings['image'])
    path = os.path.join(self.output_path, '{0}.png'.format(self.target.t))
    fig.savefig(path, format='png')
    fig.clf()
    pfacets.symlink_f(path, os.path.join(self.output_path, 'basis.png'))

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
