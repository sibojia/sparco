import mpi
import sparco

# TODO better name than "adjustment parameter" for eta?

class SparseCoder(object):
  """Generate a basis for a dataset from a pipeline of sparse coding configs.

  This class is initialized with an arbitrary number of configuration
  dictionaries containing keyword arguments for the `Spikenet` class. When run,
  it iterates over each configuration, running the sparse coding algorithm in
  full. The final basis (`phi`) and adjustment parameter (`eta`) from one
  configuration is used as the starting input to the next.
  """

  def __init__(self, *spikenet_configs):
    """Set up configuration list.
    
    Parameters
    ----------
    spikenet_configs : tuple of dicts
      Each element of tuple should be a dict with keyword arguments for a
      `Spikenet#__init__`.
    """
    self.configs = spikenet_configs

  # TODO is adjustment constant the right term here
  def run(self, phi=None, eta=.00001):
    """Run each spikenet configuration.

    Parameters
    ----------
    phi : array
      A starting basis matrix. If `None`, a random one will be generated.
    eta : float
      The starting value for the adjustment constant of the basis
    """
    self.phi, self.eta = phi, eta
    for self.t, config in enumerate(self.configs):
      config['phi'], config['eta'] = self.phi, self.eta
      self.iteration(config)

  def iteration(self, config):
    """One iteration of `run` loop.

    Runs one `Spikenet` configuration and stores the resulting `phi` and `eta`.
    
    Parameters
    ----------
    config : dict
      keyword arguments for `Spikenet#__init__`
    """
    sn = self.create_spikenet(config)
    sn.run()
    self.phi, self.eta = sn.phi.copy(), sn.eta

  def create_spikenet(self, config):
    """Create a spikenet instance. Use a special class for the MPI root.

    Parameters
    ----------
    config : dict
      keyword arguments for `Spikenet#__init__`

    Returns
    -------
    Spikenet or RootSpikenet
      RootSpikenet only if on mpi root
    """
    klass = sparco.RootSpikenet if mpi.rank == mpi.root else sparco.Spikenet
    return klass(**config)
