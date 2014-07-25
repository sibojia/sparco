import logging
import os
import pfacets
import traceutil.tracer
import sparco.trace.sp

# intended for RootSpikenet

class Tracer(traceutil.tracer.Tracer):
  """Tracer for sparco.sparse_coder.SparseCoder.

  Parameters
  ----------
  config_key_function : function
    Function that takes a `Spikenet` config as its sole argument and returns a
    string representing that config.
  RootSpikenet_config : dict
    Keyword arguments for sparco.trace.sp.Tracer.__init__
  """

  defaults = {
      'config_key_function': sparco.sp.config_key,
      'RootSpikenet_config': {}
      }

  def __init__(self, **kwargs):
    """Configure Tracer for SparseCoder."""
    kwargs = pfacets.merge(Tracer.defaults, kwargs)
    super(Tracer, self).__init__(**kwargs)

  ###################################
  ########### CUSTOM DECORATORS
  ###################################

  def t_run(tracer, orig, self, *args, **kwargs):
    logging.info('Beginning new SparseCoder run...')
    return orig(self, *args, **kwargs)

  def t_create_spikenet(tracer, orig, self, *args, **kwargs):
    sn = orig(self, *args, **kwargs)
    key = tracer.config_key_function(sn)
    logging.info('Round {0}: {1}'.format(self.t, key))
    sn_output_path = os.path.join(tracer.output_path, '{0}_{1}'.format(self.t, key))
    traceutil.tracer.apply_tracer(sparco.trace.sp.Tracer,
        output_path=sn_output_path, target=sn, **tracer.RootSpikenet_config)
    return sn

  wrappers = {
      'run': [t_run],
      'create_spikenet': [t_create_spikenet],
      }
