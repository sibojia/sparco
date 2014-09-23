# The below is a maximally thorough sample configuration file. It presents all
# available configuration parameters by copying the documentation and default
# values from the source code. Because the provided values are the same as the
# defaults, simply uncommenting the lines specifying particular options should
# have no effect on configuration. The recommended approach is to use this file
# as a reference to write a much shorter configuration file, in which any values
# being left as defaults are omitted. Alternatively, one may copy this file and
# uncomment and change the value of individual lines as needed.

###################################
########### MODE
###################################

# mode : 'ladder' or 'batch' (str)
#   Two modes are provided: 'ladder' and 'batch'. Both modes execute a sequence
#   of `Spikenet` configurations. Both modes create a timestamped directory
#   located in `output_root`, in which are placed directories containing the
#   output of each individual `Spikenet`.  Hence the output directory structure
#   is analogous to root/ladder_or_batch_dir/spikenet1,
#   root/ladder_or_batch_dir/spikenet2, etc. The difference between modes is
#   that 'ladder' mode uses the final basis and eta values of each Spikenet as
#   the initial values for the next Spikenet, while 'batch' mode initializes
#   phi and eta independently for each `Spikenet`.

mode = 'ladder'

###################################
########### SAMPLER
###################################

# sampler : dict
#   Keyword arguments for initialization of a `Sampler` instance. Configuration
#   provided here will be used globally-- i.e. it will be merged into the
#   `Sampler` instances corresponding to each `Spikenet`. To set
#   spikenet-specific `Sampler` parameters, define these settings in the
#   `sampler_settings` key for a `Spikenet` configuration.

# Keys for `sampler`:
# cache_size : int (optional)
#   Number of patches to load into memory at once.
# resample_cache : int (optional)
#   Multiplier for the cache_size to determine the number of patches that
#   should be drawn before a new cache is generated.
# hdf5_data_path : list of str (optional)
#   Last element must be the name of a dataset in the wrapped hdf5 file(s). Can
#   be preceded by group names.
# time_dimension : int (optional)
#   Dimension of the data matrix corresponding to time.
# patch_length : int (optional)
#   Number of time steps per patch
# patch_filters : list of functions (optional)
#   Used to provide selection criteria for patches. Each filter is a function
#   that should take a 2x2 matrix as its sole argument and return a Boolean
#   value. A patch is evaluated against all patch filters and must evaluate to
#   False for each one in order to be selected.
# channels : list or np.array (optional)
#   A list of indices into the channel dimension of the data matrix. Selects a
#   subset of channels for analysis. When omitted, all channels are used.

sampler = {
  # 'cache_size': 1000,
  # 'subsample': 1,
  # 'resample_cache': 1,
  # 'hdf5_data_path': ['data'],
  # 'time_dimension': 1,
  # 'patch_length': 128,
  # 'patch_filters': [std_threshold, max_u_in_bound],
  # 'pre_processors': [],
  # 'channels': None
}

###################################
########### TRACE
###################################

# trace : dict
#   Configuration for the `sparco.trace` classes corresponding to
#   `sparco` classes. Used to configure output. See README for an explanation
#   of the relationship. Some keys are the names of `sparco` classes; the values
#   consist of keyword arguments for the initialization of the corresponding
#   tracer.

# Unless you want to access advanced output functionality, you probably
# shouldn't touch this section beyond setting `output_root`. Note that
# `output_root` can also be provided on the command line.

# enable: bool (optional)
#   When `True`, create output. When False, no output-- all other trace
#   settings are ignored.
# log : dict (optional)
#   Keyword arguments for logging.basicConfig -- see standard library
#   documentation. If no filename is provided, it will be set to
#   `output_root`/sparco.log.
# config_key_function : function
#   Function that takes a `Spikenet` config as its sole argument and returns a
#   string representing that config.
# inner_output_directory : str (optional)
#   Path, relative to `output_root`, to output directory for this run.
# RootSpikenet : dict (optional)
#   Keyword arguments for sparco.trace.sp.Tracer.__init__
# SparseCoder : dict (optional)
#   Keyword arguments for sparco.trace.sparse_coder.Tracer.__init__
# output_root : str (optional)
#   The root directory for all output. All output is written to paths relative
#   to this one.

trace = {

  # 'enable': True,
  # 'output_root': None,
  # 'log': {
  #   'filename': None,
  #   'filemode': 'a+',
  #   'format': '%(asctime)s %(message)s',
  #   'level': 'DEBUG'
  #   }

  # Keys for all tracers (from traceutil.Tracer): 
  # target : obj
  #   the object being wrapped by this tracer
  # output_path : str
  #   path to a directory where this tracer will write data
  # wrappers : dict
  #   keys are method names of the wrapped object; values are lists of wrappers
  #   that are successively applied (nested) on the function. Wrapper functions
  #   must satisfy the wrapper specification.
  # wrappermerge : bool
  #   If True, the lists specified in `wrappers` will be appended to the
  #   corresponding lists defined in the `wrappers` class variable of the
  #   current class (note that this is only present in subclasses). If false,
  #   the lists are overwritten rather than appended to.

  # Keys for RootSpikenet:
  # snapshot_interval : int
  #   Number of iterations between successive writings of basis snapshots to disk.

  'RootSpikenet': {
    # 'wrappermerge': True,
    # 'wrappers': {},
    # 'snapshot_interval': 100,
  },

  # Keys for SparseCoder:
  # config_key_function : function
  #   Function that takes a `Spikenet` config as its sole argument and returns a
  #   string representing that config.
  # RootSpikenet_config : dict
  #   Keyword arguments for sparco.trace.sp.Tracer.__init__

  'SparseCoder': {
    # 'wrappermerge': True,
    # 'wrappers': {},
    # 'config_key_function': sparco.sp.config_key,
    # 'RootSpikenet_config': {}
  }

}

###################################
########### NETS
###################################

# nets : list
#   List of configuration dictionaries for `Spikenet` instances. Each dictionary
#   contains keyword arguments for initialization of a Spikenet.

# This is the most important part of the configuration and the part you are
# most likely to need to adjust. The approach used here is to specify a
# template with configuration values to be used across all Spikenets, and then
# to generate a list of configurations by merging the variable parameters for
# each configuration with the template (using `pfacets.merge`, which performs a
# deep/recursive dictionary merge). The variable parameters are specified in
# the array of arrays `ladder`.

# Keys for Spikenet:
# sampler : Sampler or other class with `get_patches`
#   The data provider. Must respond to `get_patches` with a 3d numpy array.
# patches_per_iteration : int (optional)
#   Number of patches to process in a single iteration.
# num_iterations : int (optional)
#   Total number of iterations to carry out before terminating.
# run_time_limit : number (optional)
#   Execution will terminate if this amount of time is exceeded.
# dictionary_size : int (optional)
#   The number of basis functions to generate.
# convolution_time_length : int (optional)
#   The length of each basis function in the time dimension.
# phi : 3d array (optional)
#   A starting set of basis functions.
# inference_function : function
#   A function that satisfies the inference function specification.
# inference_settings : dict
#   Keyword arguments for inference_function.
# eta : float
#   A starting value for the basis adjustment constant.
# learner_class : class
#   A class implementing the learner class specification.
# eta_up_factor : float
#   Factor by which eta will be adjusted if basis angle is below target.
# eta_down_factor : float
#   Factor by which eta will be adjusted if basis angle is above target.
# target_angle : number
#   Target angle between successive states of phi 
# max_angle : number
#   Maximum tolerable angle between successive states of phi. Angles above
#   this will cause the rejection of the most recently calculated phi.
# update_coefficient_statistics_interval : int
#   The number of iterations between updates to coefficient statistics
#   (l0/l1/l2 norm, variance).
# basis_centering_interval : int or None
#   The number of iterations between centerings of the basis.
# basis_centering_max_shift : number
#   The maximum shift for each basis function during centering.

# Keys for 'inference_settings':
# lam : float
#   L1 penalty
# maxit : int
#   Maximum number of quasi-newton iterations
# positive : bool
#   If True, only allow positive coefficients
# Sin : 2d array
#   Starting value for coefficients; if None, zeros are used.
# debug : bool
#   Print debugging information
# delta : int
#   ?????
# past : int
#   ?????
# mask : 2d array
#   An array of dims of coefficients used to mask derivative.

template = {
  # 'sampler_settings': None,
  # 'patches_per_iteration': mpi.procs,
  # 'num_iterations': 100,
  # 'run_time_limit': float("inf"),
  # 'dictionary_size': 100,
  # 'convolution_time_length': 64,
  # 'phi': None,
  # 'inference_function': sparco.sparseqn.sparseqn_batch,
  # 'inference_settings': {
  #   'lam': 0,
  #   'maxit': 15,
  #   'debug': False,
  #   'positive': False,
  #   'delta': 0.0001,
  #   'past': 6
  #   },
  # 'eta': .00001,
  # 'learner_class': sparco.learn.AngleChasingLearner,
  # 'eta_up_factor': 1.01,
  # 'eta_down_factor': .99,
  # 'target_angle': 1.,
  # 'max_angle': 2.,
  # 'update_coefficient_statistics_interval': 100,
  # 'basis_centering_interval': None,
  # 'basis_centering_max_shift': None,
  # 'basis_method': 1,  # TODO this is a temporary measure
}

# lam, maxit, num_iterations, target_angle
ladder = [[0.1, 5,  2000,  5.],
          [0.3, 10, 2000,  2.],
          [0.5, 20, 2000,  2.],
          [0.7, 25, 4000,  1.0],
          [0.9, 30, 10000, 0.5],
          [1.0, 35, 40000, 0.5]]

for lam, maxit, num_iterations, target_angle in ladder:
  variable = {
      'inference_settings': {
        'lam': lam,
        'maxit': maxit
        },
      'num_iterations': num_iterations,
      'target_angle': target_angle
      }
  nets.append(pfacets.merge(template, variable))

###################################
########### PLOTS
###################################

# This section is used for generate_plots.py only-- all plotting code is
# encapsulated here. csc.py does not use this. 

# Keys for `generate_plots_config`:
# image : dict
#   keyword arguments for `matplotlib.pyplot.imshow`
# axis :
#   keyword arguments for `matplotlib.pyplot.subplot`.

# All parameters in both dictionaries are applied to each subplot in the output
# grid.

generate_plots_config = {
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

###################################
########### FINAL CONFIG
###################################

# Here we define the final configuration object that is accessed by csc.py. 

config = {
  'mode': mode,
  'sampler': sampler,
  'trace': trace,
  'nets': nets
}