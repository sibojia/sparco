import functools
import glob
import re
import os

import h5py
import pfacets
import numpy as np

import time

###################################
########### UTILITY
###################################

# TODO perhaps invent a better way to do this-- currently duplicates a lot of code
#   in the class
def get_spikenet_parameters(config):
  config = pfacets.merge(Sampler.defaults, config)
  if config.get('channels') is not None:
    return { 'num_channels': len(config['channels']),
          'patch_length': config['patch_length'] }
  else:
    if re.search('.h5', config['input_path']):
      f = h5py.File(f, 'r')
    else:
      f = h5py.File(glob.glob(os.path.join(config['input_path'], '*.h5'))[0], 'r')
    data = reduce(lambda a,k: a[k], config['hdf5_data_path'], f)
    channel_dimension = int(not config['time_dimension'])
    ret = { 'num_channels': data.shape[channel_dimension],
        'patch_length': config['patch_length'] }
    f.close()
    return ret


###################################
########### SELECTION FILTERS
###################################

# TODO consider moving outside the class-- this would make sense considering
#   we want to allow for external selection filters
# TODO figure out how to configure the max lower bound, max upper bound;
# TODO perhaps using partials (having these values passed as params than bound in this function)

def std_threshold(self, **kwargs):
  """Check if the std of any channel exceeds the maximum.

  Returns
  -------
  bool
    True if any channel std exceeds max, False otherwise.
  """
  return np.any(u_std > u_max_std)

def max_u_in_bound(self, **kwargs):
  """Check that the normalized maximum patch value falls within bounds.
  
  Returns
  -------
  bool
    True if the maximum is within bounds, false otherwise.
  """
  return max_u_lower_bound < u_max < max_u_upper_bound

###################################
########### SAMPLER
###################################

class Sampler(object):
  """Provide random samples from a group of wrapped HDF5 files.

  This class must be provided with a directory `input_path`. The HDF5 files
  residing (directly, not recursively) are opened during initialization.

  The primary interface to this class is the method `get_patches`, which yields
  a 3D matrix drawn from the main dataset within one of the wrapped HDF5 files.
  The first dimension of the matrix is patch number; the other two dimensions
  represent time and channel number. The ordering of these dimensions is set by
  the `time_dimension` parameter.  particular file (this ordering must be
  provided by other means to the code that will process the patches). Repeated
  calls to `get_patches` draw from the same file until a certain number of
  patches have been drawn-- at this point, a new file is randomly selected from
  the available ones.

  Parameters (init)
  ----------
  cache_size : int (optional)
    Number of patches to load into memory at once.
  resample_cache : int (optional)
    Multiplier for the cache_size to determine the number of patches that
    should be drawn before a new cache is generated.
  hdf5_data_path : list of str (optional)
    Last element must be the name of a dataset in the wrapped hdf5 file(s). Can
    be preceded by group names.
  time_dimension : int (optional)
    Dimension of the data matrix corresponding to time.
  patch_length : int (optional)
    Number of time steps per patch
  patch_filters : list of functions (optional)
    Used to provide selection criteria for patches. Each filter is a function
    that should take a 2x2 matrix as its sole argument and return a Boolean
    value. A patch is evaluated against all patch filters and must evaluate to
    False for each one in order to be selected.
  channels : list or np.array (optional)
    A list of indices into the channel dimension of the data matrix. Selects a
    subset of channels for analysis. When omitted, all channels are used.
  """

  ########### INITIALIZATION

  defaults = {
      'cache_size': 1000,
      'subsample': 1,
      'resample_cache': 1,
      'hdf5_data_path': ['data'],
      'time_dimension': 1,
      'patch_length': 128,
      'patch_filters': [std_threshold, max_u_in_bound],
      'pre_processors': [],
      'channels': None
      }

  def __init__(self, **kwargs):
    """Configure, open hdf5 files, and load an initial cache."""
    pfacets.set_attributes_from_dicts(self, Sampler.defaults, kwargs)
    self.superpatch_length = self.patch_length * self.subsample * self.cache_size
    self.channel_dimension = int(not self.time_dimension)
    self.open_files()
    self.update_configuration_from_files()
    self.remove_short_files()
    sizes = np.array([ self.get_main_dataset(f).shape[self.time_dimension] for f in self.files ])
    self.relative_dataset_sizes = [ s / sizes.sum() for s in sizes ]
    self.patch_shape = (len(self.channels), self.patch_length)
    self.refresh_cache()
    self.pre_process_cache()

  def open_files(self):
    """Open streams to all h5 files in input directory."""
    if re.search('.h5', self.input_path):
      filenames = [self.input_path]
    else:
      filenames = glob.glob(os.path.join(self.input_path, '*.h5'))
    self.files = [ h5py.File(f, 'r') for f in filenames ]

  def close_files(self):
    """Close all streams to h5 files in input directory."""
    for f in self.files:
      f.close()

  def update_configuration_from_files(self):
    # TODO fix this docstring
    """Read metadata from hdf5 files and configure accordingly.

    Currently sets the `channels` parameter to all available channels, if a
    subset was not provided.

    Note that the full extent of this method is pending a decision about how to
    structure metadata.
    """
    if self.channels is None:
      num_channels = self.get_main_dataset(self.files[0]).shape[self.channel_dimension]
      self.channels = np.arange(num_channels)

  def remove_short_files(self):
    """Remove files that have a too-short main dataset.

    The minimum time length for the main dataset is stored in
    `superpatch_length`, set to set as `patch_length` * `subsample` *
    `cache_size`.
    """
    self.files = [ f for f in self.files
        if self.get_main_dataset(f).shape[self.time_dimension] >= self.superpatch_length]

  ########### sampling

  def refresh_cache(self):
    """Cache a subset of all available data in memory.

    Randomly selects a file and a continuous segment of the main dataset of
    that file of time-dimension length `superpatch_length`. Reads this into
    attribute `superpatch`. Takes every nth sample, where n is set by `subsample`.
    """
    ds = self.get_random_dataset()
    start = np.random.randint(0, ds.shape[self.time_dimension] - self.superpatch_length)
    self.superpatch = ds[start:start+self.superpatch_length:self.subsample, self.channels]
    self.patches_retrieved = 0

  def pre_process_cache(self):
    """Apply pre-processors to the raw superpatch."""
    for p in self.pre_processors:
      self.superpatch = p(self.superpatch)

  def get_patches(self, num):
    """Randomly select `num` valid patches from the cached superpatch.

    A valid patch is one that returns true for all functions in
    `self.patch_tests`. If necessary, rearrange axes to enforce an order of
    (patch, channel, time).

    Parameters
    ----------
    num : int
      The number of patches to get.

    Returns
    -------
    3d np.array
      The axes are ordered (patch number, channel, time).
    """
    if self.patches_retrieved > (self.cache_size * self.resample_cache):
      self.refresh_cache()
      self.pre_process_cache()
    self.patches_retrieved += num
    gen_func = functools.partial(pfacets.np.sample_array,
        self.superpatch, self.patch_length, axis=self.time_dimension)
    gen = iter(gen_func, None)
    patches = np.array(pfacets.generate_filtered(gen, self.patch_filter, num))
    return np.ascontiguousarray(
        np.transpose(patches, (0, self.channel_dimension+1, self.time_dimension+1)))

  def get_random_dataset(self):
    """Return the main dataset from a random hdf5 file.

    The probability of selecting a given file is weighted by that file's size.

    Returns
    -------
    h5py._hl.dataset.Dataset
      An hdf5 dataset holding this file's main multichannel timeseries. Behaves
      like a 2d np.array.
    """
    f = self.files[pfacets.np.weighted_randint(self.relative_dataset_sizes)]
    return self.get_main_dataset(f)

  def get_main_dataset(self, h5file):
    """Get the main matrix from an hdf5 file by descending `hdf5_data_path`.

    Parameters
    ----------
    h5file : h5py._hl.files.File
      The hdf5 file from which to extract the data matrix.

    Returns
    -------
    h5py._hl.dataset.Dataset
      An hdf5 dataset holding this file's main multichannel timeseries. Behaves
      like a 2d np.array.
    """
    return reduce(lambda a,k: a[k], self.hdf5_data_path, h5file)

  # TODO figure out what's going on with mean and variance here to fix this
  def patch_filter(self, patch):
    """Check if patch satisfies selection filters.

    Statistics are first calculated on the patch, which are fed to selection
    filters for analysis. Selection filters are Boolean-returning functions
    that take a patch as their first argument and keyword arguments `u_std`,
    `u_max_std`, and `u_max`. If any selection filter returns `True`, a patch
    is rejected.

    Parameters
    ----------
    patch : 2d np.array
      The patch being evaluated for selection.

    Returns
    -------
    bool
      `True` and `False` mean acceptance and rejection of a patch, respectively.
    """
    return True  # TODO standin; accepts all patches
    u = (patch - mean) / std
    u_std = u.std(axis=1)
    max_u = np.max(np.abs(u))
    kwargs = {'u': u, 'u_std': u_std, 'max_u': max_u}
    tests = map(lambda f: f(patch, **kwargs), self.patch_filters)
    return not (True in tests)
