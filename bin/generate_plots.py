#!/usr/bin/env python

# For each *.h5 files located (recursively) in the input directory, create a
# corresponding plot of the basis. Expects each h5 file to have a dataset
# called `phi`. 

import argparse
import glob
import fnmatch
import itertools
import logging
import os
import re
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for x in ['pfacets', 'traceutil', 'quasinewton']:
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__),
    '..', 'lib', x)))

import pfacets
import pfacets.matplotlib
# import sparco.trace.sp

###################################
########### PARSE COMMAND LINE ARGUMENTS
###################################

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-directory',
    help='directory containing h5 files')
arg_parser.add_argument('--separate-plots', action='store_true')
arg_parser.add_argument('--interpolation')
arg_parser.add_argument('--aspect')
arg_parser.add_argument('--origin')
args = arg_parser.parse_args()

###################################
########### MERGE CONFIGURATION
###################################

# plot_settings : dict
#   Contains settings for plots. There are two keys: `image`, which consists
#   of keyword arguments for `matplotlib.pyplot.imshow`, and `axis`, which
#   consists of keyword arguments for `matplotlib.pyplot.subplot`. All
#   parameters in both dictionaries are applied to each subplot in the output
#   grid.

defaults = {
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

config_module = pfacets.load_local_module(path=args.local_config_path,
    default_name='.sparcorc')
local_config = config_module.config if config_module is not None else {}

cli_config = pfacets.map_object_to_dict(args, {
  'interpolation': ['image', 'interpolation'],
  'aspect': ['image', 'aspect'],
  'origin': ['image', 'origin'],
  })

plot_settings = pfacets.merge(defaults, local_config, cli_config)

# TODO look into the old functionality of representing statistics on the plots
#   with rectangles in the corners
# TODO need to close this figure

def split_image()

h5_paths = pfacets.glob_recursive(args.input_directory, '*.h5')
for p in h5_paths:
  if any([os.path.basename(p) == 'basis.h5',
      (not args.separate_plots and os.path.exists(re.sub(r'\.h5$', '.png', p))),
      (args.separate_plots and os.path.exists(re.sub(r'\.h5$', '', p)))]):
    continue
  print 'plotting {0}'.format(p)
  plot_path = p.replace('.h5', '.png')
  data = h5py.File(p, 'r')['phi']
  data = np.transpose(data, (1,0,2))

  if args.separate_plots:
    odir = (os.path.join(os.path.dirname(p),
      re.search(r"\d+", os.path.basename(p)).group(0)))
    pfacets.mkdir_p(odir)
    nrows, ncols = pfacets.matplotlib.compute_grid_dimensions(len(data))
    indices = list(itertools.product(range(nrows), range(ncols)))[:len(data)]
    for i,j in indices:
      print '  plotting ({0},{1})'.format(i,j)
      plt.figure()
      plt.axes(**plot_settings['axis'])
      plt.imshow(data[i*ncols+j], **plot_settings['image'])
      plt.savefig(os.path.join(odir, "{0}_{1}.png".format(i,j)),
          bbox_inches="tight", pad_inches=0, format='png')
      plt.clf()

  else:
    fig = pfacets.matplotlib.grid_of_grids(data, im_kwargs=plot_settings['image'],
        ax_kwargs=plot_settings['axis'])
    fig.savefig(plot_path, format='png')
    fig.clf()
