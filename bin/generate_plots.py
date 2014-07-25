#!/usr/bin/env python

# For each *.h5 files located (recursively) in the input directory, create a corresponding plot of the basis. Expects each h5 file to have a dataset called `phi`. 

import argparse
import glob
import logging
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
for x in ['pfacets', 'traceutil', 'quasinewton']:
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__),
    '..', 'lib', x)))

import pfacets.matplotlib
import sparco.trace.sp

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-directory',
    help='directory containing h5 files')
args = arg_parser.parse_args()

local_config = pfacets.load_local_module(path=args.local_config_path,
    default_name='.sparcorc').config or {}

plot_settings = pfacets.merge(sparco.trace.sp.Tracer.defaults['plot_settings'],
    pfacets.get_recursive(local_config, {}, 'trace', 'RootSpikenet', 'plot_settings'))

h5_paths = glob.glob(os.path.join(args.input_directory, '**', '*.h5'))
for p in h5_paths:
  print 'plotting {0}'.format(p)
  plot_path = p.replace('.h5', '.png')
  data = h5py.File(p, 'r')['phi']
  data = np.transpose(data, (1,0,2))
  fig = pfacets.matplotlib.grid_of_grids(data, im_kwargs=plot_settings['image'],
      ax_kwargs=plot_settings['axis'])
  fig.savefig(plot_path, format='png')
  fig.clf()
