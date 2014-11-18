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
import os.path as osp
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
import pfacets.matplotlib as pmpl
# import sparco.trace.sp

###################################
########### PARSE COMMAND LINE ARGUMENTS
###################################

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-directory',
    help='directory containing h5 files')
arg_parser.add_argument('-o', '--output-root')
arg_parser.add_argument('-p', '--posmat-path')
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
posmat = np.int32(h5py.File(args.posmat_path, 'r')['electrode_layout'])
# TODO look into the old functionality of representing statistics on the plots
#   with rectangles in the corners
# TODO need to close this figure

# def split_image()

outdir = osp.join(args.output_root, osp.basename(args.input_directory))
pfacets.mkdir_p(outdir)

def row(data, buf=10):
  S, E, T = data.shape  # shank/electrode/time
  xsize = T*S + buf*(S-1)
  im = np.zeros((E, xsize))
  for i in range(len(data)):
    xs = i*(T+buf); xe = xs+T
    im[:,xs:xe] = data[i]
  return im

def time_row(data, buf=10):
  S, E, T = data.shape  # shank/electrode/time
  xsize = T*S + buf*(T-1)
  im = np.zeros((E, xsize))
  tr = np.transpose(data, (2,1,0))
  # from IPython import embed; embed()
  for i in range(len(tr)):
    xs = i*(S+buf); xe = xs+S
    im[:,xs:xe] = tr[i]
  return im

# # expects 16x32xt
# def shank_plot(data)

def get_rearr(data):
  global posmat
  rearr = np.zeros((16,32,data.shape[1]))
  for i in range(len(posmat)):
    indices = posmat[i,:]
    shank = data[indices,:]
    rearr[i] = shank
  return rearr

# expects channels x time
def shank_frames(data):
  rearr = get_rearr(data)
  buf=5
  ary1, ary2 = row(rearr[0:8], buf=buf), row(rearr[8:16], buf=buf)
  S, E, T = rearr.shape  # shank/electrode/time
  xsize = T*S/2 + buf*(S/2-1)
  rowbuffer = np.zeros((buf,xsize))
  # from IPython import embed; embed()
  im = np.vstack((ary1, rowbuffer, ary2))
  return im

# expects channels x time
def time_frames(data):
  rearr = get_rearr(data)
  buf=2
  ary1, ary2 = time_row(rearr[0:8], buf=buf), time_row(rearr[8:16], buf=buf)
  S, E, T = rearr.shape  # shank/electrode/time
  xsize = T*S/2 + buf*(T-1)
  rowbuffer = np.zeros((5,xsize))
  # from IPython import embed; embed()
  im = np.vstack((ary1, rowbuffer, ary2))
  return im

def sort_paths(paths):
  def keyfunc(p):
    pat = r'^\d+'
    gnum = int(re.search(pat, osp.basename(osp.dirname(p))).group(0))
    itnum = int(re.search(pat, osp.basename(p)).group(0))
    return gnum * 10000 + itnum
  # from IPython import embed; embed()
  x = sorted(paths, key=keyfunc)
  # x.reverse()
  return x

h5_paths = sort_paths(pfacets.glob_recursive(args.input_directory, '[0-9]*.h5'))
# from IPython import embed; embed()
for p in h5_paths:
  if os.path.basename(p) == 'basis.h5':
  # if any([os.path.basename(p) == 'basis.h5',
  #     (not args.separate_plots and os.path.exists(re.sub(r'\.h5$', '.png', p))),
  #     (args.separate_plots and os.path.exists(re.sub(r'\.h5$', '', p)))]):
    continue
  print 'plotting {0}'.format(p)
  plot_path = p.replace('.h5', '').replace(args.input_directory, outdir)
  data = h5py.File(p, 'r')['phi']
  data = np.transpose(data, (1,0,2))

  if args.separate_plots:
    # odir = (os.path.join(plot_path
    #   re.search(r"\d+", os.path.basename(p)).group(0)))
    odir = plot_path
    pfacets.mkdir_p(odir)
    # nrows, ncols = pfacets.matplotlib.compute_grid_dimensions(len(data))
    # indices = list(itertools.product(range(nrows), range(ncols)))[:len(data)]
    # for i,j in indices:
      # print '  plotting ({0},{1})'.format(i,j)
    for i in range(len(data)):
      print '  plotting basis function {0}'.format(i)
      # plt.imshow(data[i*ncols+j], **plot_settings['image'])
      # plt.imshow(data[i], **plot_settings['image'])
      # plt.savefig(os.path.join(odir, "{0}.png".format(i)),
      #     bbox_inches="tight", pad_inches=0, format='png')
      
      sp = os.path.join(odir, "{0}_shankwise.png".format(i))
      if not osp.exists(sp):
        plt.figure()
        plt.axes(**plot_settings['axis'])
        plt.imshow(shank_frames(data[i]), **plot_settings['image'])
        plt.savefig(sp, bbox_inches="tight", pad_inches=0, format='png')
        plt.clf()
      tp = os.path.join(odir, "{0}_timewise.png".format(i))
      if not osp.exists(tp):
        plt.figure()
        plt.axes(**plot_settings['axis'])
        plt.imshow(time_frames(data[i]), **plot_settings['image'])
        plt.savefig(tp, bbox_inches="tight", pad_inches=0, format='png')
        plt.clf()

  else:
    fig = pfacets.matplotlib.grid_of_grids(data, im_kwargs=plot_settings['image'],
        ax_kwargs=plot_settings['axis'])
    fig.savefig(plot_path, format='png')
    fig.clf()
