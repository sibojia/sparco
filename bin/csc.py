#!/usr/bin/env python

import argparse
import copy
import glob
import json
import logging
import os
import re
import shutil
import sys
import time

import h5py

# TODO fix this-- the issue is that tokyo is not on the load path
sys.path.insert(0,
    os.path.normpath(os.path.join(os.path.dirname(__file__),  '..')))
for x in ['pfacets', 'traceutil', 'quasinewton']:
  sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__),
    '..', 'lib', x)))

import pfacets
import traceutil

import sparco
import sparco.mpi as mpi
import sparco.trace

###################################
########### PARSE COMMAND LINE ARGUMENTS
###################################

# Command line arguments are defined and expansion functions are defined for
# them. An expansion function is used to convert the string value of a
# particular argument into some other python object which cannot be directly
# represented on the command line, such as a list. The `parse_args` function
# applies the expansion functions to modify the result of
# `arg_parser.parse_args()`.

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-b', '--basis-path',
    help='path to an hdf5 file holding an initial basis matrix in dataset "phi"')
arg_parser.add_argument('--cores', default='unknown',
    help='the number of cores being requested')
arg_parser.add_argument('-c', '--channels',
    help='comma-separated string of channel ranges and individual values')
arg_parser.add_argument('-C', '--local-config-path',
    help='path to a python module containing a configuration dictionary')
arg_parser.add_argument('-i', '--input-path',
    help='path to directory containing input files')
arg_parser.add_argument('-I', '--inner-output-directory',
    help='path, relative to `output_root`, to output directory for this run')
arg_parser.add_argument('-l', '--log-level',
    help='level of logging: DEBUG, INFO, WARNING, ERROR, CRITICAL')
arg_parser.add_argument('-L', '--log-path',
    help='path to log file')
arg_parser.add_argument('--no-output', action='store_false', dest='output',
    help='do not write any output')
arg_parser.add_argument('-o', '--output-root',
    help='path to directory containing output files')
arg_parser.add_argument('--threads', default='unknown',
    help='the number of threads being used per MPI process')
arg_parser.add_argument('-r', '--resume',  # TODO needs implementation, perhaps dump readable config
    help='path to directory with output files, will resume from here')
arg_parser.add_argument('-s', '--snapshot-interval', default=100, type=int,
    help='number of iterations between basis snapshots')
arg_parser.add_argument('-t', '--time-dimension', type=int,
    help='time dimension of provided h5 files (0 or 1)')
arg_parser.set_defaults(
    output=True
    )

def expand_channel_spec(channel_spec):
  parts = channel_spec.split(',')
  channels = []
  for p in parts:
    if re.search('-', p):
      start, end = p.split('-')
      channels += range(int(start), int(end)+1)
    else:
      channels.append(int(p))
  return channels

def parse_args():
  global arg_parser
  args = arg_parser.parse_args()
  if args.channels:
    args.channels = expand_channel_spec(args.channels)
  return args

###################################
########### MERGE CONFIGURATION
###################################

# Configuration dictionaries derived from a local configuration file and
# command line arguments are merged with the below-defined defaults. After the
# merge, a few parameters that depend upon other configuration values are set.

# Top-Level Parameters
# --------------------
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
# sampler : dict
#   Keyword arguments for initialization of a `Sampler` instance. Configuration
#   provided here will be used globally-- i.e. it will be merged into the
#   `Sampler` instances corresponding to each `Spikenet`. To set
#   spikenet-specific `Sampler` parameters, define these settings in the
#   `sampler_settings` key for a `Spikenet` configuration.
# trace : dict
#   Configuration for the `sparco.trace` classes corresponding to
#   `sparco` classes. Used to configure output. See README for an explanation
#   of the relationship. Some keys are the names of `sparco` classes; the values
#   consist of keyword arguments for the initialization of the corresponding
#   tracer.
# nets : list
#   List of configuration dictionaries for `Spikenet` instances. Each dictionary
#   contains keyword arguments for initialization of a Spikenet.  
#
# Trace Parameters
# ----------------
# enable: bool (optional)
#   When `True`, create output. When False, no output-- all other trace
#   settings are ignored.
# config_key_function : function
#   Function that takes a `Spikenet` config as its sole argument and returns a
#   string representing that config.
# output_root : str (optional)
#   The root directory for all output. All output is written to paths relative
#   to this one.
# inner_output_directory : str (optional)
#   Path, relative to `output_root`, to output directory for this run.
# RootSpikenet : dict (optional)
#   Keyword arguments for sparco.trace.sp.Tracer.__init__
# SparseCoder : dict (optional)
#   Keyword arguments for sparco.trace.sparse_coder.Tracer.__init__
# log : dict (optional)
#   Keyword arguments for logging.basicConfig -- see standard library
#   documentation. If no filename is provided, it will be set to
#   `output_root`/sparco.log.

defaults = {
  'mode': 'ladder',
  'sampler': {},
  'trace': {
    'enable': True,
    'config_key_function': sparco.sp.config_key,
    'output_root': None,
    'inner_output_directory': None,
    'RootSpikenet': {},
    'SparseCoder': {},
    'log': {
      'filename': None,
      # 'stream': sys.stdout,
      'filemode': 'a+',
      'format': '%(asctime)s %(message)s',
      'level': 'DEBUG'
      }
    },
  'nets': []
  }

if mpi.rank == mpi.root:

  args = parse_args()
  cli_config = pfacets.map_object_to_dict(args, {
      'channels': ['sampler', 'channels'],
      'input_path': ['sampler', 'input_path'],
      'time_dimension': ['sampler', 'time_dimension'],
      'inner_output_directory': ['trace', 'inner_output_directory'],
      'log_level': ['trace', 'log', 'level'],
      'log_path': ['trace', 'log', 'filename'],
      'output': ['trace', 'enable'],
      'output_root': ['trace', 'output_root'],
      'snapshot_interval': ['trace', 'RootSpikenet', 'snapshot_interval'],
      })

  if args.resume is not None:
    with open(os.path.join(args.resume, 'cli_config.json'), 'r') as f:
      prev_cli_config = json.loads(f.read())
    cli_config = pfacets.merge(prev_cli_config, cli_config)
    local_config_path = glob.glob(os.path.join(args.resume, '*.py'))[0]
  else:
    local_config_path = args.local_config_path
else:
  local_config_path = None
  cli_config = None

local_config_path = mpi.bcast_obj(local_config_path)
cli_config = mpi.bcast_obj(cli_config)

# all procs must load the local config because it contains unbound methods,
# which can't be pickled

local_config = pfacets.load_local_module(local_config_path,
    default_name='.sparcorc').config or {}
config = pfacets.merge(defaults, local_config, cli_config)

########### DERIVED AND DYNAMIC DEFAULT PARAMETERS

if mpi.rank != mpi.root:
  outdir = None
else:
  if args.resume is not None:
    outdir = args.resume
  else:
    default_inner_dir = "{0}_{1}".format(time.strftime('%y%m%d%H%M%S'), config['mode'])
    outdir = os.path.join(
        config['trace']['output_root'],
        config['trace']['inner_output_directory'] or default_inner_dir)

config['trace']['inner_output_directory'] = mpi.bcast_obj(outdir)


config['trace']['SparseCoder']['config_key_function'] = config['trace']['config_key_function']


if isinstance(config['trace']['log']['level'], str):
  config['trace']['log']['level'] = getattr(logging,
      config['trace']['log']['level'].upper())

if len(config['nets']) == 0:
  config['nets'].append({})

  # if args.basis_path not in [None, 'none']:  # TODO hack; need better way to optional pass cli args from csc.zsh
  #   config['nets'][0]['phi'] = h5py.File(args.basis_path, 'r')['phi']

for c in config['nets']:
  c['sampler_settings'] = pfacets.merge(c.get('sampler_settings', {}), config['sampler'])
  c.update(sparco.sampler.get_spikenet_parameters(c['sampler_settings']))

if mpi.rank == mpi.root: 
  if config['trace']['enable']:
    pfacets.mkdir_p(config['trace']['inner_output_directory'])

########### SAVE CONFIGURATION FOR FUTURE

if mpi.rank == mpi.root:
  logging.basicConfig(**config['trace']['log'])
  if args.local_config_path:
    shutil.copy(args.local_config_path, config['trace']['inner_output_directory'])
  cli_config_path = os.path.join(config['trace']['inner_output_directory'],
      'cli_config.json')
  with open(cli_config_path, 'w') as f:
    f.write(json.dumps(cli_config, sort_keys=True, indent=4, separators=(',', ': ')))
  resources_config_path = os.path.join(config['trace']['inner_output_directory'],
      'resources.txt')
  with open(resources_config_path, 'w') as f:
    f.write("cores {0}\nthreads{1}".format(args.cores, args.threads))

# it will need to go through expansion again so we don't dump everything

###################################
########### RESTORE PREVIOUS
###################################

def get_last_basis_snapshot(path):
  dirs = glob.glob(os.path.join(path, '*/'))
  dirs = sorted(dirs,
      key=lambda d: int(re.search(r'^\d+', os.path.basename(os.path.normpath(d))).group(0)))
  lastdir = dirs[-1] if len(dirs) >= 1 else None
  if lastdir is not None:
    files = glob.glob(os.path.join(lastdir, '[0-9]*.h5'))
    if len(files) >= 1:
      last = max(files, key=lambda f: int(re.search(r'^\d+', os.path.basename(f)).group(0)))
      return last
  return None

if mpi.rank == mpi.root:
  last_snapshot = get_last_basis_snapshot(
    os.path.join(config['trace']['inner_output_directory']))
  if last_snapshot is not None:
    ladder_or_batch_index = int(os.path.basename(os.path.dirname(last_snapshot)).split('_')[0])
    iteration_number = int(os.path.basename(last_snapshot).split('.')[0]) + 1
    logging.info('resuming ladder/batch {0} @ iteration {1}...'.format(
      ladder_or_batch_index, iteration_number))
    config['nets'][ladder_or_batch_index]['starting_iteration'] = iteration_number
    config['nets'][ladder_or_batch_index]['phi'] = h5py.File(last_snapshot, 'r')['phi']
  else:
    ladder_or_batch_index = 0
else:
  ladder_or_batch_index = 0

###################################
########### RUN
###################################

# Run convolutional sparse coding using the defined configuration. Perform
# logging configuration and directory setup if output is enabled.

if config['mode'] == 'ladder':
  sc = sparco.SparseCoder(config['nets'], start_index=ladder_or_batch_index)
  if config['trace']['enable']:
    traceutil.tracer.apply_tracer(sparco.trace.sparse_coder.Tracer,
        target=sc, RootSpikenet_config=config['trace']['RootSpikenet'],
        output_path=config['trace']['inner_output_directory'],
        **config['trace']['SparseCoder'])
  sc.run()

elif config['mode'] == 'batch':
  start_time = time.time()
  for c in config['nets']:
    if mpi.rank == mpi.root:
      sn = sparco.sp.RootSpikenet(**c)
      if config['trace']['enable']:
        output_path = os.path.join(config['trace']['inner_output_directory'],
            config['trace']['config_key_function'](sn))
        traceutil.tracer.apply_tracer(sparco.trace.sp.Tracer,
            target=sn, output_path=output_path, **config['trace']['RootSpikenet'])
    else:
      sn = sparco.sp.Spikenet(**c)
    sn.run()
