#!/usr/bin/env python

from textwrap import dedent
import os
import re
import sys

import jinja2

root = os.path.normpath(os.path.join(os.path.dirname(__file__),  '..'))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'lib', 'traceutil'))
sys.path.insert(0, os.path.join(root, 'lib', 'pfacets'))

import sparco
import sparco.trace
import traceutil

########### CODE EXTRACTION METHODS
# TODO these should probably be extracted into a general-purpose lib

def get_param_dictionary(string):
  pat = '(Parameters\n.+?\n)(.+?)(^[# ]*?\n)'
  params = re.search(pat, string, re.MULTILINE | re.S).group(2)
  params = params.replace('# ', '')  # uncomment
  params = split_indented_list(params)
  return { re.match('\w+', k).group(0) : k.rstrip() for k in params }

def get_params(docstring, reject=[]):
  m = re.search("(Parameters.*?\n\s*-+\n)(.+?)(\n\n|\Z)", docstring, re.S)
  params = m.group(2)
  separated = split_indented_list(params)
  take = filter(lambda x: re.search('\w+', x).group(0) not in reject, separated)
  return "".join(take).rstrip()

def get_defaults(filepath, klass, reject=[]):
  with open(filepath, 'r') as f:
    content = f.read()
    content = re.search("(?<=class {0}).+".format(klass), content, re.S).group(0)
    # content = re.search("(?<=def __init__).+", content, re.S).group(0)
    match = re.search("(^ +)defaults([^\n]+\n)(.+)", content, re.S | re.MULTILINE)
    indent = len(match.group(1))
    content = match.group(3)
    lines = content.split("\n")
    endline = None
    for i,ln in enumerate(lines):
      if re.search("^\s{{{0}}}\S".format(indent), ln, re.MULTILINE):
        endline = i-1
        break
    defaults = "\n".join(lines[0:endline-1])
    separated = split_indented_list(defaults)
    # from IPython import embed; embed()
    take = filter(lambda x: re.search('\w+', x).group(0) not in reject, separated)
    return "".join(take).rstrip()
    # return "\n".join(lines[0:endline-1]) if endline else None

def split_indented_list(string):
  lines = string.split("\n")
  indent1 = len(re.match("^\s*", lines[0]).group(0))
  indent2 = len(re.match("^\s*", lines[1]).group(0)) if len(lines) > 1 else indent1
  pat = "\s{{{0}}}.+?(?=^\s{{{1}}}\S|\Z)".format(indent1, indent1)
  return re.findall(pat, string, re.S | re.MULTILINE)

########### UTILITY METHODS

def comment(string):
  return "\n".join(map(lambda ln: '# ' + ln, string.split("\n")))

def setindent(string, ind):
  return(indent(dedent(string), ind))

def indent(string, ind):
  lines = string.split("\n")
  return "\n".join(map(lambda ln: (' '*ind)+ln, lines))

def format(string, ind=0):
  return indent(comment(dedent(string)), ind)

config_params, trace_params = {}, {}
with open(os.path.join(root, 'bin', 'csc.py'), 'r') as f:
  content = f.read()
  config_params = get_param_dictionary(
      content[content.find('Top-Level Parameters'):])
  trace_params = get_param_dictionary(
      content[content.find('Trace Parameters'):])


# TODO the default values for some `trace` parameters are hardcoded rather than
#   dynamically loaded from `csc.py`

template = """
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

{{ comment(config_params['mode']) }}

mode = 'ladder'

###################################
########### SAMPLER
###################################

{{ comment(config_params['sampler']) }}

# Keys for `sampler`:
{{ format(get_params(sparco.sampler.Sampler.__doc__)) }}

sampler = {
{{format(get_defaults(os.path.join(root, 'sparco', 'sampler.py'), 'Sampler'), 2)}}
}

###################################
########### TRACE
###################################

{{ comment(config_params['trace']) }}

# Unless you want to access advanced output functionality, you probably
# shouldn't touch this section beyond setting `output_root`. Note that
# `output_root` can also be provided on the command line.

{{ comment("\n".join(trace_params.values())) }}

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
{{format(get_params(traceutil.Tracer.__init__.__doc__), 2)}}

  # Keys for RootSpikenet:
{{format(get_params(sparco.trace.sp.Tracer.__doc__), 2)}}

  'RootSpikenet': {
    # 'wrappermerge': True,
    # 'wrappers': {},
{{format(get_defaults(os.path.join(root, 'sparco', 'trace', 'sp.py'), 'Tracer'),4)}}
  },

  # Keys for SparseCoder:
{{format(get_params(sparco.trace.sparse_coder.Tracer.__doc__), 2)}}

  'SparseCoder': {
    # 'wrappermerge': True,
    # 'wrappers': {},
{{format(get_defaults(os.path.join(root, 'sparco', 'trace', 'sparse_coder.py'), 'Tracer'),4)}}
  }

}

###################################
########### NETS
###################################

{{ comment(config_params['nets']) }}

# This is the most important part of the configuration and the part you are
# most likely to need to adjust. The approach used here is to specify a
# template with configuration values to be used across all Spikenets, and then
# to generate a list of configurations by merging the variable parameters for
# each configuration with the template (using `pfacets.merge`, which performs a
# deep/recursive dictionary merge). The variable parameters are specified in
# the array of arrays `ladder`.

# Keys for Spikenet:
{{format(get_params(sparco.sp.Spikenet.__doc__))}}

# Keys for 'inference_settings':
{{format(get_params(sparco.sparseqn.sparseqn_batch.__doc__, reject=['phi', 'X']))}}

template = {
{{format(get_defaults(os.path.join(root, 'sparco', 'sp.py'), 'Spikenet'),2)}}
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
########### FINAL CONFIG
###################################

# Here we define the final configuration object that is accessed by run.py. 

config = {
  'mode': mode,
  'sampler': sampler,
  'trace': trace,
  'nets': nets
}
""".strip()

with open(os.path.join(root, 'examples', 'sample_config_full.py'), 'w') as f:
  output = jinja2.Template(template).render(locals())
  f.write(output)
