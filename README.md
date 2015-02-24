# sparco

## Installation

This code depends on the well-known `numpy`, `h5py`, and `mpi4py` packages. The command-line steps below do not cover installation of these packages. One way to gain access to them is to install [the Anaconda distribution](http://continuum.io/downloads) of Python, which includes all of the above as well as many other packages useful in scientific computing.

`sparco` also depends on a number of smaller packages: `pfacets`, `traceutil`, and `quasinewton`. These are included as git submodules, because this code may need to be installed on systems where the user is not able to install packages globally. The use of `git clone --recursive` will ensure they are installed.

`quasinewton` must be linked against BLAS. If your `cblas.h` file is located in a nonstandard location, then you must set the environment variable BLASPATH to the directory containing `cblas.h` before installation.

    git clone --recursive https://github.com/smackesey/sparco
    export BLASPATH=/path/to/my/dir/containing/cblas.h  # optional
    cd sparco/lib/quasinewton
    python setup.py build_ext --inplace

## Usage

The codebase has been designed to support the calling of CSC in multiple contexts, allowing the ability to include CSC as part of a larger data-processing pipeline. However, a default entry point for the code is provided by the `csc` script. This script allows the setting of hyperparameters, input dataset location, and output location. The dataset should be provided as one or more h5 files, all storing a main data matrix at the same path. The script is configured using command line options and/or a local configuration file.

Given the presence of this script on the path, a call to `csc.py` might look like:

    ❯ csc.py \
    --input-path /path/to/dir/with/my/h5/files \
    --output-root /path/to/output/directory \
    --local-config-path /path/to/config/file

The input path is expected to be either a single `.h5` file or a directory containing one or more `.h5` files at the top level. The output path can be an arbitrary directory. All directories in the output path that do not already exist will be created. The local config file is used to set hyperparameters and other settings. In general, anything configurable can be set in the local config file, while some can be set via command line options. `csc.py` will merge the settings defined in both places, with command line settings taking precedence over config file settings. Details on the config file format can be found below.

`csc` will run until it has hit the configured number of iterations. As of yet, there is no way to quit cleanly-- a kill/interrupt signal must be used (Ctrl-C on Unix). Snapshots of the dictionary are periodically written (the inter-write interval is configurable) to the output directory.

Assuming `mpirun` is in `$PATH`, to run the code over mpi with, say, 4 processes:

    ❯ mpirun -n4 csc.py options...

## Configuration

All configuration parameters may be specified using a local configuration file. Some may also be specified via command line options. Ultimately, most parameters are mapped to keyword arguments of the `__init__` method of a class. The default values and documentation for such parameters can be found in the source at the top of the class. Other parameters are specific to the `csc.py` script and thus have their documentation there.  All user-specified configuration overrides defaults, with command-line options overriding the configuration file.

A thorough understanding of how to configure the code is best gained by reading two sources:

- the output of `csc.py --help`, which provides a description of all available command line options. (The same information is available in the `ArgParse` specification of the `csc.py` source)
- the sample configuration file `sample_config.py`. For convenience, this file contains specifications for all possible settings as well as their corresponding documentation. In practice, it is not necessary to specify so many parameters for most use cases, since the defaults are sufficient.

### Minimal Possible Configuration

Virtually all parameters have default values that will enable the code to work. The minimal possible configuration specifies an `input_path` to the directory or h5 file containing the data. This may be done on the command line or in a configuration file.

### Configuration File Structure

A configuration file is just a python module that defines a dictionary `config`. The structure of `config` is best understood by looking at the sample configuration file provided as sample_config.py, but a brief description is provided here. `config` may contain the keys:

- `mode` : `batch` or `ladder`
- `'sampler'`: dict of keyword arguments for `sparco.sampler.Sampler.__init__`. These will be applied to the samplers for *all* Spikenets; configuration for specific instances should be provided in the `sampler` key of the `Spikenet` configuration (in `nets`).
- `'nets'`: an array of dicts, each having keyword arguments for `sparco.sp.Spikenet.__init__`
- `'trace'`: a dict containing configuration for algorithm output and logging

## Architecture

The top-level division of code is between *core*, *data-loading*, and *output* classes.

### Core

Core code implements convolutional sparse coding within RAM, without reference to the source or meaning of the data used as input. Data is obtained for each algorithm iteration through a single call to an data provider object (see [](#data-loading)).

The major steps in the CSC algorithm are "learning" (given a dataset, generation of a basis) and "inference" (given a dataset and a basis, generation of coefficients). They are implemented in separate modules that plug in to a central class `Spikenet`. `Spikenet` manages the algorithm's central loop. Over the course of this loop, algorithm parameters are held constant. It is sometimes desirable to run the algorithm several times with different configurations arranged in a serial pipeline; the output of one segment is used as the input to the next. A class `SparseCoder` implements this functionality by managing a metaloop that spins up a sequence of `Spikenet` instances.

### Data-Loading

The core accesses data by calling the `get_patches` method of a "sampler" object. In principle, any object that responds to `get_patches` with a 3d array can be used in place of an instance of the included `Sampler` class. The 3d array should have an axis order of (patch number, channel, time).

`Sampler` wraps a single HDF5 file or directory containing HDF5 files. It caches a random subset of the wrapped (2d) data in memory; `get_patches` returns a set of random fragments of this cache in the form of a 3d array. The cached subset is refreshed once a configured number of patches have been drawn from it.

### Output

In the interest of clean source code and composability of core classes, all logging, visualization, and writing of intermediate and final results to disk have been implemented in a separate software layer located in `sparco.trace`. This layer may be configured (or disabled) independently of core or data-loading components. The module structure of `sparco.trace` parallels that of `sparco`-- for each class in `sparco` for which output is desired, a corresponding `Tracer` class is implemented in `sparco.trace`. These `Tracer` classes inherit from the `Tracer` class defined in the external `traceutil` package.

`traceutil` provides a framework for applying nested decorators, logging, and capturing profiling data and state snapshots of an evolving object. This provides for complex configuration of output and periodic capture of algorithm state without cluttering the core source with output-specific meta-parameters and conditional statements. Individual `traceutil.Tracer` subclasses offer access to the profiling and other functionality in to `Tracer`, while adding custom method decorators. The custom decorators perform tasks such as writing the current basis to disk on every nth `Spikenet` iteration.
