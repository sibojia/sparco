from __future__ import print_function

from distutils.extension import Extension
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from Cython.Build import cythonize

import io
import os
import sys

import numpy
import os

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.md', 'CHANGES.md')

setup(
    name='sparco',
    version='0.0.1',
    url='http://github.com/ursk/sparco/',
    license='MIT',
    author='Amir Khosrowshahi, Urs Koster, Sean Mackesey',
    author_email='s.mackesey@gmail.com',
    description='convolutional sparse coding implemented with openMPI',
    long_description=long_description,
    tests_require=['pytest'],
    install_requires=[
      'traceutil',
      'pfacets'
      ],
    packages=[''],
    include_package_data=True,
    platforms='any',
)
