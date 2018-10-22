#!/usr/bin/env python

from setuptools import setup

setup(name='tf_util',
      version='1.0',
      description='Useful tensorflow libraries.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['tf_util', 'tf_util.Bron_Kerbosch'],
      install_requires=['numpy', 'scipy', 'cvxopt', 'matplotlib'],
     )
