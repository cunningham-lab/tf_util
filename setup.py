#!/usr/bin/env python

from setuptools import setup

setup(name='tf_util',
      version='0.1',
      description='Useful tensorflow libraries.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      packages=['tf_util'],
      install_requires=['tensorflow==2.5.3', 'numpy', 'scipy', ],
     )
