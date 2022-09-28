#!/usr/bin/python3

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
	packages = ['sophon_robot'],
	package_dir = {'':''})

setup(**d)
