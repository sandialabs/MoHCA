#!/usr/bin/env python

from setuptools import setup

with open('README.md') as f:
	long_description = f.read()
	
reqs = open('requirements.txt').readlines()
	
setup(
	name='mohca_cl',
	version='1.0.0',
	description='MOdel-free Hosting Capacity Analysis for Electric Distribution',
	long_description_content_type='text/markdown',
	long_description=long_description,
	author='Joseph Azzolini, Thad Haines, Matthew Reno',
	author_email='jazzoli@sandia.gov, jthaine@sandia.gov, mjreno@sandia.gov',
	url='https://github.com/sandialabs/MoHCA',
	packages = ['mohca_cl'],
	include_package_data=True,
	setup_requires=reqs,
	install_requires=reqs,
	entry_points={'console_scripts': 'mohca_cl = mohca_cl:init_cli'}
)
