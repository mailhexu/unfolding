#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='unfolding',
    version='0.1',
    description='Bloch wave unfolding',
    author='Xu He',
    author_email='mailhexu@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    install_requires=['numpy','matplotlib','ase'],
    scripts=[],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GPLv3 license',
    ])
