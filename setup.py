#!/usr/bin/env python
import sys
from setuptools import setup, find_packages


if sys.version_info[0] != 3:
    raise SystemError("Python 3 is required!")

REQUIREMENTS = open('requirements.txt', encoding='utf-8').readlines()
REQUIREMENTS = [req.rstrip() for req in REQUIREMENTS]

setup(
    name='LISO',
    version='0.1.0',
    packages=find_packages(),
    author='Jun Zhu',
    author_email='zhujun981661@gmail.com',
    url='https://github.com/zhujun98/liso',
    download_url='https://github.com/zhujun98/liso',
    description='Python API for Linac Simulation and Optimization',
    long_description='LISO is a Python API for linac simulation and'
                     'optimization using one or a combination of'
                     'different beam dynamics codes.',
    license='GNU',

    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: GNU',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Software Development',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 ],

    install_requires=REQUIREMENTS
)
