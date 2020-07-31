import os.path as osp
import re
from setuptools import setup, find_packages


def find_version():
    with open(osp.join('liso', '__init__.py')) as fp:
        for line in fp:
            m = re.search(r'^__version__ = "(\d+\.\d+\.\d[a-z]*\d*)"', line, re.M)
            if m is None:
                # could be a hotfix
                m = re.search(r'^__version__ = "(\d.){3}\d"', line, re.M)
            if m is not None:
                return m.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name='liso',
    version=find_version(),
    packages=find_packages(),
    author='Jun Zhu',
    author_email='zhujun981661@gmail.com',
    url='https://github.com/zhujun98/liso',
    download_url='https://github.com/zhujun98/liso',
    description='Python API for Linac Simulation and Optimization',
    long_description='liso is a Python API for linac simulation and'
                     'optimization using one or a combination of'
                     'different beam dynamics codes.',
    license='GNU',

    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: GNU',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    install_requires=[
        'numpy>=1.19',
        'pandas>=1.0',
        'scipy>=1.5',
        'h5py>=2.10',
    ],
    entry_points={
    },
    extras_require={
        'testing': [
            'codecov',
            'pytest',
            'pytest-cov',
        ]
    }
)
