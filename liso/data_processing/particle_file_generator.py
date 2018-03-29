#!/usr/bin/python
"""
Author: Jun Zhu
"""
import csv


class ParticleFileGenerator(object):
    """Class for generating particle file for different codes."""
    def __init__(self, data, fpath):
        """Initialization.

        :param data: Pandas.DataFrame
            Particle data. See data_processing/phasespace_parser for details
            of the data columns.
        :param fpath: string
            Path of the output particle file.
        """
        self._data = data
        self._fpath = fpath

    def to_astra_pfile(self, charge):
        """Dump the data to an Astra particle file."""
        pass

    def to_impactt_pfile(self, header=True):
        """Dump the data to an ImpactT particle file.

        :param header: Bool
            True for an input file for Impact-T simulation and False for a
            general particle file.
        """
        with open(self._fpath, 'w') as fp:
            if header is True:
                fp.write(str(self._data.shape[0]) + '\n')

            self._data.to_csv(fp,
                              header=None,
                              index=False,
                              sep=' ',
                              quoting=csv.QUOTE_NONE,
                              escapechar=' ',
                              float_format="%.12E",
                              columns=['x', 'px', 'y', 'py', 'z', 'pz'])

