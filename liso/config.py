"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
import configparser
import os
import os.path as osp


_root_dir = osp.join(osp.expanduser("~"), ".liso")
_config_file = osp.join(_root_dir, "config.ini")

config = configparser.ConfigParser()

if not osp.isfile(_config_file):

    config['DEFAULT'] = {
        'LOG_FILE': "liso.log",
        'OPT_LOG_FILE': "liso_opt.log",
    }

    config['EXECUTABLE'] = {
        'ASTRA': "astra",
        'IMPACTT': "ImpactTv1.7linux",
        'ELEGANT': 'elegant',
    }
    config['EXECUTABLE_PARA'] = {
        'ASTRA': "astra_r62_Linux_x86_64_OpenMPI_1.6.1",
        'IMPACTT': "ImpactTv1.7linuxPara",
        'ELEGANT': "",
    }

    try:
        os.mkdir(_root_dir)
    except FileExistsError:
        pass

    with open(_config_file, "w") as fp:
        config.write(fp)
else:
    config.read(_config_file)
