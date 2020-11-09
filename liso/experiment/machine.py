"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections import OrderedDict
from concurrent.futures import as_completed, ThreadPoolExecutor
import time

try:
    from pydoocs import read as pydoocs_read
    from pydoocs import write as pydoocs_write
    from pydoocs import DoocsException, PyDoocsException
except ModuleNotFoundError:
    __pydoocs_error_msg = "pydoocs is required to communicate with a real " \
                          "machine using DOOCS control system!"
    def pydoocs_read(*args):
        raise ModuleNotFoundError(__pydoocs_error_msg)
    def pydoocs_write(*args):
        raise ModuleNotFoundError(__pydoocs_error_msg)
    class DoocsException(Exception):
        pass
    class PyDoocsException(Exception):
        pass

from ..logging import logger
from ..utils import profiler


class _DoocsWriter:
    def __init__(self):
        super().__init__()

    def update(self, executor, mapping=None):
        if mapping is None:
            return

        future_result = {executor.submit(pydoocs_write, ch, v): ch
                         for ch, v in mapping.items()}

        for future in as_completed(future_result):
            channel = future_result[future]
            try:
                future.result()
            except ModuleNotFoundError as e:
                logger.error(repr(e))
                raise
            except [DoocsException, PyDoocsException] as e:
                logger.warning(f"Failed to write to {channel}: {repr(e)}")
            except Exception as e:
                logger.error(f"Unexpected exception when writing to"
                             f" {channel}: {repr(e)}")
                # TODO: here should raise


class _DoocsReader:
    def __init__(self):
        super().__init__()

        self._channels = set()

    def update(self, executor):
        future_result = {executor.submit(pydoocs_read, ch): ch
                         for ch in self._channels}

        ret = dict()
        for future in as_completed(future_result):
            channel = future_result[future]
            try:
                ret[channel] = future.result()
            except ModuleNotFoundError as e:
                logger.error(repr(e))
                raise
            except [DoocsException, PyDoocsException] as e:
                logger.warning(f"Failed to read {channel}: {repr(e)}")
            except Exception as e:
                logger.error(f"Unexpected exception when writing to "
                             f"{channel}: {repr(e)}")
                # TODO: here should raise

        pulse_id = 0
        return pulse_id, ret

    def add_channel(self, ch):
        self._channels.add(ch)


class _DoocsMachine:
    """Base class for machine interface using DOOCS control system."""
    def __init__(self, *, delay=0.001):
        """Initialization.

        :param float delay: delay in seconds after writing new values into
            DOOCS server.
        """
        self._delay = delay

        self._controls = OrderedDict()
        self._instruments = OrderedDict()

        self._reader = _DoocsReader()
        self._writer = _DoocsWriter()

    @property
    def channels(self):
        """Return a list of all DOOCS channels."""
        return list(self._controls) + list(self._instruments)

    @property
    def controls(self):
        """Return a list of DOOCS channels for control data."""
        return list(self._controls)

    @property
    def instruments(self):
        """Return a list of DOOCS channels for instrument data."""
        return list(self._instruments)

    @property
    def schema(self):
        """Return the schema of all DOOCS channels."""
        return ({k: v.value_schema() for k, v in self._controls.items()},
                {k: v.value_schema() for k, v in self._instruments.items()})

    def add_control_channel(self, kls, address, **kwargs):
        """Add a DOOCS channel for control data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_control_channel(
                dc.FLOAT32, 'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
        """
        if address in self._controls or address in self._instruments:
            raise ValueError(f"{address} already exists!")
        self._controls[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address)

    def add_instrument_channel(self, kls, address, **kwargs):
        """Add a DOOCS channel to instrument data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_instrument_channel(
                dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                shape=(1750, 2330), dtype='uint16')
        """
        if address in self._controls or address in self._instruments:
            raise ValueError(f"{address} already exists!")
        self._instruments[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address)

    @profiler("run machine once")
    def run(self, *, executor=None, threads=2, mapping=None):
        """Run the machine once.

        :param ThreadPoolExecutor executor: a ThreadPoolExecutor object.
        :param int threads: number of threads used in constructing a
            ThreadPoolExecutor if the executor is None. Ignored otherwise.
        :param dict mapping: a dictionary with keys being the DOOCS channel
            addresses and values being the numbers to be written into the
            corresponding address.
        """
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=threads)

        self._writer.update(executor, mapping=mapping)

        time.sleep(self._delay)

        pulse_id, updated = self._reader.update(executor)

        # TODO: update and validate self._controls and self._instruments

        return (pulse_id,
                {k: self._controls[k].value for k in self._controls},
                {k: self._instruments[k].value for k in self._instruments})


class EuXFELInterface(_DoocsMachine):
    ...
