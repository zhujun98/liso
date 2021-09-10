"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from pydantic import ValidationError

from ..exceptions import LisoRuntimeError
from ..utils import profiler
from .doocs import DoocsReader, DoocsWriter, _machine_event_loop


class MachineInterface:
    # TODO: Improve when there are more than one machine types.
    def __init__(self) -> None:
        pass


class DoocsInterface(MachineInterface):
    """Base class for machine interface using DOOCS control system."""

    _facility_name = None

    def __init__(self):
        super().__init__()
        self._controls = OrderedDict()
        self._diagnostics = OrderedDict()

        self._reader = DoocsReader()
        self._writer = DoocsWriter()

    @property
    def channels(self):
        """Return a list of all DOOCS channels."""
        return list(self._controls) + list(self._diagnostics)

    @property
    def controls(self):
        """Return a list of DOOCS channels for control data."""
        return list(self._controls)

    @property
    def diagnostics(self):
        """Return a list of DOOCS channels for diagnostic data."""
        return list(self._diagnostics)

    @property
    def schema(self):
        """Return the schema of all DOOCS channels."""
        return ({k: v.value_schema() for k, v in self._controls.items()},
                {k: v.value_schema() for k, v in self._diagnostics.items()})

    def _check_address(self, address):
        if address in self._controls or address in self._diagnostics:
            raise ValueError(f"{address} already exists!")

        if not address.startswith(self._facility_name):
            raise ValueError(f"{address} must start with {self._facility_name}")

    def add_control_channel(self, kls, address, *, no_event=False, **kwargs):
        """Add a DOOCS channel for control data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        :param bool no_event: True for a non-event-based channel.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_control_channel(
                dc.FLOAT32, 'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
        """
        self._check_address(address)
        self._controls[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address, no_event)

    def add_diagnostic_channel(self, kls, address, *, no_event=False, **kwargs):
        """Add a DOOCS channel to diagnostic data.

        :param DoocsChannel kls: a concrete DoocsChannel class.
        :param str address: DOOCS address.
        :param bool no_event: True for a non-event-based channel.
        **kwargs: keyword arguments which will be passed to the constructor
            of kls after address.

        Examples:
            from liso import doocs_channels as dc
            from liso import EuXFELInterface

            m = EuXFELInterface()
            m.add_diagnostic_channel(
                dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                shape=(1750, 2330), dtype='uint16')
        """
        self._check_address(address)
        self._diagnostics[address] = kls(address=address, **kwargs)
        self._reader.add_channel(address, no_event)

    def _compile(self, mapping):
        writein = dict()
        readout = dict()
        if mapping is not None:
            for address, item in mapping.items():
                writein[address] = item['value']
                if item.get('readout', None) is not None:
                    readout_address = item['readout']
                    if readout_address not in self._controls \
                            and readout_address not in self._diagnostics:
                        raise ValueError(f"Channel {readout_address} has not "
                                         f"been registered!")

                    readout[readout_address] = (item['value'], item['tol'])

        return writein, readout

    def _update_channels(self, correlated):
        control_data = dict()
        for address, ch in self._controls.items():
            ch.value = correlated[address]  # validate
            control_data[address] = ch.value

        diagnostic_data = dict()
        for address, ch in self._diagnostics.items():
            ch.value = correlated[address]  # validate
            diagnostic_data[address] = ch.value

        return control_data, diagnostic_data

    async def _write_read_once(self, mapping, *, executor, timeout):
        writein, readout = self._compile(mapping)
        await self._writer.write_channels(executor, writein)
        return await self._reader.correlate(executor, readout, timeout=timeout)

    @profiler("machine write and read")
    def write_and_read(self, *, mapping=None, executor=None, timeout=None):
        """Write and read the machine once.

        :param dict mapping: a dictionary with keys being the DOOCS channel
            addresses and values being the numbers to be written into the
            corresponding address.
        :param ThreadPoolExecutor executor: a ThreadPoolExecutor object.
        :param float/None timeout: timeout when correlating data by macropulse
            ID, in seconds. If None, it is set to the default value 2.0.

        :raises:
            LisoRuntimeError: if validation fails or it is unable to
                correlate data.
        """
        if executor is None:
            executor = ThreadPoolExecutor()
        if timeout is None:
            timeout = 2.0

        pid, correlated = _machine_event_loop.run_until_complete(
            self._write_read_once(mapping, executor=executor, timeout=timeout))

        try:
            control_data, diagnostic_data = self._update_channels(correlated)
        except ValidationError as e:
            raise LisoRuntimeError(repr(e))

        return pid, control_data, diagnostic_data

    def take_snapshot(self, channels):
        if not channels:
            return

        return {address: data['data']
                for address, data in _machine_event_loop.run_until_complete(
                self._reader.read_channels(channels)).items()}


class EuXFELInterface(DoocsInterface):
    _facility_name = 'XFEL'


class FLASHInterface(DoocsInterface):
    _facility_name = 'FLASH'
