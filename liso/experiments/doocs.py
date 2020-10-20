"""
Distributed under the terms of the GNU General Public License v3.0.

The full license is in the file LICENSE, distributed with this software.

Copyright (C) Jun Zhu. All rights reserved.
"""


class Channel:
    def __init__(self, facility, device, location, property):
        """Initialization.

        :param str facility: facility of the doccs channel.
        :param str device: device of the doccs channel.
        :param str location: location of the doccs channel.
        :param str property: property of the doccs channel.
        """
        if not facility or not device or not location or not property:
            raise ValueError(
                "facility/device/location/property cannot be empty")

        self.facility = facility
        self.device = device
        self.location = location
        self.property = property

        self._address = f"{facility}/{device}/{location}/{property}"

    @property
    def address(self):
        return self._address

    @classmethod
    def from_address(cls, address):
        return cls(*address.split('/'))
