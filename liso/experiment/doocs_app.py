from __future__ import annotations

import argparse

from .doocs_channels import AnyDoocsChannel
from .doocs_interface import DoocsInterface


def monitor():
    parser = argparse.ArgumentParser(prog="liso-doocs-monitor")
    parser.add_argument("channels", metavar="channels", type=str, nargs="?",
                        help="DOOCS channel addresses separated by comma.")
    parser.add_argument("--file", "-f", type=str,
                        help="Read DOOCS channel addresses from the given "
                             "file.")
    parser.add_argument("--correlate", action="store_true",
                        help="Correlate all channel data by macropulse ID.")

    args = parser.parse_args()

    channels = []
    if args.channels is not None:
        channels.extend(args.channels.split(","))
    if args.file is not None:
        channels.extend(open(args.file, 'r').readlines())
    if not channels:
        raise ValueError("No DOOCS channel specified!")

    interface = DoocsInterface("DOOCS")
    for ch in channels:
        interface.add_diagnostic_channel(AnyDoocsChannel, ch)

    interface.monitor(correlate=args.correlate, validate=False)
