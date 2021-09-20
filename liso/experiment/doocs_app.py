import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from ..logging import logger
from .doocs_channels import AnyDoocsChannel
from .doocs_interface import DoocsInterface


def _parse_channel_file(filepath):
    fast, slow = [], []
    ret = fast
    with open(filepath, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("---"):
                ret = slow
                continue
            ret.append(line)
    return fast, slow


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

    fast_channels, slow_channels = [], []
    if args.channels is not None:
        fast_channels.extend(args.channels.split(","))
    if args.file is not None:
        fast, slow_channels = _parse_channel_file(args.file)
        fast_channels.extend(fast)
    if not fast_channels and not slow_channels:
        raise ValueError("No DOOCS channel specified!")

    interface = DoocsInterface("DOOCS")
    for ch in fast_channels:
        interface.add_diagnostic_channel(AnyDoocsChannel, ch)
    for ch in slow_channels:
        interface.add_diagnostic_channel(AnyDoocsChannel, ch, non_event=True)

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    correlate = args.correlate
    try:
        while True:
            pid, _, diagnostics = interface.read(
                loop, executor, correlate=correlate)
            print("-" * 80)
            print("Macropulse ID:", pid)
            print()
            print("\n".join([f"- {k}: {v}" for k, v in diagnostics.items()]))
            print("-" * 80)

            if correlate:
                time.sleep(0.001)
            else:
                time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping monitoring ...")
