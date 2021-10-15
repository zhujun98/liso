import pydoocs, pydaq

from contextlib import ContextDecorator
import time


t_start = '2021-10-06T00:30:00'
t_end = '2021-10-06T00:35:00'
exp = 'linac'

channels = [
    "FLASH.DIAG/TOROID/3GUN",
    "FLASH.DIAG/TOROID/1ORS"
]


class ConnectDoocsDaq(ContextDecorator):
    def __init__(self, *, t_start, t_end, exp, channels):
        self._t_start = t_start
        self._t_end = t_end
        self._exp = exp
        self._channels = channels

    def __enter__(self):
        pydaq.connect(start=self._t_start,
                      stop=self._t_end,
                      exp=self._exp,
                      chans=self._channels)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pydaq.disconnect()
        return False


stop = False
stats_list = []

for _ in range(10):
    channels = pydaq.getdata()
    if channels == []:
        time.sleep(0.001)
        continue

    if channels == None:
        break

    for chan in channels:
        subchan = len(chan)
        daqname = chan[0]['miscellaneous']['daqname']
        found = False
        for stats in stats_list:
            if stats['daqname'] == daqname:
                stats['events'] += 1
                chtotal = stats['events']
                found = True
                break
        if not found:
            entry = {}
            entry['daqname'] = daqname
            entry['events'] = 1
            chtotal = entry['events']
            stats_list.append(entry)
