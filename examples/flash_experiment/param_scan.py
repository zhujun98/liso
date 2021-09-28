import argparse

from liso import FLASHInterface, MachineScan
from liso import doocs_channels as dc


parser = argparse.ArgumentParser(description="FLASH interface")
parser.add_argument('pulses', type=int, default=100,
                    help='Number of pulses (scan points)')
parser.add_argument('--tasks', type=int, default=16,
                    help='Number of parallel tasks')
args = parser.parse_args()

m = FLASHInterface()

m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.GUN/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.GUN/SP.PHASE')
m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.ACC1/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.ACC1/SP.PHASE')
m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.ACC39/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.ACC39/SP.PHASE')
m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.ACC23/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.ACC23/SP.PHASE')
m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.ACC45/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.ACC45/SP.PHASE')
m.add_control_channel('FLASH.RF/LLRF.CONTROLLER/VS.ACC67/PHASE.SAMPLE', dc.FLOAT,
                      write_address='FLASH.RF/LLRF.CONTROLLER/CTRL.ACC67/SP.PHASE')

# m.add_diagnostic_channel("FLASH.DIAG/CAMERA/OTR9FL2XTDS/IMAGE_EXT_ZMQ", dc.IMAGE,
#                          shape=(1776, 2360), dtype='uint16', no_event=True)

sc = MachineScan(m)

# Uncomment the following lines if you have write authority. Be careful!!!
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.GUN/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC1/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC39/PHASE.SAMPLE', lb=182, ub=186)
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC23/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC45/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC67/PHASE.SAMPLE', lb=182, ub=186)

sc.scan(args.pulses, n_tasks=args.tasks)
