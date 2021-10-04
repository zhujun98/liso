import argparse

from liso import FLASHInterface, MachineScan
from liso import doocs_channels as dc


parser = argparse.ArgumentParser(description="FLASH interface")
parser.add_argument('--tasks', type=int, default=16,
                    help='Number of parallel tasks')
parser.add_argument('--scan', type=int, default=0,
                    help='Number of pulses (scan points)')
parser.add_argument('--monitor', action='store_true',
                    help='True for only printing the data on the screen '
                         '(data will not be correlated by macropulse ID')
args = parser.parse_args()

m = FLASHInterface()

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

m.add_diagnostic_channel("FLASH.SDIAG/BAM/1UBC2/LOW_CHARGE_ARRIVAL_TIME", dc.ARRAY,
                         shape=(7222, 2), dtype='float32')
m.add_diagnostic_channel("FLASH.SDIAG/BAM/1SFELC/LOW_CHARGE_ARRIVAL_TIME", dc.ARRAY,
                         shape=(7222, 2), dtype='float32')

m.add_diagnostic_channel("FLASH.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRISP4-141/INTENSITY.TD", dc.ARRAY,
                         shape=(240, 2), dtype='float32')
m.add_diagnostic_channel("FLASH.SDIAG/THZ_SPECTROMETER.FORMFACTOR/CRISP4-141/FORMFACTOR.XY", dc.ARRAY,
                         shape=(240, 2), dtype='float32')

m.add_diagnostic_channel("TTF2.DIAG/CAM.LOLA/6SDUMP/IMAGE_EXT_ZMQ", dc.ARRAY,
                         shape=(1024, 1360), dtype='uint16')

if args.scan > 0:
    sc = MachineScan(m, read_delay=0.5, n_reads=2)

    sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC1/PHASE.SAMPLE', lb=4.50, ub=4.50)
    sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC39/PHASE.SAMPLE', lb=-11.36, ub=-11.36)
    sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC23/PHASE.SAMPLE', lb=21.06, ub=21.06)
    sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC45/PHASE.SAMPLE', lb=-0.05, ub=0.05)
    sc.add_param('FLASH.RF/LLRF.CONTROLLER/VS.ACC67/PHASE.SAMPLE', lb=-0.05, ub=0.05)

    sc.scan(args.scan, n_tasks=args.tasks)

else:
    if args.monitor:
        m.monitor()
    else:
        m.acquire()
