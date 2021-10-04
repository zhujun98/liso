import argparse

from liso import EuXFELInterface, MachineScan
from liso import doocs_channels as dc


parser = argparse.ArgumentParser(description="XFEL interface")
parser.add_argument('--tasks', type=int, default=16,
                    help='Number of parallel tasks')
parser.add_argument('--scan', type=int, default=0,
                    help='Number of pulses (scan points)')
parser.add_argument('--monitor', action='store_true',
                    help='True for only printing the data on the screen '
                         '(data will not be correlated by macropulse ID')
args = parser.parse_args()

machine = EuXFELInterface()
m = machine

m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', dc.FLOAT,
                      write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')
m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE', dc.FLOAT)
m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', dc.FLOAT,
                      write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE')
m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE', dc.FLOAT)
m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', dc.FLOAT,
                      write_address='XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
m.add_control_channel('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE', dc.FLOAT)

m.add_diagnostic_channel('XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_ARRIVAL_TIME', dc.ARRAY,
                         shape=(7222, 2), dtype='float32')

m.add_diagnostic_channel('XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL', dc.FLOAT)

m.add_diagnostic_channel('XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.ARRAY,
                         shape=(1750, 2330), dtype='uint16')

if args.scan > 0:
    sc = MachineScan(machine)

    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', lb=-3, ub=3)
    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', lb=-3, ub=3)
    sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', lb=182, ub=186)

    sc.scan(args.scan, n_tasks=args.tasks)
else:
    if args.monitor:
        m.monitor()
    else:
        m.acquire()
