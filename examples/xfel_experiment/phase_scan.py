import argparse

from liso import MachineScan
from data_aquisition import machine


parser = argparse.ArgumentParser(description="EuXFEL interface")
parser.add_argument('pulses', type=int, default=100,
                    help='Number of pulses (scan points)')
parser.add_argument('--tasks', type=int, default=16,
                    help='Number of parallel tasks')
args = parser.parse_args()

sc = MachineScan(machine)

# Uncomment the following lines if you have write authority. Be careful!!!
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', lb=182, ub=186)

sc.scan(args.pulses, n_tasks=args.tasks)
