import argparse

from liso import EuXFELInterface, MachineScan
from liso import doocs_channels as dc

parser = argparse.ArgumentParser(description="EuXFEL interface")
parser.add_argument('pulses', type=int, default=100,
                    help='Number of pulses (scan points)')
parser.add_argument('--tasks', type=int, default=16,
                    help='Number of parallel tasks')
args = parser.parse_args()


m = EuXFELInterface()

m.add_control_channel(dc.FLOAT,
                      'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
                      'XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE')
m.add_control_channel(dc.FLOAT,
                      'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE',
                      'XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE')
m.add_control_channel(dc.FLOAT,
                      'XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE',
                      'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE')

# non-event based data
m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP')
m.add_control_channel(dc.FLOAT, 'XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP')

m.add_diagnostic_channel(
    dc.FLOAT, 'XFEL.SDIAG/BAM/47.I1/LOW_CHARGE.RESOLUTION', no_event=True)
m.add_diagnostic_channel(
    dc.INT, 'XFEL.SDIAG/BAM/47.I1/SINGLEBUNCH_NUMBER_FOR_ARRIVAL_TIME_HISTORY.1',
    no_event=True)
m.add_diagnostic_channel(
    dc.FLOAT, 'XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.1')

m.add_diagnostic_channel(dc.FLOAT, 'XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL')

m.add_diagnostic_channel(dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                         shape=(1750, 2330), dtype='uint16', no_event=True)

sc = MachineScan(m)

# Uncomment if you have write authority. Be careful!!!
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE', lb=-3, ub=3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE', lb=182, ub=186)

sc.scan(args.pulses, tasks=args.tasks)
