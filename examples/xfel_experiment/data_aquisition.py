from liso import EuXFELInterface
from liso import doocs_channels as dc


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

m.add_control_channel(
    'XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)
m.add_control_channel(
    'XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP', dc.DOUBLE, non_event=True)

# m.add_diagnostic_channel(
#     'XFEL.SDIAG/BAM/47.I1/LOW_CHARGE.RESOLUTION', dc.FLOAT, no_event=True)
# m.add_diagnostic_channel(
#     'XFEL.SDIAG/BAM/47.I1/SINGLEBUNCH_NUMBER_FOR_ARRIVAL_TIME_HISTORY.1', dc.INT, no_event=True)
# m.add_diagnostic_channel(
#     'XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.1', dc.FLOAT)

m.add_diagnostic_channel('XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL', dc.FLOAT)

# m.add_diagnostic_channel('XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ', dc.IMAGE,
#                          shape=(1750, 2330), dtype='uint16', no_event=True)


if __name__ == "__main__":
    m.acquire()
