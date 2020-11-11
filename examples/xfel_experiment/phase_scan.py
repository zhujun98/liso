from liso import EuXFELInterface, MachineScan
from liso import doocs_channels as dc


m = EuXFELInterface()

m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE')
m.add_control_channel(dc.FLOAT, 'XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL')

m.add_instrument_channel(dc.IMAGE, 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                         shape=(1750, 2330), dtype='uint16')

sc = MachineScan(m)

# Uncomment if you have write authority. Be careful!!!
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE', -3, 3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE', -3, 3)
# sc.add_param('XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE', 182, 186)

sc.scan(40, output='scan.hdf5', n_tasks=8)
