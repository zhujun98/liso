from liso import open_run


run = open_run('./r0001')

run.info()

controls = {
    'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE',
    'XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE',
    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE',
    'XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE',
    'XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE',
    'XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE',
    'XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP',
    'XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP'
}
diagnostics = {
    'XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL'
}

assert run.control_channels == controls
assert run.diagnostic_channels == diagnostics

print("First pulse: ", run.pulse_ids[0], "Last pulse: ", run.pulse_ids[-1])

_, data = run.from_id(run.pulse_ids[0])
print(data)
_, data = run.from_index(0)
print(data)
