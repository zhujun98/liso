from liso import open_sim


n = 12  # 1200

sim = open_sim('./r0001')

sim.info()

assert sorted(sim.sim_ids) == list(range(1, n + 1))

controls = {
    'gun/tws_phase',
    'gun/tws_gradient',
    'gun/gun_phase',
    'gun/gun_gradient',
}
phasespaces = {
    'gun/out'
}
assert sim.control_channels == controls
assert sim.phasespace_channels == phasespaces

_, data = sim.from_id(n)
assert len(data) == 5
_, data = sim.from_index(0)
assert len(data) == 5

assert sim.channel("gun/out", 'x').numpy().shape == (n, 1, 2000)
