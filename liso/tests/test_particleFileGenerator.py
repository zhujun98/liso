from liso.data_processing import convert_particle_file


pout = "../examples/astra_basic/injector.0400.001"
pin = "./pdata"

convert_particle_file(pout, pin, code_pout='a', code_pin='t')
