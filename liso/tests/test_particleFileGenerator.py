from liso.data_processing import convert_particle_file


pin = "../examples/astra_basic/injector.0400.001"
pout = "./pdata"

convert_particle_file(pin, pout, code_out='a', code_in='t')
