from liso import open_run

run = open_run("./scan.hdf5")

run.info()

print(run.get_controls())
