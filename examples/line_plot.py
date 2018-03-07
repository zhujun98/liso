from liso import AstraLinePlot
from liso import ImpacttLinePlot


if __name__ == "__main__":
    l1 = ImpacttLinePlot('impactt_basic/fort')
    l1.plot('Sx', y_unit='mm')
    l1.plot('betax', 'betay')
    l1.save_plot('emitx', 'emity', y_unit='um')

    l2 = AstraLinePlot('astra_basic/injector')
    l2.plot('Sx', y_unit='mm')
    l2.plot('emitx', 'emity', y_unit='um')
    l2.save_plot('Sx', 'Sy', y_unit='mm')
