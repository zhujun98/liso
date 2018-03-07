from liso import AstraPhaseSpacePlot
from liso import ImpacttPhaseSpacePlot


if __name__ == "__main__":
    psplot1 = ImpacttPhaseSpacePlot('impactt_basic/fort.107', 3.1e-12)
    psplot1.scatter('x', 'y')
    psplot1.scatter('t', 'x', ms=5, alpha=0.5)
    psplot1.scatter('x', 'xp', ms=5)
    psplot1.save_scatter('t', 'p', ms=5, filename='fort107_t-p.png')

    psplot2 = AstraPhaseSpacePlot('astra_basic/injector.0400.001')
    psplot2.cloud('x', 'y')
    psplot2.cloud('y', 'yp', ms=5)
    psplot2.cloud('t', 'p', ms=5)
    psplot2.save_cloud('x', 'xp', ms=5, x_unit='um')
