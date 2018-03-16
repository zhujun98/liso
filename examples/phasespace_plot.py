from liso import PhaseSpacePlot


if __name__ == "__main__":
    psplot2 = PhaseSpacePlot('astra', 'astra_basic/injector.0400.001', filter_size=2)
    psplot2.cloud('x', 'Y')
    psplot2.cloud('y', 'yp', ms=5)
    psplot2.cloud('T', 'delta', x_unit='ps', y1_unit='kA', ms=5)
    psplot2.save_cloud('x', 'xp', ms=5, x_unit='uM', y_unit='urAd')

    psplot1 = PhaseSpacePlot('impactt', 'impactt_basic/fort.107', charge=3.1e-12)
    psplot1.scatter('x', 'y')
    psplot1.scatter('t', 'x', ms=5, alpha=0.5)
    psplot1.scatter('x', 'xp', x_unit='um', y_unit='urad', ms=5)
    psplot1.save_scatter('T', 'P', ms=5, filename='fort107_t-p.png')
