from liso import create_line_plot


if __name__ == "__main__":
    l2 = create_line_plot('astra', 'astra_basic/injector')
    l2.plot('Sx', y_unit='Mm')
    l2.plot('emitx', 'emity', y_unit='nm')
    l2.save_plot('Sx', 'Sy', y_unit='mM')

    l1 = create_line_plot('impactt', 'impactt_basic/fort')
    l1.plot('Sx', y_unit='um')
    l1.plot('betax', 'betay')
    l1.save_plot('emitx', 'emity', y_unit='Nm')
