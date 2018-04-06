from .linac_optimization import Optimization, LinacOptimization


__all__ = [
    'Optimization',
    'LinacOptimization',
]

try:
    from .pyopt_linac_optimization import PyoptLinacOptimization
    __all__.append('PyoptLinacOptimization')
except ModuleNotFoundError:
    pass
