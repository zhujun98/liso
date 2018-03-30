from .linac_optimization import LinacOptimization


__all__ = [
    'LinacOptimization',
]

try:
    from .pyopt_linac_optimization import PyoptLinacOptimization
    __all__.append('PyoptLinacOptimization')
except ModuleNotFoundError:
    pass
