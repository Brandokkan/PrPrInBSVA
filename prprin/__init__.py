from .core import PrPrInCore
from .retrive import RetrivePPI
# from .mapping import MappingMixin
from .graph import GraphPPI
# from .metrics import MetricsMixin
# from .hubs import HubsMixin
# from .enrich import EnrichMixin
# from .viz import VizMixin


class PrPrInObject(PrPrInCore, RetrivePPI, GraphPPI):
    """Container for a protein-protein interaction analysis: raw data, derived data, and the methods."""


__all__ = ["PrPrInObject"]