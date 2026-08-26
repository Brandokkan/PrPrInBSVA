from .core import PrPrInCore
from .retrive import RetrivePPI
# from .mapping import MappingMixin
from .graph import GraphPPI
from .metrics import MetricsPPI
from .hubs import HubsPPI
# from .enrich import EnrichMixin
# from .viz import VizMixin


class PrPrInObject(PrPrInCore, RetrivePPI, GraphPPI, MetricsPPI,
                   HubsPPI):
    """Container for a protein-protein interaction analysis: raw data, derived data, and the methods."""


__all__ = ["PrPrInObject"]