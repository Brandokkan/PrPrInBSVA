from pandas import DataFrame


class PrPrInCore:
    """State container for a protein-protein interaction analysis."""

    def __init__(self, proteins=None, network=None):

        # raw data
        self.proteins = DataFrame({"stringId":[], "symbol":[]}) if proteins is None else proteins             # proteins in the object (symbol and string id)
        self.network = network                                                                                # connection data between the proteins

        # processed data
        self.graph = None                                                                                     # graph of the protein-protein interactions
        self.node_data = None                                                                                 # per-node table (similar to Seurat's meta.data)
        self.hubs = None                                                                                      # hub proteins
        self.enrichment = None                                                                                # enrichment analysis done on the proteins

    def __repr__(self):
        n = self.graph.number_of_nodes() if self.graph else 0
        e = self.graph.number_of_edges() if self.graph else 0
        slots = [k for k in ("network", "node_data", "hubs", "enrichment")
                 if getattr(self, k) is not None]
        return (f"PrPrInObject: {len(self.proteins)} proteins in the object, "
                f"{n} nodes, {e} edges\n"
                f"data: {', '.join(slots) or 'none'}\n")


if __name__ == "__main__":
    print(PrPrInCore())