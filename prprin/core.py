


class PrPrInCore:
    """State container for a protein-protein interaction analysis."""

    def __init__(self, symbol=None, string_id=None, network=None):

        # raw data
        self.symbol = [] if symbol is None else symbol                   # gene symbol
        self.string_id = [] if string_id is None else string_id          # protein stringId
        # TODO: add input check to confirm equivalence between symbol and string_id
        self.network = network                                           # connection data between the proteins

        # processed data
        self.graph = None                                                # graph of the protein-protein interactions
        self.node_data = None                                            # per-node table (Seurat's meta.data)
        self.hubs = None                                                 # hub proteins
        self.enrichment = None                                           # enrichment analysis done on the proteins

    def __repr__(self):
        n = self.graph.number_of_nodes() if self.graph else 0
        e = self.graph.number_of_edges() if self.graph else 0
        slots = [k for k in ("network", "node_data", "hubs", "enrichment")
                 if getattr(self, k) is not None]
        return (f"PrPrInObject: {len(self.string_id)} string ids, "
                f"{n} nodes, {e} edges\n"
                f"data: {', '.join(slots) or 'none'}\n")


if __name__ == "__main__":
    print(PrPrInCore())