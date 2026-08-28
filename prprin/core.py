from pandas import DataFrame


class PrPrInCore:
    """State container for a protein-protein interaction analysis."""

    def __init__(self, proteins=None, network=None, specie_id=9606):

        # raw data
        self.proteins = DataFrame({"stringId":[], "symbol":[]}) if proteins is None else proteins             # proteins in the object (symbol and string id)
        self.network = network                                                                                # connection data between the proteins
        self.specie_id = specie_id                                                                            # ID of specie of reference (NCBI taxonomy identifier). Default is Human

        # processed data
        self.graph = None                                                                                     # graph of the protein-protein interactions
        self.node_data = None                                                                                 # per-node table (similar to Seurat's meta.data)
        self.used_weight_name = None                                                                          # the name of the score used as the weight for the edges in the graph
        self.hubs = None                                                                                      # hub proteins
        self.enrichment = None                                                                                # enrichment analysis done on the proteins

    def __repr__(self):
        n = self.graph.number_of_nodes() if self.graph else 0
        e = self.graph.number_of_edges() if self.graph else 0
        slots = [k for k in ("network", "node_data", "hubs", "enrichment", "graph")
                 if getattr(self, k) is not None]
        return (f"PrPrInObject: {len(self.proteins)} proteins in the object\n"
                f"NCBI taxonomy identifier of specie of reference: {self.specie_id}\n"
                f"graph comosed by {n} nodes and {e} edges\n"
                f"data present in the object: {', '.join(slots) or 'none'}\n")


if __name__ == "__main__":
    print(PrPrInCore())