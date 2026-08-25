import networkx as nx
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability

# types of scores present for each PPI pair in STRING
EDGE_SCORES = ("score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore")

class GraphPPI:
    def create_graph(self):
        """Create a graph from the PPI data present in the 'network' container

        This method creates a networkx graph from the protein-protein interaction data
        present in the 'network' container. If no data is present in 'network', it raises an error.
        """
        # checking if PPI data is present
        if self.network is None:
            raise ValueError("there is no protein-protein interaction data saved in the 'network' container")

        # graph creation
        self.graph = nx.from_pandas_edgelist(
            self.network,
            "stringId_A",
            "stringId_B",
            edge_attr= EDGE_SCORES
        )

        return self


if __name__ == "__main__":
    # ex_ppi_df = read_csv(DATA_DIR / "ex_ppi_df.csv")
    # ppi_edges_to_net(ex_ppi_df, True)
    pass