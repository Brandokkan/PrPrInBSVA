import networkx as nx
import pandas as pd
import numpy as np


STRINGID_INDEX_NAME = "stringId"
SCORE_NAMES = ("score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore")


def calculate_unweighted_metrics(graph, index_df, verbose=False):
    """Calculate key un-weighted metrics of a graph nodes

    This function calculates key un-weighted metrics of the nodes of the 
    given graph and outputs a pandas DataFrame associating each node to 
    the calculated metrics.

    Parameters
    -------
    graph: networkx.Graph
        The graph containing the nodes of which the metrics will be calculated

    index_df: pandas.DataFrame
        The DataFrame associating each symbol to its stringId, indexed by
        stringId. The one stored in the 'proteins' container.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the un-weighted metrics associeted to each node
    """
    # input check
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    if not isinstance(index_df, pd.DataFrame):
        raise TypeError("index_df must be a Pandas DataFrame")

    #properties calculation
    degree = pd.Series(dict(nx.degree(graph)), name="degree")
    degree_centrality = pd.Series(nx.degree_centrality(graph), name="degree centrality")
    betweenes_centrality = pd.Series(nx.betweenness_centrality(graph), name="betweenness centrality")
    clustering_centrality = pd.Series(nx.clustering(graph), name="clustering coefficient")

    # final DataFrame creation
    metrics_df = pd.DataFrame([degree, degree_centrality, betweenes_centrality, clustering_centrality]).T
    final_df = pd.merge(index_df, metrics_df, left_index=True, right_index=True)

    if verbose:
        print(final_df, sep="\n")
    return final_df
    

def calculate_weighted_metrics(graph, index_df, w_type="score", zero_distance=1000000, verbose=False):
    """Calculate key weighted metrics of a graph nodes

    This function calculates key weighted metrics of the nodes of the 
    given graph and outputs a pandas DataFrame associating each node to 
    the calculated metrics.

    The attribute present in the graph used as the weight is indicated between
    parenteses

    Parameters
    -------
    graph: networkx.Graph
        The graph containing the nodes of which the metrics will be calculated

    index_df: pandas.DataFrame
        The DataFrame associating each symbol to its stringId, indexed by
        stringId. The one stored in the 'proteins' container.

    w_type: str
        Must be one of "score", "nscore", "fscore", "pscore", "ascore", "escore",
        "dscore", "tscore".

        It tells the function which attribute should it use as the weight

    zero_distance: numeric
        When calculating the weighted betweenness centrality, the weights are
        treated as distances (not similarities, like the scores of STRING). So, 
        they must be converted.

        Some of the conversion functions give an error when zero is inputed. In
        these cases, zero_distance is the number associated to the zero value.

        It is advised to input a big number (default one million).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the weighted metrics associeted to each node
    """
    # input check
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    if not isinstance(index_df, pd.DataFrame):
        raise TypeError("index_df must be a Pandas DataFrame")
    if not isinstance(w_type, str):
        raise TypeError("w_type must be a string")
    if w_type not in SCORE_NAMES:
        raise ValueError(f"w_type must be one of {SCORE_NAMES}")

    #properties calculation
    w_degree = pd.Series(dict(nx.degree(graph, weight=w_type)), name=f"weighted degree ({w_type})")
    w_degree_normalized = w_degree / w_degree.sum()
    w_degree_normalized.name = f"weighted degree normalized ({w_type})"
    w_betweenes_centrality_sub = pd.Series( # betweenness_centrality treadts weights as distances instead of similarities. They are changed accordinglly
        nx.betweenness_centrality(graph, weight=lambda u, v, d: 1 - d[w_type]),
        name=f"weighted betweenness centrality (1 - {w_type})"
    )
    w_betweenes_centrality_rec = pd.Series(
            nx.betweenness_centrality(graph, weight=lambda u, v, d: 1 / d[w_type] if d[w_type] != 0 else zero_distance),
            name=f"weighted betweenness centrality (1 / {w_type})"
        )
    w_betweenes_centrality_log = pd.Series(
            nx.betweenness_centrality(graph, weight=lambda u, v, d: -np.log(d[w_type]) if d[w_type] != 0 else zero_distance),
            name=f"weighted betweenness centrality (-log({w_type}))"
        )
    w_clustering_centrality = pd.Series(nx.clustering(graph, weight=w_type), name=f"weighted clustering coefficient ({w_type})")

    # final DataFrame creation
    metrics_df = pd.DataFrame([w_degree, w_degree_normalized, w_betweenes_centrality_sub, w_betweenes_centrality_rec, 
                               w_betweenes_centrality_log, w_clustering_centrality]).T
    final_df = pd.merge(index_df, metrics_df, left_index=True, right_index=True)

    if verbose:
        print(final_df, sep="\n")
    return final_df


class MetricsPPI:
    def calculate_metrics(self, used_weight="score", zero_distance=1000000):
        """Calculate key metrics of a graph nodes

        This method calculates key metrics of the nodes of the 
        graph stored in the 'graph' container and saves them in the 
        'node_data' container.
        
        The attribute present in the graph used as the weight is indicated between
        parenteses in the column names.
        
        Parameters
        -------
        used_weight: str
            Must be one of "score", "nscore", "fscore", "pscore", "ascore", "escore",
            "dscore", "tscore".

            It tells the function which attribute should it use as the weight

        zero_distance: numeric
            When calculating the weighted betweenness centrality, the weights are
            treated as distances (not similarities, like the scores of STRING). So, 
            they must be converted.

            Some of the conversion functions give an error when zero is inputed. In
            these cases, zero_distance is the number associated to the zero value.

            It is advised to input a big number (default one million).
        """
        # check if a graph exists
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("there is no graph saved in the 'graph' container")

        # calculate key graph parameters
        unweighted_metrics = calculate_unweighted_metrics(self.graph, self.proteins)
        weighted_metrics = calculate_weighted_metrics(self.graph, self.proteins, used_weight, zero_distance)
        self.used_weight_name = used_weight

        # create final DataFrame (the symbol is already carried by the un-weighted table)
        self.node_data = pd.merge(unweighted_metrics, weighted_metrics.drop(columns="symbol"),
                                  left_index=True, right_index=True)

        return self

if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from((("a","b", {"score":0.8}), ("c","d", {"score":0.3}), ("d","b", {"score":0}), ("d","a", {"score":0.9}),
                          ("c","e", {"score":1}), ("c","f", {"score":0.95})
                          ))
    inx_df = pd.DataFrame({"symbol":["azz","bee","caz","dam","ez","fuc"]},
                          index=pd.Index(["a","b","c","d","e","f"], name=STRINGID_INDEX_NAME))
    print(calculate_weighted_metrics(graph, inx_df))
    calculate_unweighted_metrics(graph, inx_df, verbose=True)