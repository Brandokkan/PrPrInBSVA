import networkx as nx
import pandas as pd
import numpy as np

# UNWEIGHTED_METRICS_NAMES = ("degree_centrality", "betweenness_centrality", "clustering")
STRINGID_COL_NAME = "stringId"
SCORE_NAMES = ("score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore")
ZERO_DIVISION_DISTANCE = 1000000

def unweighted_metrics(graph, index_df, verbose=False):

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
    final_df = pd.merge(index_df, metrics_df, left_on=STRINGID_COL_NAME, right_index=True)

    if verbose:
        print(final_df, sep="\n")
    return final_df
    

def weighted_metrics(graph, index_df, w_type="score", verbose=False):

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
            nx.betweenness_centrality(graph, weight=lambda u, v, d: 1 / d[w_type] if d[w_type] != 0 else ZERO_DIVISION_DISTANCE),
            name=f"weighted betweenness centrality (1 / {w_type})"
        )
    w_betweenes_centrality_log = pd.Series(
            nx.betweenness_centrality(graph, weight=lambda u, v, d: -np.log(d[w_type]) if d[w_type] != 0 else ZERO_DIVISION_DISTANCE),
            name=f"weighted betweenness centrality (-log({w_type}))"
        )
    w_clustering_centrality = pd.Series(nx.clustering(graph, weight=w_type), name=f"weighted clustering coefficient ({w_type})")

    # final DataFrame creation
    metrics_df = pd.DataFrame([w_degree, w_degree_normalized, w_betweenes_centrality_sub, w_betweenes_centrality_rec, 
                               w_betweenes_centrality_log, w_clustering_centrality]).T
    final_df = pd.merge(index_df, metrics_df, left_on=STRINGID_COL_NAME, right_index=True)

    if verbose:
        print(final_df, sep="\n")
    return final_df


class MetricsPPI:
    def calculate_metrics(self):

        # check if a graph exists
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("there is no graph saved in the 'graph' container")

        # calculate key graph parameters

        # calculate unweighted parameters


if __name__ == "__main__":
    graph = nx.Graph()
    graph.add_edges_from((("a","b", {"score":0.8}), ("c","d", {"score":0.3}), ("d","b", {"score":0}), ("d","a", {"score":0.9}),
                          ("c","e", {"score":1}), ("c","f", {"score":0.95})
                          ))
    inx_df = pd.DataFrame({STRINGID_COL_NAME:["a","b","c","d","e","f"], "symbol":["azz","bee","caz","dam","ez","fuc"]})
    print(weighted_metrics(graph, inx_df))
    unweighted_metrics(graph, inx_df, verbose=True)