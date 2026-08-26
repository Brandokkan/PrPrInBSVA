import pandas as pd
import numpy as np


HUB_METHODS = ("topk", "sd", "consensus")


def hubs_by_topk(node_data, metric="degree", k=10, ascending=False, verbose=False):

    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string telling the metric to use")
    if metric not in node_data.columns:
        raise ValueError(f"metric must indicate one of the metric columns present in node_data. The current columns in code_data are:\n{node_data.columns}")
    if not (isinstance(k, int) and not isinstance(k, bool)) or k <= 0:
        raise TypeError("k must be a positive integer value")
    if not isinstance(ascending, bool):
        raise TypeError("ascending must be a boolean value")

    # find hubs
    if ascending:
        top_k = node_data.nsmallest(k, metric, "all")
    else:
        top_k = node_data.nlargest(k, metric, "all")
    top_k_bool = node_data.index.isin(top_k.index)

    if verbose:
        print(top_k_bool)
    return top_k_bool


def hubs_by_sd(node_data, metric="degree", n_sd=2, verbose=False):

    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string telling the metric to use")
    if metric not in node_data.columns:
        raise ValueError(f"metric must indicate one of the metric columns present in node_data. The current columns in code_data are:\n{node_data.columns}")
    if not (isinstance(n_sd, (int, float)) and not isinstance(n_sd, bool)) or n_sd <= 0:
        raise TypeError("n_sd must be a positive integer value")

    # find hubs
    selected_nodes_bool = node_data[metric] > np.mean(node_data[metric]) + n_sd*np.std(node_data[metric])

    if verbose:
        print(selected_nodes_bool)
        
    return selected_nodes_bool


def hubs_by_consensus(node_data, metrics=(...), k=10, how="intersection"): ...


class HubsPPI:
    def find_hubs(self, method="sd", metric="degree", **kwargs):
        """..."""
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")
        ...
        self.node_data[f"is_hub_{method}"] = ...
        self.hubs = self.node_data.loc[self.node_data[f"is_hub_{method}"]]
        return self