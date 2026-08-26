import pandas as pd
import numpy as np
from pathlib import Path

HUB_METHODS = ["topk", "sd", "consensus"]
NAME_COLUMNS = ["stringId", "symbol"]
UNWEIGHTED_COLUMNS = ["degree", "degree centrality", "betweenness centrality", "clustering coefficient"]
BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability


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

    # empty warning
    if not any(selected_nodes_bool):
        print("\nWarning: the sd method for finding hub proteins yielded no results\n")

    if verbose:
        print(selected_nodes_bool)
        
    return selected_nodes_bool


def hubs_by_consensus(node_data, metrics="all", k=10, how="intersection", verbose=False):

    # input check
    if not isinstance(node_data, pd.DataFrame):
        raise TypeError("node_data must be a pandas DataFrame. Structured like the one contained in 'node_data'")
    if not isinstance(metrics, (str, list)):
        raise TypeError("metric must be a string or list telling the metric(s) to use")
    if not (isinstance(k, int) and not isinstance(k, bool)) or k <= 0:
        raise TypeError("k must be a positive integer value")
    if not isinstance(how, str):
        raise TypeError("'how' must be a string")
    if how not in ("sum", "intersection"):
        raise ValueError("must be either 'sum' or 'intersection'")

    # convert metrics into the actually used columns for convenience
    if metrics == "all":
        used_cols = [col for col in node_data.columns if col not in NAME_COLUMNS]
    elif metrics == "unweighted":
        used_cols = [col for col in node_data.columns if col in UNWEIGHTED_COLUMNS]
    elif not isinstance(metrics, list):
        used_cols = [col for col in node_data.columns if metrics in col]
    else:
        used_cols = metrics.copy()
    print(f"\nThe selected metrics for the consensus method for finding hub proteins are: {used_cols}\n")

    # find hubs
    rank_df = node_data[NAME_COLUMNS]
    if how == "sum":   # find hubs by taking the top k summed rank over metrics
        rank_df["rank sum"] = node_data[used_cols].rank().sum(axis=1)
        rank_df = rank_df.nlargest(k, "rank sum", "all")
    elif how == "intersection":     # find hubs by intersecting the top k rank over all metrics
        for col in used_cols:
            col_rank_df = node_data[NAME_COLUMNS]
            col_rank_df["rank"] = node_data[col].rank()
            col_rank_df = col_rank_df.nlargest(k, "rank", "all")
            rank_df = rank_df.merge(col_rank_df[NAME_COLUMNS], on=NAME_COLUMNS)
    top_k_bool = node_data["stringId"].isin(rank_df["stringId"])

    # empty warning
    if not any(top_k_bool):
        print("\nWarning: the consensus method for finding hub proteins yielded no results\n")

    if verbose:
        print(rank_df)
    
    return top_k_bool


class HubsPPI:
    def find_hubs(self, method="sd", metric="degree", k=10, n_sd=2, ascending=False, how="intersection"):
        """..."""

        # input check
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")
        if not isinstance(method, str):
            raise TypeError("'method' must be a string")
        if method not in HUB_METHODS:
            raise ValueError(f"'method' must be one of {HUB_METHODS}")

        # method selection
        if method == "topk":
            self.node_data[f"is hub ({method})"] = hubs_by_topk(self.node_data, metric, k, ascending)
        elif method == "sd":
            self.node_data[f"is hub ({method})"] = hubs_by_sd(self.node_data, metric, n_sd)
        elif method == "consensus":
            self.node_data[f"is hub ({method})"] = hubs_by_consensus(self.node_data, metric, k, how)

        # hub addition
        self.hubs = self.node_data.loc[self.node_data[f"is hub ({method})"]]

        return self


if __name__ == "__main__":
    test_node_data = pd.read_csv(DATA_DIR / "ex_node_data.csv")
    print(test_node_data)
    print(hubs_by_consensus(test_node_data, how="intersection", verbose=True, k=2, metrics="degree"))