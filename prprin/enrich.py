import stringdb
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability
NAME_COLUMNS = ["stringId", "symbol"]
POSSIBLE_ENRICHMENT_BACKGROUND_STR = ("network", "genome")


class EnrichPPI:
    def enrich(self, hub_method=None, background="network"):

        # input check
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")
        if not isinstance(self.hubs, pd.DataFrame):
            raise ValueError("there are no hubs saved in the 'hubs' container. Run find_hubs() first")
        if hub_method is None and len([col for col in self.node_data.columns if "is hub" in col]) != 1:
            raise ValueError("there is not one single 'is_hub' boolean mask in 'node_data'. Please select one specific mask in hub_method")
        if isinstance(hub_method, str) and hub_method not in self.node_data.columns:
            raise ValueError("hub_method is not a column present in 'node_data'. Duble check what method was used in find_hubs()")
        if not isinstance(background, (str, list, tuple, pd.DataFrame, pd.Series)):
            raise TypeError("background must be one of: str, list, tuple, pd.DataFrame")
        if isinstance(background, str) and background not in POSSIBLE_ENRICHMENT_BACKGROUND_STR:
            raise ValueError(f"background string value must be one of: {POSSIBLE_ENRICHMENT_BACKGROUND_STR}")

        # choose hub boolean mask
        if hub_method is None:
            selected_hub_mask = [col for col in self.node_data.columns if "is hub" in col][0]
        else:
            selected_hub_mask = hub_method

        # enrichment calculation
        hub_method_str = selected_hub_mask[selected_hub_mask.rfind("("):selected_hub_mask.rfind(")")+1]
        if background == "network":
            enrichment = stringdb.get_enrichment(self.node_data.loc[self.node_data[selected_hub_mask]]["stringId"], 
                                                      self.node_data["stringId"],
                                                      self.specie_id)
            
            run_name = f"hubs {hub_method_str} vs {background}"
        elif background == "genome":
            enrichment = stringdb.get_enrichment(self.node_data.loc[self.node_data[selected_hub_mask]]["stringId"],
                                                      species=self.specie_id)
            run_name = f"hubs {hub_method_str} vs {background}"
        elif isinstance(background, pd.DataFrame):
            enrichment = stringdb.get_enrichment(self.node_data.loc[self.node_data[selected_hub_mask]]["stringId"], 
                                                      background["stringId"],
                                                      self.specie_id)
            run_name = f"hubs {hub_method_str} vs custom background"
        else:
            enrichment = stringdb.get_enrichment(self.node_data.loc[self.node_data[selected_hub_mask]]["stringId"], 
                                                      background,
                                                      self.specie_id)
            run_name = f"hubs {hub_method_str} vs custom background"
        if len(enrichment) == 0:
            print("\nWarning: enrich() yielded no results\n")

        # self.encrichment asignment
        enrichment["run"] = run_name
        if self.enrichment is None:
            self.enrichment = enrichment
        else:
            self.enrichment = pd.concat([self.enrichment, enrichment])

        return self


if __name__ == "__main__":
    node_data = pd.read_csv(DATA_DIR / "ex_large_node_data.csv")
    hubs = node_data.loc[node_data["is hub (consensus)"]]
    enrichment = stringdb.get_enrichment(hubs["stringId"])
    print(enrichment.T)