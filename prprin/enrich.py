import stringdb
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # project path
DATA_DIR = BASE_DIR / "data" # data path safe for inter-machine operability
NAME_COLUMNS = ["symbol"]     # the stringId is the index of node_data, not a column
POSSIBLE_ENRICHMENT_BACKGROUND_STR = ("network", "genome")


class EnrichPPI:
    def enrich(self, hub_method=None, background="network"):
        """Perform a functional enrichment analysis on the hub proteins

        This method sends the hub proteins to the STRING functional enrichment
        service and appends the results to the 'enrichment' container.

        The hubs are taken from one of the "is hub (method)" boolean columns of
        'node_data', so calculate_metrics() and find_hubs() must be run first.

        Every call adds a "run" column that labels the results with the hub
        method and the background used, and its output is concatenated to what
        is already stored in 'enrichment'. This way several enrichments can be
        calculated on the same object and told apart afterwards.

        The enrichment can yield no result at all, and the chosen hub selection
        can itself be empty (a hub method can select no protein). In both cases
        a warning is printed, and an empty selection is skipped without adding
        anything to 'enrichment'.

        The species used is the one stored in the 'specie_id' container.

        Parameters
        -------
        hub_method: str or None
            The name of the "is hub (method)" column of 'node_data' telling
            which hubs are used.

            If None (default), the only such column present is used
            automatically. When more than one is present (find_hubs() was run
            with different methods), one of them must be indicated explicitly.

        background: str, list, tuple, pandas.DataFrame or pandas.Series
            The set of proteins the hubs are compared against.

            It accepts four kinds of values:
                - "network", to use every protein present in 'node_data' as the
                background.
                - "genome", to use the whole genome of the species as the
                background (the STRING default).
                - a pandas.DataFrame, to use the string ids contained in its
                index (like the one stored in 'node_data' or 'proteins').
                - a list, tuple or pandas.Series of string ids, to use exactly
                the proteins it contains.
        """

        # input check
        if not isinstance(self.node_data, pd.DataFrame):
            raise ValueError("there is no node data saved in the 'node_data' container. Run calculate_metrics() first")
        if not isinstance(self.hubs, pd.DataFrame):
            raise ValueError("there are no hubs saved in the 'hubs' container. Run find_hubs() first")
        if hub_method is None and len([col for col in self.node_data.columns if "is hub" in col]) != 1:
            raise ValueError("there is not one single 'is_hub' boolean mask in 'node_data'. Please select one specific mask in hub_method")
        if hub_method is not None and not isinstance(hub_method, str):
            raise TypeError("hub_method must be a string telling the 'is hub (method)' column to use, or None")
        if isinstance(hub_method, str) and hub_method not in self.node_data.columns:
            raise ValueError("hub_method is not a column present in 'node_data'. Duble check what method was used in find_hubs()")
        if isinstance(hub_method, str) and not pd.api.types.is_bool_dtype(self.node_data[hub_method]):
            raise TypeError(f"the column '{hub_method}' is not a boolean mask, so it cannot be used to select the hubs. "
                            f"hub_method must be one of {[col for col in self.node_data.columns if 'is hub' in col]}")
        if not isinstance(background, (str, list, tuple, pd.DataFrame, pd.Series)):
            raise TypeError("background must be one of: str, list, tuple, pd.DataFrame or pd.Series")
        if isinstance(background, str) and background not in POSSIBLE_ENRICHMENT_BACKGROUND_STR:
            raise ValueError(f"background string value must be one of: {POSSIBLE_ENRICHMENT_BACKGROUND_STR}")

        # choose hub boolean mask
        if hub_method is None:
            selected_hub_mask = [col for col in self.node_data.columns if "is hub" in col][0]
        else:
            selected_hub_mask = hub_method
        hub_ids = self.node_data.loc[self.node_data[selected_hub_mask]].index

        # empty selection warning. A hub method can select no protein at all (the "sd" and
        # the "consensus" ones already warn about it), and STRING answers a bad request to
        # an enrichment asked on no protein, so the run is skipped instead of stopping the
        # whole analysis on a result that is legitimately empty
        if len(hub_ids) == 0:
            print(f"\nWarning: '{selected_hub_mask}' selects no protein, so no enrichment was calculated for it\n")
            return self

        # enrichment calculation
        hub_method_str = selected_hub_mask[selected_hub_mask.rfind("("):selected_hub_mask.rfind(")")+1]
        if isinstance(background, str) and background == "network":
            enrichment = stringdb.get_enrichment(hub_ids,
                                                      self.node_data.index,
                                                      self.specie_id)

            run_name = f"hubs {hub_method_str} vs {background}"
        elif isinstance(background, str) and background == "genome":
            enrichment = stringdb.get_enrichment(hub_ids,
                                                      species=self.specie_id)
            run_name = f"hubs {hub_method_str} vs {background}"
        elif isinstance(background, pd.DataFrame):
            enrichment = stringdb.get_enrichment(hub_ids,
                                                      background.index,
                                                      self.specie_id)
            run_name = f"hubs {hub_method_str} vs custom background"
        else:
            enrichment = stringdb.get_enrichment(hub_ids,
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
            self.enrichment = pd.concat([self.enrichment, enrichment], ignore_index=True)

        return self
