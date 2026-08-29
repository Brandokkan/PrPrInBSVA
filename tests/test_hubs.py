"""Unit tests of the prprin.hubs module

These tests replace the '__main__' block of prprin/hubs.py, which read the
example node data and only printed the hubs found by the consensus method.
"""

import unittest

import pandas as pd

from prprin.core import PrPrInCore
from prprin.hubs import HubsPPI, hubs_by_consensus, hubs_by_sd, hubs_by_topk

STRINGID_INDEX_NAME = "stringId"


class HubsObject(PrPrInCore, HubsPPI):
    """Minimal object holding only what find_hubs() needs"""


def build_node_data():
    """Build a node table where the ranking of the proteins is known in advance

    'hub' is the most connected protein of every metric, 'second' follows it,
    and the three remaining ones are far behind.
    """
    return pd.DataFrame(
        {
            "symbol": ["HUB", "SECOND", "SMALL1", "SMALL2", "SMALL3"],
            "degree": [40.0, 30.0, 2.0, 2.0, 1.0],
            "degree centrality": [0.8, 0.6, 0.04, 0.04, 0.02],
            "betweenness centrality": [0.5, 0.3, 0.0, 0.0, 0.0],
            "clustering coefficient": [0.1, 0.2, 1.0, 1.0, 1.0],
        },
        index=pd.Index(["id1", "id2", "id3", "id4", "id5"], name=STRINGID_INDEX_NAME),
    )


class TestHubsByTopk(unittest.TestCase):
    def test_the_k_best_proteins_are_selected(self):
        """The mask flags exactly the k highest values of the metric"""
        mask = hubs_by_topk(build_node_data(), metric="degree", k=2)

        self.assertEqual(list(mask), [True, True, False, False, False])

    def test_ascending_selects_the_lowest_values(self):
        """Asking for ascending order flags the least connected proteins"""
        mask = hubs_by_topk(build_node_data(), metric="degree", k=1, ascending=True)

        self.assertEqual(list(mask), [False, False, False, False, True])

    def test_ties_are_kept(self):
        """Proteins tied with the k-th one are selected too"""
        # id3 and id4 both have a degree of 2, so asking for 3 hubs returns 4 of them
        mask = hubs_by_topk(build_node_data(), metric="degree", k=3)

        self.assertEqual(sum(mask), 4)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            hubs_by_topk(build_node_data(), metric="not a metric")

    def test_non_positive_k_raises(self):
        with self.assertRaises(TypeError):
            hubs_by_topk(build_node_data(), k=0)

    def test_boolean_k_raises(self):
        """A boolean is not accepted as the number of hubs"""
        with self.assertRaises(TypeError):
            hubs_by_topk(build_node_data(), k=True)


class TestHubsBySd(unittest.TestCase):
    def test_only_the_outliers_are_selected(self):
        """Only the proteins above the threshold are flagged"""
        node_data = build_node_data()

        mask = hubs_by_sd(node_data, metric="degree", n_sd=1)

        self.assertTrue(mask.loc["id1"])
        self.assertFalse(mask.loc["id5"])

    def test_a_high_threshold_selects_nobody(self):
        """An unreachable threshold gives an empty, but valid, mask"""
        mask = hubs_by_sd(build_node_data(), metric="degree", n_sd=100)

        self.assertFalse(mask.any())

    def test_non_positive_n_sd_raises(self):
        with self.assertRaises(TypeError):
            hubs_by_sd(build_node_data(), n_sd=0)


class TestHubsByConsensus(unittest.TestCase):
    def test_the_intersection_keeps_the_agreed_proteins(self):
        """A protein is a hub when every metric ranks it in its top k"""
        mask = hubs_by_consensus(build_node_data(), metrics="unweighted", k=1, how="intersection")

        # the clustering coefficient ranks the small proteins first, so no protein is
        # in the top 1 of every unweighted metric at once
        self.assertFalse(mask.any())

    def test_the_sum_of_the_ranks_always_selects_someone(self):
        """Summing the ranks gives a hub even when the metrics disagree"""
        mask = hubs_by_consensus(build_node_data(), metrics="unweighted", k=1, how="sum")

        self.assertTrue(mask.any())

    def test_a_single_metric_behaves_like_topk(self):
        """Asked for one metric, the consensus returns its best proteins"""
        node_data = build_node_data()

        consensus = hubs_by_consensus(node_data, metrics=["degree"], k=2, how="intersection")
        topk = hubs_by_topk(node_data, metric="degree", k=2)

        self.assertEqual(list(consensus), list(topk))


class TestFindHubsMethod(unittest.TestCase):
    def setUp(self):
        self.obj = HubsObject()
        self.obj.node_data = build_node_data()

    def test_the_mask_column_is_named_after_the_method(self):
        """Each method writes its own 'is hub (method)' column"""
        self.obj.find_hubs(method="topk", metric="degree", k=2)

        self.assertIn("is hub (topk)", self.obj.node_data.columns)

    def test_several_methods_add_several_columns(self):
        """Running more methods keeps one column per method"""
        self.obj.find_hubs(method="topk", metric="degree", k=2)
        self.obj.find_hubs(method="sd", metric="degree", n_sd=1)

        self.assertIn("is hub (topk)", self.obj.node_data.columns)
        self.assertIn("is hub (sd)", self.obj.node_data.columns)

    def test_the_hubs_container_holds_the_selected_proteins(self):
        """The 'hubs' container gets the rows of the flagged proteins"""
        self.obj.find_hubs(method="topk", metric="degree", k=2)

        self.assertEqual(len(self.obj.hubs), 2)
        self.assertEqual(set(self.obj.hubs["symbol"]), {"HUB", "SECOND"})

    def test_find_hubs_returns_self_to_allow_chaining(self):
        self.assertIs(self.obj.find_hubs(method="topk", metric="degree", k=2), self.obj)

    def test_missing_node_data_raises(self):
        """Hubs cannot be looked for before the metrics are calculated"""
        with self.assertRaises(ValueError):
            HubsObject().find_hubs()

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            self.obj.find_hubs(method="not a method")


if __name__ == "__main__":
    unittest.main()
