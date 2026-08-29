"""Unit tests of the prprin.metrics module

These tests replace the '__main__' block of prprin/metrics.py, which built a
small toy graph and only printed the metrics calculated on it. The same toy
graph is used here, with the printed values turned into assertions.
"""

import unittest

import networkx as nx
import pandas as pd

from prprin.metrics import (STRINGID_INDEX_NAME, calculate_unweighted_metrics,
                            calculate_weighted_metrics)

UNWEIGHTED_NAMES = ["degree", "degree centrality", "betweenness centrality", "clustering coefficient"]


def build_toy_graph():
    """Build the small weighted graph the '__main__' block used to print"""
    graph = nx.Graph()
    graph.add_edges_from((("a", "b", {"score": 0.8}), ("c", "d", {"score": 0.3}),
                          ("d", "b", {"score": 0}), ("d", "a", {"score": 0.9}),
                          ("c", "e", {"score": 1}), ("c", "f", {"score": 0.95})))
    return graph


def build_index_df():
    """Build the stringId to symbol table of the toy graph"""
    return pd.DataFrame({"symbol": ["azz", "bee", "caz", "dam", "ez", "fuc"]},
                        index=pd.Index(["a", "b", "c", "d", "e", "f"], name=STRINGID_INDEX_NAME))


class TestUnweightedMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = calculate_unweighted_metrics(build_toy_graph(), build_index_df())

    def test_one_row_per_node_indexed_by_stringid(self):
        """Every node of the graph gets a row, keyed by its stringId"""
        self.assertEqual(len(self.metrics), 6)
        self.assertEqual(self.metrics.index.name, STRINGID_INDEX_NAME)
        self.assertEqual(set(self.metrics.index), set("abcdef"))

    def test_the_symbol_is_kept_beside_the_metrics(self):
        """The metadata of the proteins survives the merge with the metrics"""
        self.assertEqual(self.metrics.at["a", "symbol"], "azz")

    def test_every_unweighted_metric_is_present(self):
        """The four required unweighted metrics are calculated"""
        for name in UNWEIGHTED_NAMES:
            with self.subTest(metric=name):
                self.assertIn(name, self.metrics.columns)

    def test_degree_matches_the_graph(self):
        """The degree of a node is the number of its interactions"""
        graph = build_toy_graph()

        for node in graph.nodes:
            with self.subTest(node=node):
                self.assertEqual(self.metrics.at[node, "degree"], graph.degree(node))

    def test_leaf_nodes_have_no_betweenness(self):
        """A node with a single interaction lies on no shortest path"""
        self.assertAlmostEqual(self.metrics.at["e", "betweenness centrality"], 0.0)
        self.assertAlmostEqual(self.metrics.at["f", "betweenness centrality"], 0.0)

    def test_degree_centrality_is_the_normalized_degree(self):
        """The degree centrality is the degree divided by the other nodes"""
        self.assertAlmostEqual(self.metrics.at["c", "degree centrality"], 3 / 5)

    def test_wrong_graph_type_raises(self):
        with self.assertRaises(TypeError):
            calculate_unweighted_metrics("not a graph", build_index_df())

    def test_wrong_index_type_raises(self):
        with self.assertRaises(TypeError):
            calculate_unweighted_metrics(build_toy_graph(), "not a DataFrame")


class TestWeightedMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = calculate_weighted_metrics(build_toy_graph(), build_index_df())

    def test_the_used_weight_is_written_in_the_column_names(self):
        """The score used as the weight is shown between parentheses"""
        for column in self.metrics.columns:
            if column != "symbol":
                with self.subTest(column=column):
                    self.assertIn("score", column)

    def test_weighted_degree_is_the_sum_of_the_scores(self):
        """The weighted degree adds up the scores of the interactions"""
        # c is linked to d (0.3), e (1) and f (0.95)
        self.assertAlmostEqual(self.metrics.at["c", "weighted degree (score)"], 0.3 + 1 + 0.95)

    def test_normalized_weighted_degree_sums_to_one(self):
        """The normalized weighted degree is a fraction of the total"""
        self.assertAlmostEqual(self.metrics["weighted degree normalized (score)"].sum(), 1.0)

    def test_the_three_distance_conversions_are_calculated(self):
        """The scores are turned into distances in the three documented ways"""
        for conversion in ("1 - score", "1 / score", "-log(score)"):
            with self.subTest(conversion=conversion):
                self.assertIn(f"weighted betweenness centrality ({conversion})", self.metrics.columns)

    def test_a_zero_score_does_not_break_the_conversions(self):
        """The edge of weight zero gives finite distances, not infinities"""
        # the b-d edge has a score of exactly 0, which 1/score and -log(score) cannot take
        for conversion in ("1 / score", "-log(score)"):
            with self.subTest(conversion=conversion):
                values = self.metrics[f"weighted betweenness centrality ({conversion})"]
                self.assertTrue(values.notna().all())
                self.assertTrue((values.abs() != float("inf")).all())

    def test_unknown_weight_raises(self):
        """Only the STRING scores can be used as the weight"""
        with self.assertRaises(ValueError):
            calculate_weighted_metrics(build_toy_graph(), build_index_df(), w_type="not a score")

    def test_wrong_weight_type_raises(self):
        with self.assertRaises(TypeError):
            calculate_weighted_metrics(build_toy_graph(), build_index_df(), w_type=42)


if __name__ == "__main__":
    unittest.main()
