"""Unit tests of the prprin.graph module

These tests replace the '__main__' block of prprin/graph.py, which only held
commented out calls.
"""

import unittest

import networkx as nx
import pandas as pd

from prprin.core import PrPrInCore
from prprin.graph import EDGE_SCORES, GraphPPI


class GraphObject(PrPrInCore, GraphPPI):
    """Minimal object holding only what create_graph() needs"""


def build_network_df():
    """Build a small interaction table shaped like the one STRING returns"""
    edges = [("a", "b", 0.9), ("b", "c", 0.4), ("c", "a", 0.7)]
    rows = []
    for node_a, node_b, score in edges:
        row = {"stringId_A": node_a, "stringId_B": node_b}
        row.update({score_name: score for score_name in EDGE_SCORES})
        rows.append(row)
    return pd.DataFrame(rows)


class TestCreateGraph(unittest.TestCase):
    def test_graph_is_built_from_the_network_table(self):
        """Every interaction of the table becomes an edge of the graph"""
        obj = GraphObject(network=build_network_df())

        obj.create_graph()

        self.assertIsInstance(obj.graph, nx.Graph)
        self.assertEqual(obj.graph.number_of_nodes(), 3)
        self.assertEqual(obj.graph.number_of_edges(), 3)
        self.assertEqual(set(obj.graph.nodes), {"a", "b", "c"})

    def test_every_score_is_kept_on_the_edges(self):
        """The metadata of the interactions is preserved as edge attributes"""
        obj = GraphObject(network=build_network_df()).create_graph()

        edge_attributes = obj.graph.edges["a", "b"]

        self.assertEqual(sorted(edge_attributes), sorted(EDGE_SCORES))
        self.assertAlmostEqual(edge_attributes["score"], 0.9)

    def test_create_graph_returns_self_to_allow_chaining(self):
        """create_graph() gives the object back so the methods can be chained"""
        obj = GraphObject(network=build_network_df())

        self.assertIs(obj.create_graph(), obj)

    def test_missing_network_raises(self):
        """Building a graph without interaction data is an error"""
        with self.assertRaises(ValueError):
            GraphObject().create_graph()

    def test_wrong_network_type_raises(self):
        """The interaction data has to be a DataFrame"""
        with self.assertRaises(ValueError):
            GraphObject(network="not a DataFrame").create_graph()


if __name__ == "__main__":
    unittest.main()
