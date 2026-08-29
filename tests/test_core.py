"""Unit tests of the prprin.core module

These tests replace the '__main__' block of prprin/core.py, which only printed
a freshly built container.
"""

import unittest

import pandas as pd

from prprin.core import PrPrInCore


class TestPrPrInCore(unittest.TestCase):
    def test_empty_container_defaults(self):
        """A new container starts empty, human and indexed by stringId"""
        core = PrPrInCore()

        self.assertEqual(len(core.proteins), 0)
        self.assertEqual(core.proteins.index.name, "stringId")
        self.assertEqual(list(core.proteins.columns), ["symbol"])
        self.assertEqual(core.specie_id, 9606)

    def test_processed_containers_start_empty(self):
        """Every container filled by the analysis starts at None"""
        core = PrPrInCore()

        for container in ("network", "graph", "node_data", "used_weight_name", "hubs", "enrichment"):
            with self.subTest(container=container):
                self.assertIsNone(getattr(core, container))

    def test_given_values_are_stored(self):
        """The values given to the constructor end up in the containers"""
        proteins = pd.DataFrame({"symbol": ["TP53"]}, index=pd.Index(["9606.ENSP00000269305"], name="stringId"))
        network = pd.DataFrame({"stringId_A": ["a"], "stringId_B": ["b"]})

        core = PrPrInCore(proteins=proteins, network=network, specie_id=10090)

        self.assertIs(core.proteins, proteins)
        self.assertIs(core.network, network)
        self.assertEqual(core.specie_id, 10090)

    def test_repr_of_an_empty_container(self):
        """The representation of an empty container reports no data"""
        text = repr(PrPrInCore())

        self.assertIn("0 proteins in the object", text)
        self.assertIn("9606", text)
        self.assertIn("none", text)

    def test_repr_lists_the_filled_containers(self):
        """The representation names the containers that hold something"""
        core = PrPrInCore()
        core.network = pd.DataFrame({"stringId_A": ["a"]})

        self.assertIn("network", repr(core))


if __name__ == "__main__":
    unittest.main()
