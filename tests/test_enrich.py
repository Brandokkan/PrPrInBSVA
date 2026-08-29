"""Unit tests of the prprin.enrich module

These tests replace the '__main__' block of prprin/enrich.py, which asked the
STRING servers for a real enrichment and only printed it.

The enrichment itself needs the STRING servers, so what is tested here is the
checking of the parameters, which is what has to reject a wrong call before any
request is sent. The call to STRING is replaced by a stub, so the tests stay
offline and repeatable.
"""

import unittest
from unittest import mock

import pandas as pd

from prprin.core import PrPrInCore
from prprin.enrich import POSSIBLE_ENRICHMENT_BACKGROUND_STR, EnrichPPI

STRINGID_INDEX_NAME = "stringId"


class EnrichObject(PrPrInCore, EnrichPPI):
    """Minimal object holding only what enrich() needs"""


def build_node_data(hub_columns=("is hub (topk)",)):
    """Build a node table already carrying the wanted hub masks"""
    node_data = pd.DataFrame(
        {"symbol": ["TP53", "BRCA1", "ATM"], "degree": [10.0, 5.0, 1.0]},
        index=pd.Index(["id1", "id2", "id3"], name=STRINGID_INDEX_NAME),
    )
    for column in hub_columns:
        node_data[column] = [True, False, False]
    return node_data


def build_enrichment_answer():
    """Build the table the STRING enrichment service answers with"""
    return pd.DataFrame({
        "category": ["Process"],
        "term": ["GO:0006281"],
        "description": ["DNA repair"],
        "number_of_genes": [1],
        "number_of_genes_in_background": [10],
        "inputGenes": ["id1"],
        "preferredNames": ["TP53"],
        "p_value": [0.001],
        "fdr": [0.01],
    })


def build_ready_object(hub_columns=("is hub (topk)",)):
    """Build an object that already went through find_hubs()"""
    obj = EnrichObject()
    obj.node_data = build_node_data(hub_columns)
    obj.hubs = obj.node_data.loc[obj.node_data[hub_columns[0]]]
    return obj


class TestEnrichParameterChecks(unittest.TestCase):
    """No request must leave the program when the parameters are wrong"""

    def test_missing_node_data_raises(self):
        """The enrichment cannot run before the metrics are calculated"""
        with self.assertRaises(ValueError):
            EnrichObject().enrich()

    def test_missing_hubs_raises(self):
        """The enrichment cannot run before the hubs are found"""
        obj = EnrichObject()
        obj.node_data = build_node_data(hub_columns=())

        with self.assertRaises(ValueError):
            obj.enrich()

    def test_several_hub_masks_without_a_choice_raises(self):
        """With more than one hub method, one has to be named explicitly"""
        obj = build_ready_object(hub_columns=("is hub (topk)", "is hub (consensus)"))

        with self.assertRaises(ValueError):
            obj.enrich()

    def test_unknown_hub_method_raises(self):
        with self.assertRaises(ValueError):
            build_ready_object().enrich("is hub (never run)")

    def test_wrong_background_type_raises(self):
        with self.assertRaises(TypeError):
            build_ready_object().enrich(background=42)

    def test_unknown_background_string_raises(self):
        """Only the two documented background strings are accepted"""
        with self.assertRaises(ValueError):
            build_ready_object().enrich(background="the whole universe")

    def test_the_documented_background_strings_are_accepted(self):
        """'network' and 'genome' are the two valid special backgrounds"""
        self.assertEqual(POSSIBLE_ENRICHMENT_BACKGROUND_STR, ("network", "genome"))


class TestEnrichResults(unittest.TestCase):
    """What the method does with the answer of STRING, without calling it"""

    def test_the_answer_is_stored_and_labelled_with_the_run(self):
        """The results are saved with a label telling how they were obtained"""
        obj = build_ready_object()

        with mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *args, **kwargs: build_enrichment_answer()):
            obj.enrich("is hub (topk)", background="network")

        self.assertIsInstance(obj.enrichment, pd.DataFrame)
        self.assertIn("run", obj.enrichment.columns)
        self.assertEqual(obj.enrichment["run"].iloc[0], "hubs (topk) vs network")

    def test_several_runs_are_concatenated(self):
        """Running the enrichment twice keeps both results, told apart"""
        obj = build_ready_object(hub_columns=("is hub (topk)", "is hub (consensus)"))

        with mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *args, **kwargs: build_enrichment_answer()):
            obj.enrich("is hub (topk)", background="network")
            obj.enrich("is hub (consensus)", background="genome")

        self.assertEqual(len(obj.enrichment), 2)
        self.assertEqual(set(obj.enrichment["run"]),
                         {"hubs (topk) vs network", "hubs (consensus) vs genome"})

    def test_a_custom_background_is_labelled_as_such(self):
        """A background given as a list is reported as custom"""
        obj = build_ready_object()

        with mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *args, **kwargs: build_enrichment_answer()):
            obj.enrich("is hub (topk)", background=["id1", "id2", "id3"])

        self.assertIn("custom background", obj.enrichment["run"].iloc[0])

    def test_only_the_hubs_are_sent_to_string(self):
        """The proteins sent for the enrichment are the flagged ones only"""
        obj = build_ready_object()

        with mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *args, **kwargs: build_enrichment_answer()) as fake_enrichment:
            obj.enrich("is hub (topk)", background="network")

        sent_proteins = list(fake_enrichment.call_args.args[0])
        self.assertEqual(sent_proteins, ["id1"])

    def test_enrich_returns_self_to_allow_chaining(self):
        obj = build_ready_object()

        with mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *args, **kwargs: build_enrichment_answer()):
            self.assertIs(obj.enrich("is hub (topk)"), obj)

    def test_an_empty_answer_is_stored_without_crashing(self):
        """STRING finding nothing is a warning, not an error"""
        obj = build_ready_object()
        empty_answer = build_enrichment_answer().iloc[0:0]

        with mock.patch("prprin.enrich.stringdb.get_enrichment", side_effect=lambda *args, **kwargs: empty_answer.copy()):
            obj.enrich("is hub (topk)")

        self.assertEqual(len(obj.enrichment), 0)


if __name__ == "__main__":
    unittest.main()
