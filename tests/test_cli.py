"""Unit tests of the prprin.cli module

These tests check that every parameter of the analysis can be given on the
command line, which is what makes the program usable inside an automated
pipeline.

Nothing here reaches the STRING servers: the parser and the helpers are tested
on their own, and the analysis itself is checked with a stubbed retrieval.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from prprin.cli import (build_parser, main, resolve_background,
                        resolve_highlight_argument, run_analysis)
from prprin.core import PrPrInCore

STRINGID_INDEX_NAME = "stringId"


def build_node_data():
    """Build a node table already carrying a hub mask"""
    node_data = pd.DataFrame(
        {"symbol": ["TP53", "BRCA1"], "degree": [2.0, 1.0]},
        index=pd.Index(["id1", "id2"], name=STRINGID_INDEX_NAME),
    )
    node_data["is hub (topk)"] = [True, False]
    return node_data


def build_enrichment():
    """Build an enrichment table shaped like the one enrich() stores"""
    return pd.DataFrame({
        "term": ["GO:0006281"],
        "description": ["DNA repair"],
        "run": ["hubs (topk) vs network"],
    })


class TestParser(unittest.TestCase):
    def test_the_proteins_are_required(self):
        """The program cannot be run without saying what to analyse"""
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_the_defaults_do_not_hide_a_hard_coded_path(self):
        """Nothing but the proteins is needed, and no input path is built in"""
        args = build_parser().parse_args(["TP53 BRCA1"])

        self.assertEqual(args.proteins, "TP53 BRCA1")
        self.assertEqual(args.species, 9606)
        self.assertEqual(args.hub_method, ["consensus"])
        self.assertEqual(args.background, "network")

    def test_every_stage_of_the_analysis_can_be_tuned(self):
        """The parameters of each step are reachable from the command line"""
        args = build_parser().parse_args([
            "data/test_big", "--species", "10090", "--required-score", "700",
            "--weight", "escore", "--hub-method", "topk", "sd", "--hub-metric",
            "betweenness centrality", "--top-k", "7", "--n-sd", "3",
            "--background", "genome", "--category", "KEGG", "--top-n", "5",
            "--outdir", "results", "--prefix", "run1_",
        ])

        self.assertEqual(args.species, 10090)
        self.assertEqual(args.required_score, 700)
        self.assertEqual(args.weight, "escore")
        self.assertEqual(args.hub_method, ["topk", "sd"])
        self.assertEqual(args.hub_metric, "betweenness centrality")
        self.assertEqual(args.top_k, 7)
        self.assertEqual(args.n_sd, 3)
        self.assertEqual(args.background, "genome")
        self.assertEqual(args.category, "KEGG")
        self.assertEqual(args.top_n, 5)
        self.assertEqual(args.outdir, "results")
        self.assertEqual(args.prefix, "run1_")

    def test_an_unknown_hub_method_is_refused(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["TP53", "--hub-method", "not a method"])

    def test_an_unknown_weight_is_refused(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["TP53", "--weight", "not a score"])

    def test_the_steps_can_be_skipped(self):
        args = build_parser().parse_args(["TP53", "--no-enrich", "--no-network", "--quiet"])

        self.assertTrue(args.no_enrich)
        self.assertTrue(args.no_network)
        self.assertTrue(args.quiet)


class TestResolveBackground(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.background_file = Path(self.temp_dir.name) / "background.txt"
        self.background_file.write_text("TP53\nBRCA1\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_the_special_strings_are_passed_through(self):
        self.assertEqual(resolve_background("network"), "network")
        self.assertEqual(resolve_background("genome"), "genome")

    def test_a_file_becomes_a_custom_background(self):
        """A custom background can be given as a file of identifiers"""
        self.assertEqual(resolve_background(str(self.background_file)), ["TP53", "BRCA1"])

    def test_a_missing_background_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            resolve_background("./no_such_background.txt")


class TestResolveHighlightArgument(unittest.TestCase):
    def setUp(self):
        self.obj = PrPrInCore()
        self.obj.node_data = build_node_data()
        self.obj.enrichment = build_enrichment()

    def test_nothing_given_highlights_nothing(self):
        self.assertIsNone(resolve_highlight_argument(None, self.obj))

    def test_a_hub_mask_column_is_passed_through(self):
        self.assertEqual(resolve_highlight_argument("is hub (topk)", self.obj), "is hub (topk)")

    def test_an_enriched_term_is_passed_through(self):
        self.assertEqual(resolve_highlight_argument("GO:0006281", self.obj), "GO:0006281")

    def test_the_description_of_a_term_is_passed_through(self):
        self.assertEqual(resolve_highlight_argument("DNA repair", self.obj), "DNA repair")

    def test_protein_identifiers_are_split_on_the_spaces(self):
        self.assertEqual(resolve_highlight_argument("TP53 BRCA1", self.obj), ["TP53", "BRCA1"])


class TestRunAnalysis(unittest.TestCase):
    """The pipeline is driven only by the command line parameters"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out = Path(self.temp_dir.name) / "results"

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_with(self, extra_args):
        """Run the analysis with a stubbed retrieval and enrichment"""
        args = build_parser().parse_args(
            ["TP53 BRCA1 ATM MDM2", "--outdir", str(self.out), "--quiet"] + extra_args)

        network = pd.DataFrame({
            "stringId_A": ["id1", "id1", "id1", "id2"],
            "stringId_B": ["id2", "id3", "id4", "id3"],
            "preferredName_A": ["TP53"] * 3 + ["BRCA1"],
            "preferredName_B": ["BRCA1", "ATM", "MDM2", "ATM"],
            "score": [0.9, 0.8, 0.7, 0.6],
        })
        for score_name in ("nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore"):
            network[score_name] = network["score"]

        enrichment_answer = pd.DataFrame({
            "category": ["Process"], "term": ["GO:0006281"], "description": ["DNA repair"],
            "number_of_genes": [2], "number_of_genes_in_background": [20],
            "inputGenes": ["id1,id2"], "preferredNames": ["TP53,BRCA1"],
            "p_value": [0.001], "fdr": [0.01],
        })

        with mock.patch("prprin.retrive.interaction_retrival", return_value=network), \
             mock.patch("prprin.enrich.stringdb.get_enrichment",
                        side_effect=lambda *a, **k: enrichment_answer.copy()):
            return run_analysis(args)

    def test_the_whole_pipeline_fills_every_container(self):
        obj = self.run_with(["--hub-method", "topk", "--top-k", "2"])

        self.assertIsNotNone(obj.network)
        self.assertIsNotNone(obj.graph)
        self.assertIsNotNone(obj.node_data)
        self.assertIsNotNone(obj.hubs)
        self.assertIsNotNone(obj.enrichment)

    def test_the_results_are_written_in_the_chosen_folder(self):
        """--outdir decides where everything goes, nothing is hard coded"""
        self.run_with(["--hub-method", "topk", "--top-k", "2"])

        self.assertTrue((self.out / "network.html").is_file())
        self.assertTrue((self.out / "clustermap.png").is_file())
        self.assertTrue((self.out / "enrichment.png").is_file())

    def test_the_prefix_is_used_for_every_file(self):
        self.run_with(["--hub-method", "topk", "--top-k", "2", "--prefix", "run1_"])

        self.assertTrue((self.out / "run1_network.html").is_file())

    def test_several_hub_methods_give_several_enrichment_runs(self):
        """One enrichment run per hub method, so they can be compared"""
        obj = self.run_with(["--hub-method", "topk", "consensus", "--top-k", "2"])

        self.assertIn("is hub (topk)", obj.node_data.columns)
        self.assertIn("is hub (consensus)", obj.node_data.columns)
        self.assertEqual(len(set(obj.enrichment["run"])), 2)

    def test_the_skipping_flags_are_obeyed(self):
        obj = self.run_with(["--hub-method", "topk", "--top-k", "2",
                             "--no-enrich", "--no-network", "--no-clustermap"])

        self.assertIsNone(obj.enrichment)
        self.assertFalse((self.out / "network.html").exists())
        self.assertFalse((self.out / "clustermap.png").exists())

    def test_the_tables_can_be_saved(self):
        self.run_with(["--hub-method", "topk", "--top-k", "2", "--save-tables"])

        for table in ("network", "node_data", "enrichment"):
            with self.subTest(table=table):
                self.assertTrue((self.out / f"{table}.csv").is_file())


class TestMain(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_cwd = Path.cwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.temp_dir.cleanup()

    def test_a_bad_input_file_gives_a_failing_exit_code(self):
        """An automated pipeline has to see that the analysis did not work"""
        self.assertEqual(main(["./no_such_file.txt", "--quiet"]), 1)

    def test_a_bad_parameter_stops_the_program(self):
        with self.assertRaises(SystemExit):
            main(["TP53", "--hub-method", "not a method"])


if __name__ == "__main__":
    unittest.main()
