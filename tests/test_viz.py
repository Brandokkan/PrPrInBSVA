"""Unit tests of the prprin.viz module

These tests replace the '__main__' block of prprin/viz.py, which read the
example node data and only wrote a clustermap of it.

Every plot is written inside a temporary folder, so running the tests does not
touch the 'plots' folder of the project.
"""

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from prprin.viz import (BOOL_COLORS, DEFAULT_COLOR, HIGHLIGHT_COLOR, _scale,
                        enrichment_dotplot, labels_to_ids, metrics_clustermap,
                        node_colors, node_tooltips, plot_a_graph,
                        relabel_select_menu, resolve_highlight)

STRINGID_INDEX_NAME = "stringId"


def build_node_data():
    """Build a node table carrying a hub mask and a few metrics"""
    node_data = pd.DataFrame(
        {
            "symbol": ["TP53", "BRCA1", "ATM", "MDM2"],
            "degree": [3.0, 2.0, 2.0, 1.0],
            "betweenness centrality": [0.5, 0.1, 0.1, 0.0],
            "weighted degree (score)": [2.4, 1.5, 1.5, 0.9],
        },
        index=pd.Index(["id1", "id2", "id3", "id4"], name=STRINGID_INDEX_NAME),
    )
    node_data["is hub (topk)"] = [True, False, False, False]
    return node_data


def build_graph():
    """Build the graph matching the node table"""
    graph = nx.Graph()
    graph.add_edges_from((("id1", "id2", {"score": 0.9}), ("id1", "id3", {"score": 0.8}),
                          ("id1", "id4", {"score": 0.7}), ("id2", "id3", {"score": 0.6})))
    return graph


def build_enrichment():
    """Build an enrichment table shaped like the one enrich() stores"""
    return pd.DataFrame({
        "category": ["Process", "Process", "KEGG"],
        "term": ["GO:0006281", "GO:0006974", "hsa04115"],
        "description": ["DNA repair", "DNA damage response", "p53 signaling pathway"],
        "number_of_genes": [3, 2, 2],
        "number_of_genes_in_background": [30, 40, 20],
        "inputGenes": ["id1,id2,id3", "id1,id2", "id1,id4"],
        "preferredNames": ["TP53,BRCA1,ATM", "TP53,BRCA1", "TP53,MDM2"],
        "p_value": [0.0001, 0.001, 0.01],
        # a zero fdr is what STRING really answers for the strongest terms
        "fdr": [0.0, 0.001, 0.02],
        "run": ["hubs (topk) vs network"] * 2 + ["hubs (consensus) vs network"],
    })


class TestScale(unittest.TestCase):
    def test_values_are_mapped_on_the_whole_range(self):
        scaled = _scale([0.0, 5.0, 10.0], 10, 50)

        self.assertAlmostEqual(scaled[0], 10)
        self.assertAlmostEqual(scaled[-1], 50)
        self.assertAlmostEqual(scaled[1], 30)

    def test_a_constant_input_lands_in_the_middle(self):
        """A metric that never changes must not give zero sized nodes"""
        scaled = _scale([7.0, 7.0, 7.0], 10, 50)

        self.assertTrue(np.allclose(scaled, 30))

    def test_an_empty_input_stays_empty(self):
        self.assertEqual(len(_scale([], 10, 50)), 0)


class TestNodeColors(unittest.TestCase):
    def test_no_column_gives_the_default_color(self):
        colors, legend = node_colors(build_node_data(), color_by=None)

        self.assertTrue((colors == DEFAULT_COLOR).all())
        self.assertEqual(legend["kind"], "none")

    def test_a_boolean_column_gives_two_colors(self):
        """The hub masks are drawn with two contrasting colors"""
        colors, legend = node_colors(build_node_data(), color_by="is hub (topk)")

        self.assertEqual(colors.loc["id1"], BOOL_COLORS[True])
        self.assertEqual(colors.loc["id2"], BOOL_COLORS[False])
        self.assertEqual(legend["kind"], "categorical")

    def test_a_numeric_column_gives_a_gradient(self):
        """The metrics are drawn with a continuous color map"""
        colors, legend = node_colors(build_node_data(), color_by="degree")

        self.assertEqual(legend["kind"], "gradient")
        self.assertEqual(legend["vmin"], 1.0)
        self.assertEqual(legend["vmax"], 3.0)
        self.assertTrue(colors.str.startswith("#").all())

    def test_unknown_column_raises(self):
        with self.assertRaises(ValueError):
            node_colors(build_node_data(), color_by="not a column")


class TestNodeTooltips(unittest.TestCase):
    def test_every_node_gets_a_tooltip(self):
        tooltips = node_tooltips(build_node_data())

        self.assertEqual(set(tooltips), set(build_node_data().index))

    def test_the_tooltip_carries_every_metric(self):
        """The metadata of a protein is all shown in its tooltip"""
        tooltip = node_tooltips(build_node_data())["id1"]

        self.assertIn("TP53", tooltip)
        self.assertIn("degree", tooltip)
        self.assertIn("is hub (topk)", tooltip)

    def test_the_tooltip_is_plain_text_on_several_lines(self):
        """No HTML is used, because vis-network writes it with innerText"""
        tooltip = node_tooltips(build_node_data())["id1"]

        self.assertIn("\n", tooltip)
        self.assertNotIn("<", tooltip)


class TestLabelsToIds(unittest.TestCase):
    def test_symbols_and_stringids_are_both_accepted(self):
        found = labels_to_ids(["TP53", "id2"], build_node_data(), verbose=False)

        self.assertEqual(list(found), ["id1", "id2"])

    def test_unknown_identifiers_are_dropped(self):
        found = labels_to_ids(["TP53", "NOT A PROTEIN"], build_node_data(), verbose=False)

        self.assertEqual(list(found), ["id1"])

    def test_repeated_identifiers_are_kept_once(self):
        found = labels_to_ids(["TP53", "id1"], build_node_data(), verbose=False)

        self.assertEqual(list(found), ["id1"])


class TestResolveHighlight(unittest.TestCase):
    def test_none_highlights_nothing(self):
        self.assertEqual(len(resolve_highlight(None, build_node_data())), 0)

    def test_a_hub_mask_column_selects_the_hubs(self):
        found = resolve_highlight("is hub (topk)", build_node_data())

        self.assertEqual(list(found), ["id1"])

    def test_an_enriched_term_selects_its_proteins(self):
        """The results of the enrichment can be shown back on the network"""
        found = resolve_highlight("GO:0006281", build_node_data(), build_enrichment())

        self.assertEqual(set(found), {"id1", "id2", "id3"})

    def test_the_description_of_a_term_works_too(self):
        found = resolve_highlight("DNA repair", build_node_data(), build_enrichment())

        self.assertEqual(set(found), {"id1", "id2", "id3"})

    def test_a_list_of_symbols_is_accepted(self):
        found = resolve_highlight(["TP53", "MDM2"], build_node_data(), build_enrichment())

        self.assertEqual(set(found), {"id1", "id4"})

    def test_a_non_boolean_column_raises(self):
        with self.assertRaises(TypeError):
            resolve_highlight("degree", build_node_data())


class TestRelabelSelectMenu(unittest.TestCase):
    """The node dropdown must let the user search by gene symbol"""

    def test_the_option_text_becomes_the_label_but_the_value_stays_the_id(self):
        html = ('<select id="select-node"><option selected>Select a Node by ID</option>'
                '<option value="id1">id1</option></select>')

        relabelled = relabel_select_menu(html, {"id1": "TP53"})

        self.assertIn('<option value="id1">TP53</option>', relabelled)
        self.assertIn("Select a Node by symbol", relabelled)

    def test_an_unknown_id_keeps_its_own_text(self):
        html = '<select id="select-node"><option value="id9">id9</option></select>'

        relabelled = relabel_select_menu(html, {"id1": "TP53"})

        self.assertIn('<option value="id9">id9</option>', relabelled)


class TestPlotAGraph(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_a_self_contained_page_is_written(self):
        written = plot_a_graph(build_graph(), build_node_data(),
                               color_by="is hub (topk)", file=self.out / "network.html")

        self.assertTrue(written.is_file())
        html = written.read_text(encoding="utf-8")
        self.assertIn("vis-network", html)
        self.assertIn("TP53", html)

    def test_the_heading_is_written_only_once(self):
        """pyvis puts the heading in its template twice, only one must survive"""
        written = plot_a_graph(build_graph(), build_node_data(),
                               heading="my heading", file=self.out / "network.html")

        self.assertEqual(written.read_text(encoding="utf-8").count("my heading"), 1)

    def test_the_source_graph_is_not_modified(self):
        """pyvis writes into the graph it is given, so a copy must be used"""
        graph = build_graph()
        edges_before = {(a, b): dict(data) for a, b, data in graph.edges(data=True)}

        plot_a_graph(graph, build_node_data(), file=self.out / "network.html")

        self.assertTrue(all(not attributes for _, attributes in graph.nodes(data=True)))
        self.assertEqual({(a, b): dict(data) for a, b, data in graph.edges(data=True)}, edges_before)

    def test_interactions_below_the_minimum_score_are_dropped(self):
        written = plot_a_graph(build_graph(), build_node_data(),
                               min_score=0.75, file=self.out / "network.html")

        html = written.read_text(encoding="utf-8")
        # only the 0.9 and 0.8 interactions survive the filter
        self.assertEqual(html.count('"score":'), 2)

    def test_the_highlighted_nodes_get_a_ring(self):
        written = plot_a_graph(build_graph(), build_node_data(), highlight="is hub (topk)",
                               file=self.out / "network.html")

        self.assertIn(HIGHLIGHT_COLOR, written.read_text(encoding="utf-8"))

    def test_unknown_size_column_raises(self):
        with self.assertRaises(ValueError):
            plot_a_graph(build_graph(), build_node_data(), size_by="not a column",
                         file=self.out / "network.html")

    def test_a_non_numeric_size_column_raises(self):
        with self.assertRaises(TypeError):
            plot_a_graph(build_graph(), build_node_data(), size_by="symbol",
                         file=self.out / "network.html")


class TestMetricsClustermap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_an_image_is_written(self):
        written = metrics_clustermap(build_node_data(), metrics_to_show="all",
                                     file=self.out / "clustermap.png")

        self.assertTrue(written.is_file())
        self.assertGreater(written.stat().st_size, 0)

    def test_the_hub_strip_is_accepted(self):
        written = metrics_clustermap(build_node_data(), metrics_to_show="all",
                                     hub_method="is hub (topk)", normalize=True,
                                     file=self.out / "clustermap.png")

        self.assertTrue(written.is_file())

    def test_wrong_metrics_type_raises(self):
        with self.assertRaises(TypeError):
            metrics_clustermap("not a DataFrame", file=self.out / "clustermap.png")

    def test_unknown_hub_method_raises(self):
        with self.assertRaises(ValueError):
            metrics_clustermap(build_node_data(), hub_method="is hub (never run)",
                               file=self.out / "clustermap.png")


class TestEnrichmentDotplot(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_an_image_is_written(self):
        written = enrichment_dotplot(build_enrichment(), file=self.out / "enrichment.png")

        self.assertTrue(written.is_file())
        self.assertGreater(written.stat().st_size, 0)

    def test_a_zero_fdr_does_not_break_the_plot(self):
        """STRING answers fdr = 0 for the strongest terms, -log10 would be infinite"""
        enrichment = build_enrichment()
        enrichment["fdr"] = 0.0

        written = enrichment_dotplot(enrichment, category="all", file=self.out / "enrichment.png")

        self.assertTrue(written.is_file())

    def test_several_runs_are_plotted_together(self):
        written = enrichment_dotplot(build_enrichment(), category="all",
                                     file=self.out / "enrichment.png")

        self.assertTrue(written.is_file())

    def test_an_empty_selection_writes_nothing(self):
        """Asking for a category with no term is a warning, not an error"""
        self.assertIsNone(enrichment_dotplot(build_enrichment(), category="RCTM",
                                             file=self.out / "enrichment.png"))

    def test_unknown_run_raises(self):
        with self.assertRaises(ValueError):
            enrichment_dotplot(build_enrichment(), run="never run", file=self.out / "enrichment.png")

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            enrichment_dotplot(build_enrichment().drop(columns=["fdr"]),
                               file=self.out / "enrichment.png")

    def test_non_positive_top_n_raises(self):
        with self.assertRaises(TypeError):
            enrichment_dotplot(build_enrichment(), top_n=0, file=self.out / "enrichment.png")


if __name__ == "__main__":
    unittest.main()
