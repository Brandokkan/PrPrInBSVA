"""Unit tests of the prprin.retrive module

These tests replace the '__main__' block of prprin/retrive.py, which called a
main() function that only printed the unique proteins of the example data (and
was marked to be deleted).

Nothing here touches the STRING servers, so the tests can be run offline.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from prprin.retrive import (DATA_DIR, get_unique_proteins, read_identifiers_file,
                            resolve_input_path)


def build_ppi_df():
    """Build a small interaction table shaped like the one STRING returns"""
    return pd.DataFrame({
        "stringId_A": ["id1", "id2", "id1"],
        "stringId_B": ["id2", "id3", "id3"],
        "preferredName_A": ["TP53", "BRCA1", "TP53"],
        "preferredName_B": ["BRCA1", "FANCL", "FANCL"],
        "score": [0.9, 0.4, 0.7],
    })


class TestGetUniqueProteins(unittest.TestCase):
    def test_every_protein_appears_once(self):
        """The proteins of both columns are gathered without repetitions"""
        proteins = get_unique_proteins(build_ppi_df())

        self.assertEqual(len(proteins), 3)
        self.assertEqual(set(proteins.index), {"id1", "id2", "id3"})

    def test_the_table_is_indexed_by_stringid(self):
        """The result is keyed by stringId and holds the symbols"""
        proteins = get_unique_proteins(build_ppi_df())

        self.assertEqual(proteins.index.name, "stringId")
        self.assertEqual(list(proteins.columns), ["symbol"])
        self.assertEqual(proteins.at["id1", "symbol"], "TP53")

    def test_the_example_file_is_read_correctly(self):
        """The example interaction data of the project can still be used"""
        proteins = get_unique_proteins(pd.read_csv(DATA_DIR / "ex_ppi_df.csv", index_col=0))

        self.assertGreater(len(proteins), 0)
        self.assertIn("TP53", set(proteins["symbol"]))

    def test_wrong_input_type_raises(self):
        with self.assertRaises(TypeError):
            get_unique_proteins("not a DataFrame")


class TestResolveInputPath(unittest.TestCase):
    """The file of proteins must be found wherever the user keeps it

    This is what makes the program runnable from any folder: a path is looked
    for as it was written before being looked for in the data folder.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.genes_file = self.temp_path / "my_genes.txt"
        self.genes_file.write_text("TP53\nBRCA1\n\nATM\n", encoding="utf-8")
        self.previous_cwd = Path.cwd()

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.temp_dir.cleanup()

    def test_an_absolute_path_is_found(self):
        """A file given with its full path is read where it is"""
        self.assertEqual(resolve_input_path(str(self.genes_file)), self.genes_file)

    def test_a_path_relative_to_the_working_directory_is_found(self):
        """A file beside the user, not beside the project, is found too"""
        os.chdir(self.temp_path)

        found = resolve_input_path("./my_genes.txt")

        self.assertIsNotNone(found)
        self.assertTrue(found.samefile(self.genes_file))

    def test_a_bare_name_still_reaches_the_data_folder(self):
        """The example files of the project keep working by their bare name"""
        found = resolve_input_path("test_big")

        self.assertIsNotNone(found)
        self.assertTrue(found.samefile(DATA_DIR / "test_big"))

    def test_plain_identifiers_are_not_taken_for_a_path(self):
        """A list of protein symbols is not a file, and must not raise"""
        self.assertIsNone(resolve_input_path("TP53 BRCA1 ATM"))

    def test_a_missing_file_raises_instead_of_being_read_as_proteins(self):
        """A path that matches nothing is an error, not a protein identifier

        Without this, a mistyped path would be sent to STRING as if it were a
        gene name, and the analysis would silently run on the wrong network.
        """
        with self.assertRaises(FileNotFoundError):
            resolve_input_path("./does_not_exist.txt")

        with self.assertRaises(FileNotFoundError):
            resolve_input_path("some/missing/folder/genes.txt")

    def test_wrong_input_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_input_path(42)


class TestReadIdentifiersFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.genes_file = Path(self.temp_dir.name) / "my_genes.txt"
        self.genes_file.write_text("TP53\n  BRCA1  \n\nATM\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_identifiers_are_read_stripped_and_without_empty_lines(self):
        """One identifier per line, with the blanks dropped"""
        self.assertEqual(read_identifiers_file(self.genes_file), ["TP53", "BRCA1", "ATM"])

    def test_the_example_file_of_the_project_is_read(self):
        """The example input of the project holds the expected proteins"""
        identifiers = read_identifiers_file(DATA_DIR / "test_big")

        self.assertIn("TP53", identifiers)
        self.assertTrue(all(identifier for identifier in identifiers))


if __name__ == "__main__":
    unittest.main()
