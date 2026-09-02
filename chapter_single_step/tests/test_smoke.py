import csv
import json
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from predict import CurriculumSingleStepPredictor, parse_chapters  # noqa: E402


class DatasetTests(unittest.TestCase):
    def test_dataset_counts(self):
        dataset = json.loads(
            (PACKAGE_DIR / "data" / "library_1270_chapter_templates.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(dataset["template_instances"], 1272)
        self.assertEqual(dataset["unique_templates"], 927)
        self.assertEqual(len(dataset["templates"]), 927)
        original_reactions = json.loads(
            (PACKAGE_DIR.parent / "reaction_template.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(original_reactions), 1272)
        self.assertEqual(
            sum(row["instance_count"] for row in dataset["templates"]),
            len(original_reactions),
        )
        self.assertEqual(len({row["template_hash"] for row in dataset["templates"]}), 927)
        self.assertEqual(sum(dataset["new_unique_templates_by_chapter"].values()), 927)
        self.assertNotIn("mapping_mode", dataset)
        self.assertNotIn("mapping_warning", dataset)
        forbidden = {
            "mapping_confidence",
            "mapping_status",
            "center_cluster_id",
            "candidate_core_reaction_ids",
            "candidate_core_reaction_names",
        }
        self.assertTrue(all(forbidden.isdisjoint(row) for row in dataset["templates"]))
        self.assertTrue(all("core_reaction_ids" in row for row in dataset["templates"]))

        with (PACKAGE_DIR / "data" / "chapter_reaction_template_summary.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(sum(int(row["reactions"]) for row in summary), 317)
        self.assertEqual(sum(int(row["templates"]) for row in summary), 1272)
        self.assertEqual(sum(int(row["unique_templates"]) for row in summary), 927)

    def test_explicit_chapter_parser(self):
        self.assertEqual(parse_chapters("1,2,3"), (1, 2, 3))
        self.assertEqual(parse_chapters("7, 5,6,5"), (5, 6, 7))

    def test_known_chapters_5_6_7_prediction(self):
        predictor = CurriculumSingleStepPredictor(chapters=[5, 6, 7])
        result = predictor.predict("CCOC(=O)CC", top_k=3)
        precursor_sets = {row["reactants_smiles"] for row in result["predictions"]}
        self.assertIn("CCC(=O)O.CCO", precursor_sets)
        self.assertEqual(result["requested_chapters"], [5, 6, 7])
        self.assertEqual(result["template_inventory"]["available_unique_templates"], 378)
        self.assertEqual(
            result["template_inventory"]["unique_templates_by_requested_chapter"],
            {"5": 308, "6": 31, "7": 39},
        )
        self.assertNotIn("mapping_mode", result)
        self.assertNotIn("mapping_warning", result)
        self.assertTrue(
            all(
                "mapping_status" not in evidence and "mapping_confidence" not in evidence
                for prediction in result["predictions"]
                for evidence in prediction["evidence"]
            )
        )

    def test_chapter_5_dihydroxylation_templates(self):
        predictor = CurriculumSingleStepPredictor(chapters=[5])
        result = predictor.predict("CCC(O)C(O)CC", top_k=5)
        precursor_sets = {row["reactants_smiles"] for row in result["predictions"]}
        self.assertIn("CCC=CCC.O=[Mn](=O)(=O)[O-]", precursor_sets)
        self.assertIn("CCC=CCC.O=[Os](=O)(=O)=O", precursor_sets)
        self.assertEqual(result["template_inventory"]["available_unique_templates"], 308)

    def test_chapters_without_templates_are_accepted(self):
        predictor = CurriculumSingleStepPredictor(chapters=[1, 2, 3])
        result = predictor.predict("CCOC(=O)CC", top_k=1)
        self.assertEqual(result["requested_chapters"], [1, 2, 3])
        self.assertEqual(result["template_inventory"]["available_unique_templates"], 22)
        self.assertEqual(
            result["template_inventory"]["unique_templates_by_requested_chapter"],
            {"1": 0, "2": 22, "3": 0},
        )


if __name__ == "__main__":
    unittest.main()
