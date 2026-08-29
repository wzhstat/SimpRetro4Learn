import json
import sys
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from predict import CurriculumSingleStepPredictor  # noqa: E402


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

    def test_known_chapter_7_prediction(self):
        predictor = CurriculumSingleStepPredictor(chapter=7)
        result = predictor.predict("CCOC(=O)CC", top_k=3)
        precursor_sets = {row["reactants_smiles"] for row in result["predictions"]}
        self.assertIn("CCC(=O)O.CCO", precursor_sets)
        self.assertEqual(result["template_inventory"]["available_unique_templates"], 565)
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
        predictor = CurriculumSingleStepPredictor(chapter=5)
        result = predictor.predict("CCC(O)C(O)CC", top_k=5)
        precursor_sets = {row["reactants_smiles"] for row in result["predictions"]}
        self.assertIn("CCC=CCC.O=[Mn](=O)(=O)[O-]", precursor_sets)
        self.assertIn("CCC=CCC.O=[Os](=O)(=O)=O", precursor_sets)
        self.assertEqual(result["template_inventory"]["available_unique_templates"], 495)


if __name__ == "__main__":
    unittest.main()
