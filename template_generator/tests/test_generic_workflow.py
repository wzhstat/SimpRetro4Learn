"""Lightweight contract tests for the reviewable generic-reaction workflow."""

from pathlib import Path
import re
import sys
import unittest

from rdkit import Chem


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

import generate_templates as generator  # noqa: E402


class GenericWorkflowTests(unittest.TestCase):
    def test_unmapped_whole_r_reaction_expands_to_concrete_auto_map_input(self):
        record = {
            "reaction_id": "2-test",
            "reaction_name": "Substitution",
            "reaction": "[R1]O[R2].[XH]>>[R2][X]",
            "domains": {
                "R1": ["secondary"],
                "R2": ["tertiary"],
                "X": ["Br"],
            },
            "atom_sources": {},
            "condition": "",
            "source": "",
            "chapter": "2",
            "mapping_mode": "auto",
            "allowed_variants": "",
        }

        rows = generator.expand_record(record, max_combinations=10)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["mapping_mode"], "auto")
        self.assertEqual(
            row["r_group_selection"],
            "R1=secondary;R2=tertiary;X=Br",
        )
        self.assertEqual(row["reactant"], "Br.CC(C)OC(C)(C)C")
        self.assertEqual(row["product"], "CC(C)(C)Br")
        self.assertNotRegex(row["reactant"] + row["product"], r"\[(?:R|X)")
        self.assertNotRegex(row["reactant"] + row["product"], r":\d+")
        for side in (row["reactant"], row["product"]):
            for smiles in side.split("."):
                self.assertIsNotNone(Chem.MolFromSmiles(smiles))

    def test_reverse_abstraction_includes_single_side_organic_group(self):
        template = (
            "[Br;H0;D1;+0:5]-[C;H0;D4;+0:1](-[C;D1;H3:2])"
            "(-[C;D1;H3:3])-[C;D1;H3:4]>>C-C(-C)-O-"
            "[C;H0;D4;+0:1](-[C;D1;H3:2])(-[C;D1;H3:3])-"
            "[C;D1;H3:4].[BrH;D0;+0:5]"
        )

        reaction, domains = generator._generic_whole_r_forward_from_template(
            template
        )

        self.assertEqual(reaction, "[R1]O[R2].[XH]>>[R2][X]")
        self.assertEqual(
            domains,
            {"R1": "secondary", "R2": "tertiary", "X": "Br"},
        )

    def test_fixed_h_slots_become_implicit_and_surviving_r_is_renumbered(self):
        group = {
            "reaction": "BrC([R1])([R2])[R3]>>OC([R1])([R2])[R3]",
            "domains": {"R1": ["methyl"], "R2": ["H"], "R3": ["H"]},
            "variants": [
                {"R1": "methyl", "R2": "H", "R3": "H"}
            ],
        }

        removed = generator._reverse_elide_fixed_h_group(group)

        self.assertEqual(removed, 2)
        self.assertEqual(generator.placeholder_names(group["reaction"]), ["R"])
        self.assertEqual(group["domains"], {"R": ["methyl"]})
        self.assertEqual(group["variants"], [{"R": "methyl"}])
        self.assertNotIn("[H]", group["reaction"])

    def test_h_carbon_mixed_domain_remains_a_real_r_choice(self):
        group = {
            "reaction": "Br[R]>>O[R]",
            "domains": {"R": ["H", "methyl"]},
            "variants": [{"R": "H"}, {"R": "methyl"}],
        }

        removed = generator._reverse_elide_fixed_h_group(group)

        self.assertEqual(removed, 0)
        self.assertEqual(group["reaction"], "Br[R]>>O[R]")

    def test_site_qualified_r_groups_share_a_domain_but_choose_independently(self):
        record = {
            "reaction_id": "site-test",
            "reaction_name": "Site-qualified example",
            "reaction": "C([R1a])([R1b])Br>>C([R1a])([R1b])O",
            "domains": {"R1": ["H", "methyl"]},
            "atom_sources": {},
            "condition": "",
            "source": "",
            "chapter": "2",
            "mapping_mode": "auto",
            "allowed_variants": "R1a=H,R1b=methyl | R1a=methyl,R1b=H",
            "template_derived": True,
        }

        rows = generator.expand_record(record, max_combinations=10)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["r_group_selection"] for row in rows},
            {
                "R1a=H;R1b=methyl",
                "R1a=methyl;R1b=H",
            },
        )
        self.assertTrue(all(row["reactant"] == "CCBr" for row in rows))
        self.assertTrue(all(row["product"] == "CCO" for row in rows))

    def test_grignard_coupling_uses_each_rooted_alkyl_as_one_r_group(self):
        secondary_secondary = (
            "[C;D1;H3:5]-[CH;D3;+0:4](-[CH3;D1;+0:2])-"
            "[CH;D3;+0:6](-[CH3;D1;+0:3])-[CH3;D1;+0:1]>>"
            "Cl-[Mg]-[CH;D3;+0:1](-[CH3;D1;+0:2])-[CH3;D1;+0:3]."
            "F-[CH;D3;+0:4](-[C;D1;H3:5])-[CH3;D1;+0:6]"
        )
        primary_secondary = (
            "[C;D1;H3:4]-[CH;D3;+0:3](-[C;D1;H3:5])-"
            "[CH2;D2;+0:1]-[C;D1;H3:2]>>"
            "Cl-[Mg]-[CH2;D2;+0:1]-[C;D1;H3:2]."
            "I-[CH;D3;+0:3](-[C;D1;H3:4])-[C;D1;H3:5]"
        )

        first = generator._generic_grignard_halide_coupling(
            secondary_secondary
        )
        second = generator._generic_grignard_halide_coupling(
            primary_secondary
        )

        self.assertEqual(
            first,
            (
                "[R1][Mg][X1].[R2][X2]>>[R1][R2]",
                {"R1": "secondary", "X1": "Cl", "R2": "secondary", "X2": "F"},
            ),
        )
        self.assertEqual(
            second,
            (
                "[R1][Mg][X1].[R2][X2]>>[R1][R2]",
                {"R1": "primary", "X1": "Cl", "R2": "secondary", "X2": "I"},
            ),
        )

    def test_h_is_an_explicit_supported_r_domain(self):
        self.assertIn("H", generator.FRAGMENTS)
        self.assertEqual(generator.canonicalize_r_domain("hydrogen"), "H")

    def test_reverse_qc_rejects_a_template_that_cannot_self_replay(self):
        invalid_template = (
            "[CH3;D1;+0:1]>>Cl-[Mg]-[CH3;D1;+0:1]"
        )

        replayed, reason = generator._reverse_self_replay(
            invalid_template, "CC"
        )

        self.assertFalse(replayed)
        self.assertIn("no precursors", reason)

    def test_reverse_exclusions_filename_tracks_the_main_output(self):
        self.assertEqual(
            generator._reverse_exclusions_output_path(
                Path("template_derived_reactions.csv")
            ),
            Path("template_derived_exclusions.csv"),
        )
        self.assertEqual(
            generator._reverse_exclusions_output_path(Path("review.csv")),
            Path("review_exclusions.csv"),
        )

    def test_all_x_placeholders_expand_before_rdkit_processing(self):
        record = {
            "reaction_id": "x-test",
            "reaction_name": "Halogen example",
            "reaction": "[R][X]>>[R]O",
            "domains": {"R": ["methyl"], "X": ["Cl", "Br"]},
            "atom_sources": {},
            "condition": "",
            "source": "",
            "chapter": "2",
            "mapping_mode": "auto",
            "allowed_variants": "",
        }

        rows = generator.expand_record(record, max_combinations=10)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["r_group_selection"] for row in rows},
            {"R=methyl;X=Cl", "R=methyl;X=Br"},
        )
        self.assertTrue(
            all(re.search(r"Cl|Br", row["reactant"]) for row in rows)
        )
        self.assertTrue(
            all("[X" not in row["reactant"] + row["product"] for row in rows)
        )


if __name__ == "__main__":
    unittest.main()
