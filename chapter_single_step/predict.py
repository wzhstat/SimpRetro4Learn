#!/usr/bin/env python
"""Single-step retrosynthesis with explicitly selected course chapters."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from rdchiral.initialization import rdchiralReactants, rdchiralReaction
    from rdchiral.main import rdchiralRun
    from rdkit import Chem
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing chemistry dependency: {exc.name}. Install requirements.txt first."
    ) from exc


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = PACKAGE_DIR / "data" / "library_1270_chapter_templates.json"
DEFAULT_STOCK = PACKAGE_DIR / "data" / "emol_under_4_carbons.txt"
AVAILABLE_CHAPTERS = list(range(1, 15))
DEFAULT_WEIGHTS = (0.1, 0.2, 0.5, 0.0)


def parse_chapters(value: str) -> tuple[int, ...]:
    """Parse a comma-separated chapter list into a sorted, unique tuple."""
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "chapters must be a comma-separated list, for example: 1,2,3"
        )
    try:
        chapters = tuple(sorted({int(part.strip()) for part in parts}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "chapters must contain integers separated by commas, for example: 5,6,7"
        ) from exc
    unsupported = [chapter for chapter in chapters if chapter not in AVAILABLE_CHAPTERS]
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported chapter(s): {', '.join(map(str, unsupported))}; "
            f"choose chapters from {AVAILABLE_CHAPTERS[0]} to {AVAILABLE_CHAPTERS[-1]}"
        )
    return chapters


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True)


def load_inventory(path: Path) -> set[str]:
    inventory = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        canonical = canonical_smiles(line.strip())
        if canonical:
            inventory.add(canonical)
    return inventory


def mapped_atom_count(smiles: str) -> int:
    return len([int(value) for value in re.findall(r":(\d+)", smiles) if int(value) < 900])


def complexity_score(product_mol: Chem.Mol, raw_reactants: Sequence[str]) -> float:
    if len(raw_reactants) <= 1:
        return 0.0
    product_atoms = product_mol.GetNumAtoms()
    counts = [mapped_atom_count(smiles) for smiles in raw_reactants]
    if not counts:
        return 0.0
    main_index = max(range(len(counts)), key=counts.__getitem__)
    main_mol = Chem.MolFromSmiles(raw_reactants[main_index])
    if main_mol is None or main_mol.GetNumAtoms() >= product_atoms:
        return 0.0
    mean = product_atoms / len(counts)
    mean_absolute_error = sum(abs(mean - count) for count in counts) / len(counts)
    return product_atoms / (1.0 + mean_absolute_error)


def availability_score(
    product_mol: Chem.Mol,
    raw_to_canonical: dict[str, str | None],
    inventory: set[str],
) -> float:
    product_atoms = product_mol.GetNumAtoms()
    raw_reactants = list(raw_to_canonical)
    if not raw_reactants:
        return 0.0
    counts = [mapped_atom_count(smiles) for smiles in raw_reactants]
    main_index = max(range(len(counts)), key=counts.__getitem__)
    main_mol = Chem.MolFromSmiles(raw_reactants[main_index])
    main_atoms = main_mol.GetNumAtoms() if main_mol else product_atoms
    score = 0.0
    for raw, canonical in raw_to_canonical.items():
        if canonical in inventory:
            addition = mapped_atom_count(raw)
            score += addition if main_atoms < product_atoms or addition > 2 else 0
        if canonical and any(metal in canonical for metal in ("Mg", "Li", "Zn")):
            if canonical not in inventory:
                score -= 5.0
    return score


def ring_disconnection_score(product_mol: Chem.Mol, reactant_mols: Sequence[Chem.Mol]) -> float:
    product_rings = product_mol.GetRingInfo().NumRings()
    reactant_rings = 0
    for mol in reactant_mols:
        if mol is None:
            continue
        for ring in mol.GetRingInfo().AtomRings():
            atoms = [mol.GetAtomWithIdx(index) for index in ring]
            if any(atom.GetSymbol() in {"B", "Si"} for atom in atoms):
                continue
            map_numbers = [atom.GetAtomMapNum() for atom in atoms]
            if map_numbers and min(map_numbers) < 900:
                reactant_rings += 1
    return 1.0 if product_rings > reactant_rings else 0.0


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class CurriculumSingleStepPredictor:
    """Execute library-1270 templates assigned to explicitly selected chapters."""

    def __init__(
        self,
        chapters: Sequence[int] | int,
        dataset_path: Path | str = DEFAULT_DATASET,
        stock_path: Path | str = DEFAULT_STOCK,
        weights: Sequence[float] = DEFAULT_WEIGHTS,
    ) -> None:
        requested_chapters = (chapters,) if isinstance(chapters, int) else tuple(chapters)
        if not requested_chapters:
            raise ValueError("at least one chapter must be selected")
        unsupported = sorted(
            {chapter for chapter in requested_chapters if chapter not in AVAILABLE_CHAPTERS}
        )
        if unsupported:
            raise ValueError(
                f"Unsupported chapter(s) {unsupported}; choose chapters from "
                f"{AVAILABLE_CHAPTERS[0]} to {AVAILABLE_CHAPTERS[-1]}"
            )
        if len(weights) != 4:
            raise ValueError("weights must contain four values: CD AS RD SS")
        self.chapters = tuple(sorted(set(requested_chapters)))
        self.dataset_path = Path(dataset_path)
        self.stock_path = Path(stock_path)
        self.weights = tuple(float(value) for value in weights)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Template dataset not found: {self.dataset_path}")
        if not self.stock_path.exists():
            raise FileNotFoundError(f"Starting-material inventory not found: {self.stock_path}")

        self.dataset = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if self.dataset.get("library_version") != "library_1270":
            raise ValueError("This package requires the library_1270 dataset")
        self.template_records = [
            record
            for record in self.dataset["templates"]
            if int(record["intro_chapter"]) in self.chapters
        ]
        self.inventory = load_inventory(self.stock_path)
        self.compiled_templates: list[tuple[dict[str, Any], Any]] = []
        self.compile_error_count = 0
        for record in self.template_records:
            try:
                self.compiled_templates.append((record, rdchiralReaction(record["template"])))
            except Exception:
                self.compile_error_count += 1

    def predict(self, product: str, top_k: int = 10) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        product_canonical = canonical_smiles(product)
        if product_canonical is None:
            raise ValueError(f"Invalid product SMILES: {product}")
        product_mol = Chem.MolFromSmiles(product_canonical)
        rd_product = rdchiralReactants(product_canonical)
        w1, w2, w3, w4 = self.weights
        candidates: dict[tuple[str, ...], dict[str, Any]] = {}
        execution_errors = 0
        matching_templates = 0
        invalid_results = 0
        start = time.perf_counter()

        for record, compiled in self.compiled_templates:
            try:
                mapped_results = rdchiralRun(compiled, rd_product, keep_mapnums=True)
            except Exception:
                execution_errors += 1
                continue
            if not mapped_results:
                continue
            matching_templates += 1
            specificity = 1.0 / len(mapped_results)
            for raw_result in mapped_results:
                raw_parts = [part for part in raw_result.split(".") if part]
                canonical_parts = [canonical_smiles(part) for part in raw_parts]
                if not canonical_parts or any(part is None for part in canonical_parts):
                    invalid_results += 1
                    continue
                precursor_tuple = tuple(sorted(str(part) for part in canonical_parts))
                if not precursor_tuple or product_canonical in precursor_tuple:
                    continue
                raw_to_canonical = dict(zip(raw_parts, canonical_parts))
                reactant_mols = [Chem.MolFromSmiles(part) for part in raw_parts]
                components = {
                    "complexity_disconnection": complexity_score(product_mol, raw_parts),
                    "starting_material_availability": availability_score(
                        product_mol, raw_to_canonical, self.inventory
                    ),
                    "ring_disconnection": ring_disconnection_score(product_mol, reactant_mols),
                    "template_specificity": specificity,
                }
                score = (
                    w1 * components["complexity_disconnection"]
                    + w2 * components["starting_material_availability"]
                    + w3 * components["ring_disconnection"]
                    + w4 * components["template_specificity"]
                )
                evidence = {
                    "template_hash": record["template_hash"],
                    "template_smarts": record["template"],
                    "intro_chapter": int(record["intro_chapter"]),
                    "instance_count": int(record["instance_count"]),
                    "reaction_family": record.get("reaction_family", ""),
                    "core_reaction_ids": record.get("core_reaction_ids", ""),
                    "conditions": record.get("conditions", ""),
                    "score": float(score),
                    "score_components": components,
                }
                candidate = candidates.setdefault(
                    precursor_tuple,
                    {
                        "reactants": list(precursor_tuple),
                        "reactants_smiles": ".".join(precursor_tuple),
                        "best_score": float(score),
                        "best_score_components": components,
                        "first_available_chapter": int(record["intro_chapter"]),
                        "evidence": [],
                    },
                )
                candidate["first_available_chapter"] = min(
                    candidate["first_available_chapter"], int(record["intro_chapter"])
                )
                known_hashes = {item["template_hash"] for item in candidate["evidence"]}
                if evidence["template_hash"] not in known_hashes:
                    candidate["evidence"].append(evidence)
                if score > candidate["best_score"]:
                    candidate["best_score"] = float(score)
                    candidate["best_score_components"] = components

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item["best_score"], item["reactants_smiles"]),
        )[:top_k]
        probabilities = softmax([item["best_score"] for item in ordered])
        for rank, (candidate, probability) in enumerate(zip(ordered, probabilities), start=1):
            candidate["rank"] = rank
            candidate["policy_probability"] = float(probability)
            candidate["template_support_count"] = len(candidate["evidence"])
            candidate["reaction_families"] = sorted(
                {
                    item["reaction_family"]
                    for item in candidate["evidence"]
                    if item["reaction_family"]
                }
            )
            candidate["evidence"].sort(
                key=lambda item: (-item["score"], item["template_hash"])
            )

        return {
            "product_input": product,
            "product_canonical": product_canonical,
            "requested_chapters": list(self.chapters),
            "snapshot_definition": (
                "library-1270 templates assigned only to the explicitly requested chapters"
            ),
            "library_version": "library_1270",
            "stock_file": self.stock_path.name,
            "weights": list(self.weights),
            "top_k": top_k,
            "template_inventory": {
                "available_unique_templates": len(self.template_records),
                "unique_templates_by_requested_chapter": {
                    str(chapter): sum(
                        int(record["intro_chapter"]) == chapter
                        for record in self.template_records
                    )
                    for chapter in self.chapters
                },
                "compiled_templates": len(self.compiled_templates),
                "template_compile_errors": self.compile_error_count,
            },
            "execution_stats": {
                "matching_templates": matching_templates,
                "template_execution_errors": execution_errors,
                "invalid_generated_results": invalid_results,
                "unique_predictions_before_top_k": len(candidates),
                "returned_predictions": len(ordered),
                "elapsed_seconds": time.perf_counter() - start,
            },
            "predictions": ordered,
        }


def print_predictions(result: dict[str, Any]) -> None:
    print("\nSingle-step retrosynthesis (library 1270)")
    print(f"Product: {result['product_canonical']}")
    chapters = ", ".join(map(str, result["requested_chapters"]))
    print(f"Chapters: {chapters} (explicit selection)")
    print(
        f"Templates: {result['template_inventory']['available_unique_templates']}; "
        f"unique predictions: {result['execution_stats']['unique_predictions_before_top_k']}"
    )
    if not result["predictions"]:
        print("No valid single-step precursor prediction was found.")
        return
    print()
    for candidate in result["predictions"]:
        families = ", ".join(candidate["reaction_families"]) or "unclassified"
        print(f"[{candidate['rank']}] {candidate['reactants_smiles']}")
        print(
            f"    score={candidate['best_score']:.4f}; "
            f"policy={candidate['policy_probability']:.4f}; "
            f"first_chapter={candidate['first_available_chapter']}; "
            f"family={families}; templates={candidate['template_support_count']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Library-1270 single-step retrosynthesis using templates from explicitly "
            "selected course chapters."
        )
    )
    parser.add_argument("-p", "--product", help="Product SMILES. Prompted if omitted.")
    parser.add_argument(
        "-c", "--chapters", type=parse_chapters, metavar="CHAPTERS",
        help="Comma-separated chapters to use, for example 1,2,3 or 5,6,7.",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("-o", "--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stock", type=Path, default=DEFAULT_STOCK)
    parser.add_argument(
        "--weights", nargs=4, type=float, metavar=("CD", "AS", "RD", "SS"),
        default=DEFAULT_WEIGHTS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        product = args.product or input("Product SMILES: ").strip()
        chapters = args.chapters
        if chapters is None:
            chapters = parse_chapters(
                input(
                    f"Chapters ({AVAILABLE_CHAPTERS[0]}-{AVAILABLE_CHAPTERS[-1]}, "
                    "comma-separated): "
                ).strip()
            )
        predictor = CurriculumSingleStepPredictor(
            chapters=chapters,
            dataset_path=args.dataset,
            stock_path=args.stock,
            weights=args.weights,
        )
        result = predictor.predict(product, top_k=args.top_k)
        print_predictions(result)
        if args.output:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nJSON written to: {output}")
        return 0
    except (argparse.ArgumentTypeError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
