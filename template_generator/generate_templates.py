#!/usr/bin/env python
"""Build executable templates from ChemDraw-friendly [R] reactions.

The ``expand`` stage creates an inspectable ``preprocessed_data.csv``. After
curation, the ``extract`` stage atom-maps those reactions and writes the final
template and condition JSON files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
import sys
from typing import Sequence

try:
    from rdkit import Chem
except ModuleNotFoundError as exc:
    raise SystemExit(
        "RDKit is required. Run: pip install -r requirements.txt"
    ) from exc


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
EXTRACTOR_DIR = PROJECT_DIR / "template"
if str(EXTRACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACTOR_DIR))

try:
    from get_template import template_extractor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Template extractor not found at template/get_template/template_extractor.py."
    ) from exc


PLACEHOLDER_PATTERN = re.compile(r"\[R([A-Za-z0-9_]*)\]")
DEFAULT_DOMAINS = ["methyl", "primary", "secondary", "tertiary", "aryl"]
DEFAULT_ATOM_SOURCES = {"O": "O"}

DOMAIN_ALIASES = {
    "methyl": "methyl",
    "me": "methyl",
    "甲基": "methyl",
    "primary": "primary",
    "1": "primary",
    "1degree": "primary",
    "一级": "primary",
    "一级碳": "primary",
    "secondary": "secondary",
    "2": "secondary",
    "2degree": "secondary",
    "二级": "secondary",
    "二级碳": "secondary",
    "tertiary": "tertiary",
    "3": "tertiary",
    "3degree": "tertiary",
    "三级": "tertiary",
    "三级碳": "tertiary",
    "aryl": "aryl",
    "aromatic": "aryl",
    "ar": "aryl",
    "芳基": "aryl",
    "芳环": "aryl",
}


@dataclass(frozen=True)
class FragmentDefinition:
    smiles: str
    attachment_atom: int = 0


FRAGMENTS = {
    "methyl": FragmentDefinition("[CH3]"),
    "primary": FragmentDefinition("[CH2]C"),
    "secondary": FragmentDefinition("[CH](C)C"),
    "tertiary": FragmentDefinition("[C](C)(C)C"),
    "aryl": FragmentDefinition("c1ccccc1"),
}


class GenerationError(ValueError):
    """An invalid input row or generic reaction."""


def split_reaction(reaction: str) -> tuple[str, str]:
    if reaction.count(">>") != 1:
        raise GenerationError("Reaction must contain exactly one '>>'.")
    reactants, products = (part.strip() for part in reaction.split(">>"))
    if not reactants or not products:
        raise GenerationError("Both sides of the reaction must be non-empty.")
    return reactants, products


def placeholder_names(reaction: str) -> list[str]:
    names: list[str] = []
    for suffix in PLACEHOLDER_PATTERN.findall(reaction):
        name = f"R{suffix}" if suffix else "R"
        if name not in names:
            names.append(name)
    return names


def canonicalize_domain(value: str) -> str:
    key = value.strip().lower().replace("°", "degree").replace(" ", "")
    if key not in DOMAIN_ALIASES:
        choices = ", ".join(FRAGMENTS)
        raise GenerationError(
            f"Unknown R-group class '{value}'. Choose from: {choices}."
        )
    return DOMAIN_ALIASES[key]


def normalize_domains(
    placeholders: Sequence[str], raw_domains: dict[str, list[str]]
) -> dict[str, list[str]]:
    unknown = sorted(set(raw_domains) - set(placeholders))
    if unknown:
        raise GenerationError(
            f"R-group classes were supplied for absent placeholders: {unknown}."
        )

    domains: dict[str, list[str]] = {}
    for name in placeholders:
        values = raw_domains.get(name, DEFAULT_DOMAINS)
        normalized: list[str] = []
        for value in values:
            domain = canonicalize_domain(value)
            if domain not in normalized:
                normalized.append(domain)
        if not normalized:
            raise GenerationError(f"Placeholder {name} has no allowed classes.")
        domains[name] = normalized
    return domains


def _replace_placeholders(side: str, dummy_maps: dict[str, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        suffix = match.group(1)
        name = f"R{suffix}" if suffix else "R"
        return f"[*:{dummy_maps[name]}]"

    return PLACEHOLDER_PATTERN.sub(replace, side)


def _graft_fragment(
    molecule: Chem.Mol,
    dummy_map: int,
    fragment: FragmentDefinition,
) -> Chem.Mol:
    dummy_indices = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == dummy_map
    ]
    if not dummy_indices:
        return molecule
    if len(dummy_indices) > 1:
        raise GenerationError(
            "The same placeholder may occur at most once on each reaction side. "
            "Use [R1], [R2], ... for independent positions."
        )

    dummy_idx = dummy_indices[0]
    dummy = molecule.GetAtomWithIdx(dummy_idx)
    if dummy.GetDegree() > 1:
        raise GenerationError("Each [R] placeholder must have one attachment at most.")

    fragment_mol = Chem.MolFromSmiles(fragment.smiles)
    if fragment_mol is None:
        raise RuntimeError(f"Invalid internal fragment: {fragment.smiles}")
    combined = Chem.CombineMols(molecule, fragment_mol)
    editable = Chem.RWMol(combined)
    offset = molecule.GetNumAtoms()
    if dummy.GetDegree() == 1:
        bond = dummy.GetBonds()[0]
        neighbour = bond.GetOtherAtomIdx(dummy_idx)
        editable.AddBond(
            int(neighbour),
            int(offset + fragment.attachment_atom),
            bond.GetBondType(),
        )
    editable.RemoveAtom(dummy_idx)
    expanded = editable.GetMol()
    Chem.SanitizeMol(expanded)
    return expanded


def expand_side(
    side: str,
    selection: dict[str, str],
    dummy_maps: dict[str, int],
) -> str:
    molecule = Chem.MolFromSmiles(_replace_placeholders(side, dummy_maps))
    if molecule is None:
        raise GenerationError(f"RDKit could not parse this reaction side: {side}")
    for name, domain in selection.items():
        molecule = _graft_fragment(molecule, dummy_maps[name], FRAGMENTS[domain])
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise GenerationError("An unrecognized wildcard remains after R-group expansion.")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    normalized = Chem.MolFromSmiles(smiles)
    if normalized is None:
        raise GenerationError(f"RDKit could not normalize expanded SMILES: {smiles}")
    return Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True)


def _heavy_element_counts(smiles: str) -> Counter[str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise GenerationError(f"RDKit could not parse expanded SMILES: {smiles}")
    return Counter(
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1
    )


def _parse_atom_sources(value: str) -> dict[str, str]:
    value = value.strip()
    if not value:
        return {}
    if "=" not in value:
        return {"*": value}

    sources: dict[str, str] = {}
    for assignment in re.split(r"[;|]", value):
        assignment = assignment.strip()
        if not assignment:
            continue
        if "=" not in assignment:
            raise GenerationError(
                "Multiple atom sources must use ELEMENT=SMILES, e.g. O=O;N=N."
            )
        element, smiles = (part.strip() for part in assignment.split("=", 1))
        if not re.fullmatch(r"[A-Z][a-z]?", element) or not smiles:
            raise GenerationError(f"Invalid atom_source assignment: {assignment}")
        if Chem.MolFromSmiles(smiles) is None:
            raise GenerationError(f"Invalid atom_source SMILES: {smiles}")
        sources[element] = smiles
    return sources


def add_atom_sources(
    reactants: str,
    products: str,
    atom_sources: dict[str, str],
) -> str:
    """Add explicit sources for product atoms absent from the generic reactants."""
    reactant_counts = _heavy_element_counts(reactants)
    product_counts = _heavy_element_counts(products)
    additions: list[str] = []

    while True:
        deficits = {
            element: count - reactant_counts.get(element, 0)
            for element, count in product_counts.items()
            if count > reactant_counts.get(element, 0)
        }
        if not deficits:
            break
        element = sorted(deficits)[0]
        if element in atom_sources:
            source = atom_sources[element]
        elif "*" in atom_sources:
            source = atom_sources["*"]
        elif element in DEFAULT_ATOM_SOURCES:
            source = DEFAULT_ATOM_SOURCES[element]
        else:
            raise GenerationError(
                f"Product has {deficits[element]} additional {element} atom(s). "
                f"Set atom_source to {element}=<SMILES>."
            )
        source_counts = _heavy_element_counts(source)
        if source_counts.get(element, 0) == 0:
            raise GenerationError(
                f"Atom source '{source}' does not contain the required {element}."
            )
        additions.append(source)
        reactant_counts.update(source_counts)

    return ".".join([reactants, *additions])


def _split_choices(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,|]", value) if part.strip()]


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GenerationError("CSV has no header row.")
        headers = [str(name).strip() for name in reader.fieldnames]
        required = {"reaction_id", "reaction_name", "reaction"}
        missing = sorted(required - set(headers))
        if missing:
            raise GenerationError(f"CSV is missing required columns: {missing}")

        records: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for row_number, raw in enumerate(reader, start=2):
            row = {
                str(key).strip(): ("" if value is None else str(value).strip())
                for key, value in raw.items()
            }
            if not any(row.values()):
                continue
            reaction_id = row.get("reaction_id", "")
            reaction_name = row.get("reaction_name", "")
            reaction = row.get("reaction", "")
            if not reaction_id or not reaction_name or not reaction:
                raise GenerationError(
                    f"CSV row {row_number} requires reaction_id, reaction_name, "
                    "and reaction."
                )
            if reaction_id in seen_ids:
                raise GenerationError(f"Duplicate reaction_id: {reaction_id}")
            seen_ids.add(reaction_id)

            placeholders = placeholder_names(reaction)
            if not placeholders:
                raise GenerationError(
                    f"{reaction_id}: reaction must contain [R], [R1], ..."
                )
            domains: dict[str, list[str]] = {}
            for column, value in row.items():
                if not value or re.fullmatch(r"R[1-9][0-9]*", column) is None:
                    continue
                placeholder = column
                if column == "R1" and "R" in placeholders and "R1" not in placeholders:
                    placeholder = "R"
                domains[placeholder] = _split_choices(value)

            records.append(
                {
                    "reaction_id": reaction_id,
                    "reaction_name": reaction_name,
                    "reaction": reaction.replace(" ", ""),
                    "domains": domains,
                    "atom_sources": _parse_atom_sources(row.get("atom_source", "")),
                    "condition": row.get("condition", ""),
                    "source": row.get("source", ""),
                    "chapter": row.get("chapter", ""),
                }
            )
    if not records:
        raise GenerationError("CSV contains no reaction rows.")
    return records


def expand_record(
    record: dict[str, object], max_combinations: int
) -> list[dict[str, str]]:
    generic_reaction = str(record["reaction"])
    reactant_side, product_side = split_reaction(generic_reaction)
    placeholders = placeholder_names(generic_reaction)
    domains = normalize_domains(placeholders, record["domains"])  # type: ignore[arg-type]
    combinations = list(itertools.product(*(domains[name] for name in placeholders)))
    if len(combinations) > max_combinations:
        raise GenerationError(
            f"{record['reaction_id']} expands to {len(combinations)} combinations; "
            f"the limit is {max_combinations}."
        )

    dummy_maps = {name: 8000 + index for index, name in enumerate(placeholders)}
    rows: list[dict[str, str]] = []
    for index, values in enumerate(combinations, start=1):
        selection = dict(zip(placeholders, values))
        reactants = expand_side(reactant_side, selection, dummy_maps)
        products = expand_side(product_side, selection, dummy_maps)
        reactants = add_atom_sources(
            reactants,
            products,
            record["atom_sources"],  # type: ignore[arg-type]
        )
        rows.append(
            {
                "_id": f"{record['reaction_id']}-{index:03d}",
                "reaction_id": str(record["reaction_id"]),
                "reaction_name": str(record["reaction_name"]),
                "r_group_selection": ";".join(
                    f"{name}={selection[name]}" for name in placeholders
                ),
                "reactant": reactants,
                "product": products,
                "condition": str(record["condition"]),
                "source": str(record["source"]),
                "chapter": str(record["chapter"]),
            }
        )
    return rows


def write_preprocessed_data(rows: Sequence[dict[str, str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "_id",
        "reaction_id",
        "reaction_name",
        "r_group_selection",
        "reactant",
        "product",
        "condition",
        "source",
        "chapter",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _extract_one(
    reaction: dict[str, str], radius: int, timeout_seconds: int = 20
) -> dict[str, str]:
    from func_timeout import func_timeout

    try:
        result = func_timeout(
            timeout_seconds,
            template_extractor.extract_from_reaction,
            args=(reaction, radius),
        )
    except Exception:
        return {"reaction_id": reaction["_id"]}
    return result or {"reaction_id": reaction["_id"]}


def _reaction_smarts(result: dict[str, str]) -> str:
    value = result.get("reaction_smarts", "")
    return value if isinstance(value, str) else ""


def _usable_template(value: object) -> bool:
    if not isinstance(value, str) or not value or ">>" not in value:
        return False
    return "." not in value.split(">>", 1)[0]


def extract_templates(
    input_path: Path,
    template_output: Path,
    condition_output: Path,
    add_radius_0: bool,
    jobs: int,
) -> dict[str, int]:
    try:
        import pandas as pd
        from joblib import Parallel, delayed
        from rxnmapper import BatchedMapper
    except ModuleNotFoundError as exc:
        raise GenerationError(
            f"Missing extraction dependency '{exc.name}'. Run: pip install -r requirements.txt"
        ) from exc

    data = pd.read_csv(input_path, encoding="utf-8-sig", keep_default_na=False)
    required = ["_id", "reactant", "product", "condition"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise GenerationError(f"Preprocessed CSV is missing columns: {missing}")
    if data.empty:
        raise GenerationError("Preprocessed CSV contains no reactions.")

    for column in required:
        data[column] = data[column].astype(str)
    if any(not value.strip() for value in data["_id"]):
        raise GenerationError("Every row requires a non-empty _id.")
    if data["_id"].duplicated().any():
        raise GenerationError("Every _id must be unique.")

    reactions = [
        f"{reactant}>>{product}"
        for reactant, product in zip(data["reactant"], data["product"])
    ]
    mapper = BatchedMapper(batch_size=20)
    mapped_reactions = list(mapper.map_reactions(reactions))
    if len(mapped_reactions) != len(data):
        raise GenerationError("Atom mapper returned a different number of reactions.")

    data["reaction"] = mapped_reactions
    split_mapped = data["reaction"].str.split(">>", n=1, expand=True)
    if split_mapped.shape[1] != 2:
        raise GenerationError("Atom mapper returned an invalid reaction.")
    data["reactants"] = split_mapped[0]
    data["products"] = split_mapped[1]
    extraction_input = data[["_id", "reactants", "products"]].to_dict("records")

    templates_r1 = Parallel(n_jobs=jobs, verbose=4)(
        delayed(_extract_one)(reaction, 1) for reaction in extraction_input
    )
    data["template_r1"] = [_reaction_smarts(result) for result in templates_r1]

    if add_radius_0:
        templates_r0 = Parallel(n_jobs=jobs, verbose=4)(
            delayed(_extract_one)(reaction, 0) for reaction in extraction_input
        )
        data["template_r0"] = [_reaction_smarts(result) for result in templates_r0]
    else:
        data["template_r0"] = ""

    data.to_csv(input_path, index=False, encoding="utf-8-sig")

    templates = [value for value in data["template_r1"] if _usable_template(value)]
    if add_radius_0:
        templates.extend(
            value for value in data["template_r0"] if _usable_template(value)
        )

    condition_map: dict[str, list[str]] = {}
    for _, row in data.iterrows():
        row_templates = [row["template_r1"]]
        if add_radius_0:
            row_templates.append(row["template_r0"])
        for template in row_templates:
            if not _usable_template(template):
                continue
            conditions = condition_map.setdefault(template, [])
            condition = str(row["condition"]).strip()
            if condition and condition not in conditions:
                conditions.append(condition)

    template_output.parent.mkdir(parents=True, exist_ok=True)
    condition_output.parent.mkdir(parents=True, exist_ok=True)
    template_output.write_text(
        json.dumps(templates, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    condition_output.write_text(
        json.dumps(condition_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "reaction_count": len(data),
        "template_instance_count": len(templates),
        "unique_template_count": len(set(templates)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand [R] reactions and extract executable templates."
    )
    commands = parser.add_subparsers(dest="command")

    expand = commands.add_parser("expand", help="Create a CSV for human checking.")
    expand.add_argument("--input", type=Path, required=True, help="Generic reaction CSV.")
    expand.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR / "output" / "preprocessed_data.csv",
        help="Output preprocessed_data.csv path.",
    )
    expand.add_argument(
        "--max-combinations",
        type=int,
        default=200,
        help="Maximum expanded rows per generic reaction.",
    )

    extract = commands.add_parser(
        "extract", help="Map checked reactions and export templates."
    )
    extract.add_argument("--input", type=Path, required=True, help="Checked CSV path.")
    extract.add_argument(
        "--template-output",
        type=Path,
        default=PROJECT_DIR / "reaction_template.json",
    )
    extract.add_argument(
        "--condition-output",
        type=Path,
        default=PROJECT_DIR / "template_condition.json",
    )
    extract.add_argument(
        "--add-radius-0",
        action="store_true",
        help="Include radius-0 templates in addition to radius-1 templates.",
    )
    extract.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel extraction jobs; default uses all available cores.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "expand":
            if args.max_combinations < 1:
                raise GenerationError("--max-combinations must be positive.")
            records = load_records(args.input)
            rows = [
                row
                for record in records
                for row in expand_record(record, args.max_combinations)
            ]
            output_path = write_preprocessed_data(rows, args.output.resolve())
            print(f"Input reactions: {len(records)}")
            print(f"Expanded reactions: {len(rows)}")
            print(f"Check this file before extraction: {output_path}")
        elif args.command == "extract":
            summary = extract_templates(
                args.input.resolve(),
                args.template_output.resolve(),
                args.condition_output.resolve(),
                args.add_radius_0,
                args.jobs,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Templates: {args.template_output.resolve()}")
            print(f"Conditions: {args.condition_output.resolve()}")
        else:
            raise GenerationError("Choose either the 'expand' or 'extract' command.")
        return 0
    except (GenerationError, FileNotFoundError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
