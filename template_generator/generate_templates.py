#!/usr/bin/env python
"""Build executable templates from ChemDraw-friendly [R]/[X] reactions.

The ``reverse`` stage makes an unmapped review draft directly from a template
library. The ``expand`` stage creates an inspectable ``preprocessed_data.csv``.
After curation, ``extract`` maps ordinary rows, preserves verified mapped rows,
and writes the template and condition JSON files.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import itertools
import json
import networkx as nx
from pathlib import Path
import re
import sys
from typing import Sequence

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdChemReactions
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


R_PLACEHOLDER_PATTERN = re.compile(
    r"\[R(?P<rgroup>[1-9][0-9]*)?(?P<rsite>[a-z])?"
    r"(?::(?P<rmap>[1-9][0-9]*))?\]"
)
X_PLACEHOLDER_PATTERN = re.compile(
    r"\[X([1-9][0-9]*)?([Hh+\-0-9]*)(?::([1-9][0-9]*))?\]"
)
PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:(?P<rkind>R)(?P<rsuffix>[1-9][0-9]*)?"
    r"(?P<rsite>[a-z])?"
    r"(?::(?P<rmap>[1-9][0-9]*))?|"
    r"(?P<xkind>X)(?P<xsuffix>[1-9][0-9]*)?"
    r"(?P<xdecorations>[Hh+\-0-9]*)(?::(?P<xmap>[1-9][0-9]*))?)\]"
)
DEFAULT_R_DOMAINS = [
    "methyl",
    "primary",
    "secondary",
    "tertiary",
    "aryl",
    "alkenyl",
    "alkynyl",
    "benzyl",
    "secondary_benzylic",
    "allyl",
    "secondary_allylic",
    "propargyl",
    "cycloalkyl",
]
DEFAULT_X_DOMAINS = ["F", "Cl", "Br", "I"]
DEFAULT_ATOM_SOURCES = {"O": "O"}
HALOGEN_ATOMIC_NUMBERS = {9: "F", 17: "Cl", 35: "Br", 53: "I"}
HALOGEN_SYMBOLS = frozenset(HALOGEN_ATOMIC_NUMBERS.values())
ATOM_TOKEN_PATTERN = re.compile(r"\[[^\[\]]+\]|Br|Cl|F(?![a-z])|I(?![a-z])")

DOMAIN_ALIASES = {
    "h": "H",
    "hydrogen": "H",
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
    "alkenyl": "alkenyl",
    "vinyl": "alkenyl",
    "alkynyl": "alkynyl",
    "ethynyl": "alkynyl",
    "benzyl": "benzyl",
    "secondary_benzylic": "secondary_benzylic",
    "secondarybenzyl": "secondary_benzylic",
    "secondarybenzylic": "secondary_benzylic",
    "allyl": "allyl",
    "secondary_allylic": "secondary_allylic",
    "secondaryallyl": "secondary_allylic",
    "secondaryallylic": "secondary_allylic",
    "propargyl": "propargyl",
    "cycloalkyl": "cycloalkyl",
    "methoxycarbonyl": "methoxycarbonyl",
    "carbomethoxy": "methoxycarbonyl",
    "co2me": "methoxycarbonyl",
}


@dataclass(frozen=True)
class FragmentDefinition:
    smiles: str
    attachment_atom: int = 0


FRAGMENTS = {
    "H": FragmentDefinition("[H]"),
    # Implicit valence lets a reverse-generated whole-R block occur alone on
    # one side and gain/lose its single external bond on the other side.
    "methyl": FragmentDefinition("C"),
    "primary": FragmentDefinition("CC"),
    "secondary": FragmentDefinition("C(C)C"),
    "tertiary": FragmentDefinition("C(C)(C)C"),
    "aryl": FragmentDefinition("c1ccccc1"),
    "alkenyl": FragmentDefinition("C=C"),
    "alkynyl": FragmentDefinition("C#C"),
    "benzyl": FragmentDefinition("Cc1ccccc1"),
    "secondary_benzylic": FragmentDefinition("C(C)c1ccccc1"),
    "allyl": FragmentDefinition("CC=C"),
    "secondary_allylic": FragmentDefinition("C(C)C=C"),
    "propargyl": FragmentDefinition("CC#C"),
    "cycloalkyl": FragmentDefinition("C1CCCCC1"),
    # Carbonyl-attached ester substituent, e.g. the electron-withdrawing group
    # in methyl acrylate used in Diels-Alder examples.
    "methoxycarbonyl": FragmentDefinition("C(=O)OC"),
}


class GenerationError(ValueError):
    """An invalid input row or generic reaction."""


def _r_name(group: str | None, site: str | None = None) -> str:
    return f"R{group or ''}{site or ''}"


def _r_domain_name(name: str) -> str:
    """Return the shared domain label for a possibly site-qualified R name."""
    match = re.fullmatch(r"R([1-9][0-9]*)?([a-z])?", name)
    if match is None:
        raise GenerationError(f"Invalid R placeholder name: {name}")
    return f"R{match.group(1) or ''}"


def split_reaction(reaction: str) -> tuple[str, str]:
    if reaction.count(">>") != 1:
        raise GenerationError("Reaction must contain exactly one '>>'.")
    reactants, products = (part.strip() for part in reaction.split(">>"))
    if not reactants or not products:
        raise GenerationError("Both sides of the reaction must be non-empty.")
    return reactants, products


def placeholder_names(reaction: str) -> list[str]:
    names: list[str] = []
    for match in PLACEHOLDER_PATTERN.finditer(reaction):
        kind = "R" if match.group("rkind") else "X"
        suffix = match.group("rsuffix") if kind == "R" else match.group("xsuffix")
        suffix = suffix or ""
        site = match.group("rsite") if kind == "R" else ""
        name = f"{kind}{suffix}{site or ''}" if suffix or site else kind
        if name not in names:
            names.append(name)
    return names


def canonicalize_r_domain(value: str) -> str:
    key = value.strip().lower().replace("°", "degree").replace(" ", "")
    if key not in DOMAIN_ALIASES:
        choices = ", ".join(FRAGMENTS)
        raise GenerationError(
            f"Unknown R-group class '{value}'. Choose from: {choices}."
        )
    return DOMAIN_ALIASES[key]


def canonicalize_x_domain(value: str) -> str:
    aliases = {"f": "F", "cl": "Cl", "br": "Br", "i": "I"}
    key = value.strip().lower().replace(" ", "")
    if key not in aliases:
        raise GenerationError(
            f"Unknown halogen '{value}'. Choose from: {', '.join(DEFAULT_X_DOMAINS)}."
        )
    return aliases[key]


def normalize_domains(
    placeholders: Sequence[str], raw_domains: dict[str, list[str]]
) -> dict[str, list[str]]:
    domain_names = {
        _r_domain_name(name) if name.startswith("R") else name
        for name in placeholders
    }
    unknown = sorted(set(raw_domains) - set(placeholders) - domain_names)
    if unknown:
        raise GenerationError(
            f"Domains were supplied for absent placeholders: {unknown}."
        )

    domains: dict[str, list[str]] = {}
    for name in placeholders:
        is_halogen = name.startswith("X")
        domain_name = _r_domain_name(name) if not is_halogen else name
        values = raw_domains.get(
            name,
            raw_domains.get(
                domain_name,
                DEFAULT_X_DOMAINS if is_halogen else DEFAULT_R_DOMAINS,
            ),
        )
        normalized: list[str] = []
        for value in values:
            domain = (
                canonicalize_x_domain(value)
                if is_halogen
                else canonicalize_r_domain(value)
            )
            if domain not in normalized:
                normalized.append(domain)
        if not normalized:
            raise GenerationError(f"Placeholder {name} has no allowed classes.")
        domains[name] = normalized
    return domains


def _replace_r_placeholders(side: str, dummy_maps: dict[str, int]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = _r_name(match.group("rgroup"), match.group("rsite"))
        return f"[*:{dummy_maps[name]}]"

    return R_PLACEHOLDER_PATTERN.sub(replace, side)


def _r_placeholder_maps(reaction: str) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for match in R_PLACEHOLDER_PATTERN.finditer(reaction):
        name = _r_name(match.group("rgroup"), match.group("rsite"))
        atom_map = int(match.group("rmap")) if match.group("rmap") else None
        if name in result and result[name] != atom_map:
            raise GenerationError(
                f"Placeholder [{name}] must use the same atom map on both sides."
            )
        result[name] = atom_map
    return result


def _replace_x_placeholders(side: str, selection: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        suffix = match.group(1) or ""
        decorations = match.group(2) or ""
        atom_map = match.group(3)
        element = selection[f"X{suffix}"]
        map_suffix = f":{atom_map}" if atom_map else ""
        if decorations or atom_map:
            return f"[{element}{decorations}{map_suffix}]"
        return element

    return X_PLACEHOLDER_PATTERN.sub(replace, side)


def _graft_fragment(
    molecule: Chem.Mol,
    dummy_map: int,
    fragment: FragmentDefinition,
    fragment_atom_maps: Sequence[int] | None = None,
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

    # Avoid trying to remove a standalone explicit H before it has been
    # attached to the reaction scaffold.
    fragment_mol = Chem.MolFromSmiles(fragment.smiles, sanitize=False)
    if fragment_mol is None:
        raise RuntimeError(f"Invalid internal fragment: {fragment.smiles}")
    Chem.SanitizeMol(fragment_mol)
    if fragment_atom_maps is not None:
        if len(fragment_atom_maps) < fragment_mol.GetNumAtoms():
            raise GenerationError("Not enough atom maps were reserved for an R group.")
        for atom, atom_map in zip(fragment_mol.GetAtoms(), fragment_atom_maps):
            atom.SetAtomMapNum(int(atom_map))
    combined = Chem.CombineMols(molecule, fragment_mol)
    editable = Chem.RWMol(combined)
    offset = molecule.GetNumAtoms()
    if dummy.GetDegree() == 1:
        bond = dummy.GetBonds()[0]
        if fragment.smiles == "[H]" and bond.GetBondType() != Chem.BondType.SINGLE:
            raise GenerationError("R=H is only valid at a single-bond attachment.")
        neighbour = bond.GetOtherAtomIdx(dummy_idx)
        editable.AddBond(
            int(neighbour),
            int(offset + fragment.attachment_atom),
            bond.GetBondType(),
        )
    editable.RemoveAtom(dummy_idx)
    expanded = editable.GetMol()
    try:
        Chem.SanitizeMol(expanded)
        # Use explicit H while grafting, then return ordinary canonical SMILES
        # for atom mapping and template extraction.
        expanded = Chem.RemoveHs(expanded)
        Chem.SanitizeMol(expanded)
    except Exception as exc:
        raise GenerationError(
            f"Could not attach fragment '{fragment.smiles}' to [R]."
        ) from exc
    return expanded


def expand_side(
    side: str,
    selection: dict[str, str],
    dummy_maps: dict[str, int],
    preserve_atom_maps: bool = False,
    fragment_atom_maps: dict[str, Sequence[int]] | None = None,
) -> str:
    concrete_side = _replace_x_placeholders(side, selection)
    molecule = Chem.MolFromSmiles(_replace_r_placeholders(concrete_side, dummy_maps))
    if molecule is None:
        raise GenerationError(f"RDKit could not parse this reaction side: {side}")
    for name, domain in selection.items():
        if not name.startswith("R"):
            continue
        maps = None if fragment_atom_maps is None else fragment_atom_maps.get(name)
        molecule = _graft_fragment(
            molecule, dummy_maps[name], FRAGMENTS[domain], maps
        )
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        raise GenerationError("An unrecognized wildcard remains after R-group expansion.")
    if not preserve_atom_maps:
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


def _valid_preserved_mapping(reactants: str, products: str) -> bool:
    reactant_mol = Chem.MolFromSmiles(reactants)
    product_mol = Chem.MolFromSmiles(products)
    if reactant_mol is None or product_mol is None:
        return False
    reactant_maps = [
        atom.GetAtomMapNum()
        for atom in reactant_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    ]
    product_maps = [atom.GetAtomMapNum() for atom in product_mol.GetAtoms()]
    if not reactant_maps or not product_maps:
        return False
    if any(value <= 0 for value in product_maps):
        return False
    if len(set(reactant_maps)) != len(reactant_maps):
        return False
    if len(set(product_maps)) != len(product_maps):
        return False
    return True


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


def _allowed_variant_selections(
    value: str,
    placeholders: Sequence[str],
    domains: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Expand correlated placeholder choices without inventing cross-combinations.

    ``allowed_variants`` uses a compact, reviewable syntax such as
    ``R1=H,R2=methyl | R1=methyl,R2=H``.  A placeholder omitted from one
    alternative still uses its ordinary domain column; reverse-generated rows
    use this to keep R-group correlations while allowing X to remain generic.
    """
    fixed_alternatives: list[dict[str, str]] = []
    if value.lstrip().startswith("["):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GenerationError("allowed_variants contains invalid JSON.") from exc
        if not isinstance(decoded, list) or not all(
            isinstance(item, dict) for item in decoded
        ):
            raise GenerationError(
                "JSON allowed_variants must be a list of NAME/VALUE objects."
            )
        fixed_alternatives = [
            {str(name).strip(): str(choice).strip() for name, choice in item.items()}
            for item in decoded
        ]
    else:
        alternatives = [part.strip() for part in value.split("|") if part.strip()]
        for alternative in alternatives:
            fixed: dict[str, str] = {}
            for assignment in alternative.split(","):
                assignment = assignment.strip()
                if not assignment:
                    continue
                if "=" not in assignment:
                    raise GenerationError(
                        "allowed_variants assignments must use NAME=VALUE."
                    )
                name, raw_value = (
                    part.strip() for part in assignment.split("=", 1)
                )
                if name in fixed:
                    raise GenerationError(
                        f"allowed_variants assigns placeholder '{name}' twice."
                    )
                fixed[name] = raw_value
            fixed_alternatives.append(fixed)
    if not fixed_alternatives:
        raise GenerationError("allowed_variants contains no alternatives.")

    selections: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for raw_fixed in fixed_alternatives:
        fixed: dict[str, str] = {}
        for name, raw_value in raw_fixed.items():
            if name not in placeholders:
                raise GenerationError(
                    f"allowed_variants refers to absent placeholder '{name}'."
                )
            normalized = (
                canonicalize_x_domain(raw_value)
                if name.startswith("X")
                else canonicalize_r_domain(raw_value)
            )
            if normalized not in domains[name]:
                raise GenerationError(
                    f"allowed_variants value {name}={normalized} is outside "
                    f"the declared domain {domains[name]}."
                )
            fixed[name] = normalized

        variable_names = [name for name in placeholders if name not in fixed]
        for values in itertools.product(*(domains[name] for name in variable_names)):
            selection = dict(fixed)
            selection.update(zip(variable_names, values))
            key = tuple((name, selection[name]) for name in placeholders)
            if key not in seen:
                seen.add(key)
                selections.append(selection)
    return selections


def _query_integer(atom: Chem.Atom, query_name: str) -> int | None:
    match = re.search(rf"{re.escape(query_name)} (-?\d+) = val", atom.DescribeQuery())
    return int(match.group(1)) if match else None


def _query_is_aromatic(atom: Chem.Atom) -> bool:
    return atom.GetIsAromatic() or "AtomIsAromatic 1 = val" in atom.DescribeQuery()


def _atom_from_query(source: Chem.Atom) -> Chem.Atom:
    atomic_number = source.GetAtomicNum()
    if not atomic_number:
        raise GenerationError(
            f"Cannot materialize wildcard atom '{source.GetSmarts()}'."
        )
    atom = Chem.Atom(atomic_number)
    formal_charge = _query_integer(source, "AtomFormalCharge")
    atom.SetFormalCharge(
        source.GetFormalCharge() if formal_charge is None else formal_charge
    )
    atom.SetIsotope(source.GetIsotope())
    atom.SetIsAromatic(_query_is_aromatic(source))
    atom.SetAtomMapNum(source.GetAtomMapNum())
    atom.SetChiralTag(source.GetChiralTag())
    hydrogen_count = _query_integer(source, "AtomHCount")
    if hydrogen_count is not None:
        atom.SetNumExplicitHs(hydrogen_count)
        atom.SetNoImplicit(True)
    elif _query_is_aromatic(source) and source.GetAtomicNum() != 6:
        atom.SetNoImplicit(True)
    return atom


def _aromatic_components(query: Chem.Mol) -> list[list[int]]:
    remaining = {
        atom.GetIdx() for atom in query.GetAtoms() if _query_is_aromatic(atom)
    }
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        component = [start]
        stack = [start]
        while stack:
            atom_index = stack.pop()
            for neighbour in query.GetAtomWithIdx(atom_index).GetNeighbors():
                neighbour_index = neighbour.GetIdx()
                bond = query.GetBondBetweenAtoms(atom_index, neighbour_index)
                if (
                    neighbour_index in remaining
                    and bond.GetBondType() == Chem.BondType.AROMATIC
                ):
                    remaining.remove(neighbour_index)
                    component.append(neighbour_index)
                    stack.append(neighbour_index)
        components.append(component)
    return components


def _generated_atom_map(
    registry: dict[tuple[object, ...], int], key: tuple[object, ...]
) -> int:
    if key not in registry:
        registry[key] = 50000 + len(registry)
    return registry[key]


def _materialize_query(
    query: Chem.Mol, registry: dict[tuple[object, ...], int]
) -> Chem.Mol:
    """Create a mapped witness molecule for a radius-limited SMARTS query."""
    editable = Chem.RWMol()
    old_to_new = {
        source.GetIdx(): editable.AddAtom(_atom_from_query(source))
        for source in query.GetAtoms()
    }

    for source in query.GetBonds():
        begin = old_to_new[source.GetBeginAtomIdx()]
        end = old_to_new[source.GetEndAtomIdx()]
        editable.AddBond(begin, end, source.GetBondType())
        bond = editable.GetBondBetweenAtoms(begin, end)
        bond.SetBondDir(source.GetBondDir())

    for component in _aromatic_components(query):
        component_set = set(component)
        edge_count = sum(
            1
            for bond in query.GetBonds()
            if bond.GetBeginAtomIdx() in component_set
            and bond.GetEndAtomIdx() in component_set
        )
        if edge_count >= len(component):
            continue
        aromatic_degrees = {
            atom_index: sum(
                1
                for neighbour in query.GetAtomWithIdx(atom_index).GetNeighbors()
                if neighbour.GetIdx() in component_set
                and query.GetBondBetweenAtoms(
                    atom_index, neighbour.GetIdx()
                ).GetBondType()
                == Chem.BondType.AROMATIC
            )
            for atom_index in component
        }
        endpoints = [
            atom_index
            for atom_index, degree in aromatic_degrees.items()
            if degree <= 1
        ]
        if len(component) == 1:
            ordered = component
            endpoints = [component[0], component[0]]
        else:
            if len(endpoints) != 2:
                raise GenerationError(
                    "Cannot complete a branched, radius-truncated aromatic query."
                )
            start = min(
                endpoints,
                key=lambda atom_index: query.GetAtomWithIdx(
                    atom_index
                ).GetAtomMapNum(),
            )
            ordered = [start]
            previous, current = -1, start
            while len(ordered) < len(component):
                candidates = [
                    neighbour.GetIdx()
                    for neighbour in query.GetAtomWithIdx(current).GetNeighbors()
                    if neighbour.GetIdx() in component_set
                    and neighbour.GetIdx() != previous
                ]
                if not candidates:
                    break
                previous, current = current, candidates[0]
                ordered.append(current)
            endpoints = [ordered[0], ordered[-1]]

        signature = tuple(
            query.GetAtomWithIdx(atom_index).GetAtomMapNum()
            for atom_index in ordered
        )
        ring_size = (
            5
            if any(
                query.GetAtomWithIdx(atom_index).GetAtomicNum() in {8, 16}
                for atom_index in component
            )
            else 6
        )
        previous = old_to_new[endpoints[0]]
        for position in range(ring_size - len(component)):
            carbon = Chem.Atom(6)
            carbon.SetIsAromatic(True)
            carbon.SetAtomMapNum(
                _generated_atom_map(
                    registry, ("aromatic", signature, position)
                )
            )
            new_index = editable.AddAtom(carbon)
            editable.AddBond(previous, new_index, Chem.BondType.AROMATIC)
            previous = new_index
        editable.AddBond(
            previous, old_to_new[endpoints[1]], Chem.BondType.AROMATIC
        )

    for old_index, new_index in old_to_new.items():
        query_atom = query.GetAtomWithIdx(old_index)
        atom = editable.GetAtomWithIdx(new_index)
        if not query_atom.GetAtomMapNum():
            continue
        explicit_degree = _query_integer(query_atom, "AtomExplicitDegree")
        target_degree = (
            explicit_degree
            if explicit_degree is not None
            else atom.GetDegree()
            if _query_is_aromatic(query_atom)
            else 2
        )
        while atom.GetDegree() < target_degree:
            carbon = Chem.Atom(6)
            carbon.SetAtomMapNum(
                _generated_atom_map(
                    registry,
                    ("boundary", query_atom.GetAtomMapNum(), atom.GetDegree()),
                )
            )
            carbon_index = editable.AddAtom(carbon)
            editable.AddBond(new_index, carbon_index, Chem.BondType.SINGLE)

    materialized = editable.GetMol()
    try:
        Chem.SanitizeMol(materialized)
    except Exception as exc:
        raise GenerationError("Could not materialize a template SMARTS query.") from exc
    return materialized


def _query_smarts_to_smiles(
    smarts: str, registry: dict[tuple[object, ...], int]
) -> str:
    components: list[str] = []
    for component in smarts.split("."):
        query = Chem.MolFromSmarts(component)
        if query is None:
            raise GenerationError(f"RDKit could not parse template SMARTS: {component}")
        molecule = _materialize_query(query, registry)
        smiles = Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        if Chem.MolFromSmiles(smiles) is None:
            raise GenerationError(
                f"RDKit could not parse reverse-generated SMILES: {smiles}"
            )
        components.append(smiles)
    return ".".join(components)


def _halogen_symbol_from_token(token: str) -> str | None:
    if not token.startswith("["):
        return token if token in HALOGEN_SYMBOLS else None
    body = token[1:-1]
    atomic_number = re.match(r"^#(9|17|35|53)(?=$|[;,:+\-&!])", body)
    if atomic_number:
        return HALOGEN_ATOMIC_NUMBERS[int(atomic_number.group(1))]
    element = re.match(
        r"^(?:\d+)?(Cl|Br|F|I)(?=$|[;,:+\-@HhD&!])", body
    )
    return element.group(1) if element else None


def _replace_halogen_token(token: str, placeholder: str) -> str:
    if not token.startswith("["):
        return f"[{placeholder}]"
    body = token[1:-1]
    atomic_number = re.match(r"^#(?:9|17|35|53)", body)
    if atomic_number:
        return f"[{placeholder}{body[atomic_number.end():]}]"
    element = re.match(r"^(?:\d+)?(?:Cl|Br|F|I)", body)
    if element:
        return f"[{placeholder}{body[element.end():]}]"
    return token


def _halogen_placeholders(text: str) -> tuple[str, dict[str, str]]:
    symbols: list[str] = []
    for match in ATOM_TOKEN_PATTERN.finditer(text):
        symbol = _halogen_symbol_from_token(match.group(0))
        if symbol is not None and symbol not in symbols:
            symbols.append(symbol)
    if not symbols:
        return text, {}
    if len(symbols) == 1:
        element_to_placeholder = {symbols[0]: "X"}
    else:
        element_to_placeholder = {
            symbol: f"X{index}" for index, symbol in enumerate(symbols, start=1)
        }

    def replace(match: re.Match[str]) -> str:
        symbol = _halogen_symbol_from_token(match.group(0))
        if symbol is None:
            return match.group(0)
        return _replace_halogen_token(
            match.group(0), element_to_placeholder[symbol]
        )

    abstracted = ATOM_TOKEN_PATTERN.sub(replace, text)
    return abstracted, element_to_placeholder


def _replace_halogen_elements(
    text: str, element_to_placeholder: dict[str, str]
) -> str:
    def replace(match: re.Match[str]) -> str:
        symbol = _halogen_symbol_from_token(match.group(0))
        if symbol is None or symbol not in element_to_placeholder:
            return match.group(0)
        return _replace_halogen_token(
            match.group(0), element_to_placeholder[symbol]
        )

    return ATOM_TOKEN_PATTERN.sub(replace, text)


def _forward_reaction_from_template(
    template: str, element_to_placeholder: dict[str, str]
) -> str:
    product_query, precursor_query = split_reaction(template)
    registry: dict[tuple[object, ...], int] = {}
    product = _query_smarts_to_smiles(product_query, registry)
    precursors = _query_smarts_to_smiles(precursor_query, registry)
    forward = f"{precursors}>>{product}"
    return _replace_halogen_elements(forward, element_to_placeholder)


def _reverse_atom_signature(atom: Chem.Atom) -> tuple[object, ...]:
    return (
        atom.GetAtomicNum(),
        bool(_query_is_aromatic(atom)),
        _query_integer(atom, "AtomFormalCharge"),
        _query_integer(atom, "AtomHCount"),
        _query_integer(atom, "AtomExplicitDegree"),
    )


def _reverse_query_side(side: str) -> Chem.Mol:
    molecule = Chem.MolFromSmarts(side)
    if molecule is None:
        raise GenerationError(f"RDKit could not parse template SMARTS: {side}")
    return molecule


def _reverse_atoms_by_map(molecule: Chem.Mol) -> dict[int, Chem.Atom]:
    return {
        atom.GetAtomMapNum(): atom
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def _reverse_bonds_by_map(
    molecule: Chem.Mol,
) -> dict[tuple[int, int], tuple[str, bool]]:
    bonds: dict[tuple[int, int], tuple[str, bool]] = {}
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtom().GetAtomMapNum()
        end = bond.GetEndAtom().GetAtomMapNum()
        if begin <= 0 or end <= 0:
            continue
        bonds[tuple(sorted((begin, end)))] = (
            str(bond.GetBondType()),
            bool(bond.GetIsAromatic()),
        )
    return bonds


def _reverse_changed_maps(
    template: str,
) -> tuple[set[int], Chem.Mol, Chem.Mol]:
    product_smarts, precursor_smarts = split_reaction(template)
    product = _reverse_query_side(product_smarts)
    precursor = _reverse_query_side(precursor_smarts)
    product_atoms = _reverse_atoms_by_map(product)
    precursor_atoms = _reverse_atoms_by_map(precursor)
    product_bonds = _reverse_bonds_by_map(product)
    precursor_bonds = _reverse_bonds_by_map(precursor)

    changed = set(product_atoms) ^ set(precursor_atoms)
    for atom_map in set(product_atoms) & set(precursor_atoms):
        if _reverse_atom_signature(product_atoms[atom_map]) != _reverse_atom_signature(
            precursor_atoms[atom_map]
        ):
            changed.add(atom_map)
    for key in set(product_bonds) | set(precursor_bonds):
        if product_bonds.get(key) != precursor_bonds.get(key):
            changed.update(key)
    return changed, product, precursor


def _reverse_h_slots(
    product_query: Chem.Mol,
    precursor_query: Chem.Mol,
    changed: set[int],
) -> list[int]:
    product_atoms = _reverse_atoms_by_map(product_query)
    precursor_atoms = _reverse_atoms_by_map(precursor_query)
    slots: list[int] = []
    for atom_map in sorted(changed & set(product_atoms) & set(precursor_atoms)):
        if product_atoms[atom_map].GetAtomicNum() != 6:
            continue
        product_h = _query_integer(product_atoms[atom_map], "AtomHCount")
        precursor_h = _query_integer(precursor_atoms[atom_map], "AtomHCount")
        if product_h is None or precursor_h is None:
            continue
        slots.extend([atom_map] * min(product_h, precursor_h))
    return slots


@dataclass(frozen=True)
class _ReverseTransferBlock:
    """A carbon block represented by one whole-R placeholder."""

    maps: frozenset[int]
    root_map: int
    domain: str


def _reverse_whole_bond_key(
    bond: Chem.Bond,
) -> tuple[str, bool, str]:
    return (
        str(bond.GetBondType()),
        bool(bond.GetIsAromatic()),
        str(bond.GetStereo()),
    )


def _reverse_common_carbon_graph(
    product: Chem.Mol, precursor: Chem.Mol
) -> nx.Graph:
    """Return carbon atoms/bonds conserved by the remapped witness."""
    product_atoms = _reverse_atoms_by_map(product)
    precursor_atoms = _reverse_atoms_by_map(precursor)
    common = {
        atom_map
        for atom_map in set(product_atoms) & set(precursor_atoms)
        if product_atoms[atom_map].GetAtomicNum() == 6
        and precursor_atoms[atom_map].GetAtomicNum() == 6
        and product_atoms[atom_map].GetFormalCharge()
        == precursor_atoms[atom_map].GetFormalCharge()
        and product_atoms[atom_map].GetIsAromatic()
        == precursor_atoms[atom_map].GetIsAromatic()
    }
    graph = nx.Graph()
    graph.add_nodes_from(common)
    for product_bond in product.GetBonds():
        first = product_bond.GetBeginAtom().GetAtomMapNum()
        second = product_bond.GetEndAtom().GetAtomMapNum()
        if first not in common or second not in common:
            continue
        precursor_bond = precursor.GetBondBetweenAtoms(
            precursor_atoms[first].GetIdx(), precursor_atoms[second].GetIdx()
        )
        if (
            precursor_bond is not None
            and _reverse_whole_bond_key(product_bond)
            == _reverse_whole_bond_key(precursor_bond)
        ):
            graph.add_edge(first, second)
    return graph


def _reverse_block_boundary(
    molecule: Chem.Mol, atom_maps: set[int]
) -> list[tuple[int, int, tuple[str, bool, str]]]:
    boundary: list[tuple[int, int, tuple[str, bool, str]]] = []
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_map = begin.GetAtomMapNum()
        end_map = end.GetAtomMapNum()
        if (begin_map in atom_maps) == (end_map in atom_maps):
            continue
        root, outside = (begin, end) if begin_map in atom_maps else (end, begin)
        boundary.append(
            (
                root.GetAtomMapNum(),
                outside.GetIdx(),
                _reverse_whole_bond_key(bond),
            )
        )
    return boundary


def _reverse_whole_r_domain(
    molecule: Chem.Mol, atom_maps: set[int], root_map: int
) -> str:
    """Classify a rooted conserved block by its local organic environment."""
    atoms = _reverse_atoms_by_map(molecule)
    root = atoms[root_map]
    if root.GetIsAromatic():
        return "aryl"
    if root.IsInRing():
        return "cycloalkyl"

    inside_bonds = [
        bond
        for bond in root.GetBonds()
        if bond.GetOtherAtom(root).GetAtomMapNum() in atom_maps
    ]
    if any(bond.GetBondType() == Chem.BondType.TRIPLE for bond in inside_bonds):
        return "alkynyl"
    if any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in inside_bonds):
        return "alkenyl"

    carbon_neighbours = sum(
        bond.GetOtherAtom(root).GetAtomicNum() == 6 for bond in inside_bonds
    )
    inside_neighbours = [bond.GetOtherAtom(root) for bond in inside_bonds]
    if any(neighbour.GetIsAromatic() for neighbour in inside_neighbours):
        return "secondary_benzylic" if carbon_neighbours == 2 else "benzyl"

    adjacent_unsaturation = {
        bond.GetBondType()
        for neighbour in inside_neighbours
        for bond in neighbour.GetBonds()
        if bond.GetOtherAtom(neighbour).GetIdx() != root.GetIdx()
        and bond.GetOtherAtom(neighbour).GetAtomMapNum() in atom_maps
        and bond.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
    }
    if Chem.BondType.TRIPLE in adjacent_unsaturation:
        # No secondary propargylic block occurs in the current library; fall
        # back to the substitution class if one is encountered in new data.
        return "propargyl" if carbon_neighbours == 1 else "secondary"
    if Chem.BondType.DOUBLE in adjacent_unsaturation:
        return "secondary_allylic" if carbon_neighbours == 2 else "allyl"

    return {0: "methyl", 1: "primary", 2: "secondary"}.get(
        carbon_neighbours, "tertiary"
    )


def _reverse_transfer_blocks(
    product: Chem.Mol, precursor: Chem.Mol
) -> list[_ReverseTransferBlock]:
    """Find maximal conserved carbon blocks with at most one port per side.

    A block may have one external connection on both sides, or zero on one
    side and one on the other.  The latter represents an R-H terminus that is
    bonded or debonded.  A completely disconnected 0/0 spectator is skipped.
    """
    graph = _reverse_common_carbon_graph(product, precursor)
    blocks: list[_ReverseTransferBlock] = []

    def visit(nodes: set[int]) -> None:
        if not nodes:
            return
        product_boundary = _reverse_block_boundary(product, nodes)
        precursor_boundary = _reverse_block_boundary(precursor, nodes)
        roots = [
            edges[0][0]
            for edges in (product_boundary, precursor_boundary)
            if len(edges) == 1
        ]
        if (
            len(product_boundary) <= 1
            and len(precursor_boundary) <= 1
            and len(product_boundary) + len(precursor_boundary) >= 1
            and len(set(roots)) == 1
        ):
            root_map = roots[0]
            blocks.append(
                _ReverseTransferBlock(
                    maps=frozenset(nodes),
                    root_map=root_map,
                    domain=_reverse_whole_r_domain(product, nodes, root_map),
                )
            )
            return

        # A multi-port reaction centre cannot itself be one R group.  Remove
        # its interface atoms and retry each remaining pendant carbon block.
        interface_maps = {
            root for root, _, _ in product_boundary + precursor_boundary
        }
        if not interface_maps:
            return
        remaining = nodes - interface_maps
        if not remaining:
            return
        for component in nx.connected_components(graph.subgraph(remaining)):
            visit(set(component))

    for component in nx.connected_components(graph):
        visit(set(component))
    return sorted(blocks, key=lambda block: (block.root_map, sorted(block.maps)))


def _reverse_assign_side_local_maps(
    precursor: Chem.Mol, product: Chem.Mol
) -> None:
    """Give unmapped context atoms distinct temporary identities per side."""
    used_maps = {
        atom.GetAtomMapNum()
        for molecule in (precursor, product)
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    next_map = max([500000, *used_maps]) + 1
    for molecule in (precursor, product):
        for atom in molecule.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(next_map)
                next_map += 1


def _reverse_side_local_blocks(
    molecule: Chem.Mol,
    opposite: Chem.Mol,
    occupied: set[int],
) -> list[_ReverseTransferBlock]:
    """Find maximal one-port carbon blocks present on only one side."""
    opposite_maps = set(_reverse_atoms_by_map(opposite))
    candidates = {
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and atom.GetAtomMapNum() > 0
        and atom.GetAtomMapNum() not in opposite_maps
        and atom.GetAtomMapNum() not in occupied
    }
    graph = nx.Graph()
    graph.add_nodes_from(candidates)
    for bond in molecule.GetBonds():
        first = bond.GetBeginAtom().GetAtomMapNum()
        second = bond.GetEndAtom().GetAtomMapNum()
        if first in candidates and second in candidates:
            graph.add_edge(first, second)

    blocks: list[_ReverseTransferBlock] = []

    def visit(nodes: set[int]) -> None:
        if not nodes:
            return
        boundary = _reverse_block_boundary(molecule, nodes)
        if len(boundary) == 1:
            root_map = boundary[0][0]
            blocks.append(
                _ReverseTransferBlock(
                    maps=frozenset(nodes),
                    root_map=root_map,
                    domain=_reverse_whole_r_domain(
                        molecule, nodes, root_map
                    ),
                )
            )
            return
        if not boundary:
            return
        interface_maps = {root for root, _, _ in boundary}
        remaining = nodes - interface_maps
        if not remaining:
            return
        for component in nx.connected_components(graph.subgraph(remaining)):
            visit(set(component))

    for component in nx.connected_components(graph):
        visit(set(component))
    return sorted(blocks, key=lambda block: (block.root_map, sorted(block.maps)))


def _reverse_renumber_r_by_occurrence(
    reaction: str, domains: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Number R groups by first appearance in the generic reaction string."""
    old_names: list[str] = []
    for match in re.finditer(r"\[(R[1-9][0-9]*)\]", reaction):
        name = match.group(1)
        if name not in old_names:
            old_names.append(name)
    renaming = {
        old_name: f"R{index}"
        for index, old_name in enumerate(old_names, start=1)
    }
    reaction = re.sub(
        r"\[(R[1-9][0-9]*)\]",
        lambda match: f"[{renaming[match.group(1)]}]",
        reaction,
    )
    updated_domains = {
        renaming.get(name, name): value for name, value in domains.items()
    }
    if len(old_names) == 1:
        reaction = reaction.replace("[R1]", "[R]")
        updated_domains["R"] = updated_domains.pop("R1")
    return reaction, updated_domains


def _reverse_contract_blocks(
    molecule: Chem.Mol,
    blocks: Sequence[_ReverseTransferBlock],
    h_slots: Sequence[tuple[int, int]],
) -> tuple[Chem.Mol, dict[int, str]]:
    """Contract all whole-R blocks simultaneously on one reaction side."""
    labels: dict[int, str] = {}
    block_by_map = {
        atom_map: index
        for index, block in enumerate(blocks, start=1)
        for atom_map in block.maps
    }
    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    block_to_new: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        block_index = block_by_map.get(atom_map)
        if block_index is not None:
            if block_index not in block_to_new:
                marker = 900000 + block_index
                dummy = Chem.Atom(0)
                dummy.SetAtomMapNum(marker)
                block_to_new[block_index] = editable.AddAtom(dummy)
                labels[marker] = f"R{block_index}"
            old_to_new[atom.GetIdx()] = block_to_new[block_index]
            continue
        copied = Chem.Atom(atom)
        copied.SetAtomMapNum(0)
        old_to_new[atom.GetIdx()] = editable.AddAtom(copied)

    added_edges: set[tuple[int, int]] = set()
    for bond in molecule.GetBonds():
        begin = old_to_new[bond.GetBeginAtomIdx()]
        end = old_to_new[bond.GetEndAtomIdx()]
        if begin == end:
            continue
        edge = tuple(sorted((begin, end)))
        if edge in added_edges:
            continue
        editable.AddBond(begin, end, bond.GetBondType())
        new_bond = editable.GetBondBetweenAtoms(begin, end)
        if new_bond is not None:
            new_bond.SetBondDir(bond.GetBondDir())
            new_bond.SetStereo(bond.GetStereo())
        added_edges.add(edge)

    for anchor_map, slot_index in h_slots:
        atom_by_map = {
            atom.GetAtomMapNum(): atom.GetIdx()
            for atom in molecule.GetAtoms()
            if atom.GetAtomMapNum() > 0
        }
        if anchor_map not in atom_by_map:
            continue
        old_anchor = atom_by_map[anchor_map]
        if old_anchor not in old_to_new:
            continue
        anchor = editable.GetAtomWithIdx(old_to_new[old_anchor])
        if anchor.GetNumExplicitHs() > 0:
            anchor.SetNumExplicitHs(anchor.GetNumExplicitHs() - 1)
        marker = 950000 + slot_index
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(marker)
        dummy_index = editable.AddAtom(dummy)
        editable.AddBond(anchor.GetIdx(), dummy_index, Chem.BondType.SINGLE)
        labels[marker] = f"R{len(blocks) + slot_index}"
    return editable.GetMol(), labels


def _reverse_whole_pseudo_side(
    molecule: Chem.Mol, labels: dict[int, str]
) -> str:
    """Serialize map-free pseudo-SMILES with organic/R components first."""
    components: list[tuple[tuple[int, int, str], str]] = []
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False):
        r_atoms = [
            atom
            for atom in fragment.GetAtoms()
            if atom.GetAtomMapNum() in labels
        ]
        carbon_count = sum(
            atom.GetAtomicNum() == 6 for atom in fragment.GetAtoms()
        )
        heavy_count = sum(
            atom.GetAtomicNum() > 1 for atom in fragment.GetAtoms()
        )

        # A standalone neutral hydrogen halide is written explicitly so the
        # later X abstraction retains its H decoration as [XH].
        if (
            fragment.GetNumAtoms() == 1
            and fragment.GetAtomWithIdx(0).GetAtomicNum()
            in HALOGEN_ATOMIC_NUMBERS
            and fragment.GetAtomWithIdx(0).GetTotalNumHs() > 0
        ):
            smiles = f"[{fragment.GetAtomWithIdx(0).GetSymbol()}H]"
        else:
            rooted_at = -1
            if r_atoms and carbon_count == 0:
                rooted_at = r_atoms[0].GetIdx()
            smiles = Chem.MolToSmiles(
                fragment,
                canonical=True,
                isomericSmiles=True,
                rootedAtAtom=rooted_at,
            )
        for marker, name in sorted(labels.items(), reverse=True):
            smiles = smiles.replace(f"[*:{marker}]", f"[{name}]")

        has_r = bool(r_atoms)
        category = (
            0
            if has_r and carbon_count
            else 1
            if has_r
            else 2
            if carbon_count
            else 3
        )
        components.append(((category, -heavy_count, smiles), smiles))
    return ".".join(value for _, value in sorted(components))


def _generic_whole_r_forward_from_template(
    template: str,
) -> tuple[str, dict[str, str]]:
    """Convert a normalized template to an unmapped whole-R reaction.

    Conserved blocks are shared between both sides.  Additional carbon
    context found only on one side is represented by a side-local R group when
    it has exactly one external interface; reaction-centre skeletons with
    multiple interfaces remain explicit.
    """
    changed, product_query, precursor_query = _reverse_changed_maps(template)
    witness = _forward_reaction_from_template(template, {})
    reactant_side, product_side = split_reaction(witness)
    precursor = Chem.MolFromSmiles(reactant_side)
    product = Chem.MolFromSmiles(product_side)
    if precursor is None or product is None:
        raise GenerationError("Could not parse a normalized template witness.")

    _reverse_assign_side_local_maps(precursor, product)
    shared_blocks = _reverse_transfer_blocks(product, precursor)
    occupied_maps = {
        atom_map for block in shared_blocks for atom_map in block.maps
    }
    precursor_local = _reverse_side_local_blocks(
        precursor, product, occupied_maps
    )
    product_local = _reverse_side_local_blocks(
        product, precursor, occupied_maps
    )
    blocks = [*shared_blocks, *precursor_local, *product_local]
    removed_maps = {atom_map for block in blocks for atom_map in block.maps}
    h_slot_maps = [
        atom_map
        for atom_map in _reverse_h_slots(product_query, precursor_query, changed)
        if atom_map not in removed_maps
    ]
    h_slots = [
        (atom_map, index)
        for index, atom_map in enumerate(h_slot_maps, start=1)
    ]
    precursor_generic, precursor_labels = _reverse_contract_blocks(
        precursor, blocks, h_slots
    )
    product_generic, product_labels = _reverse_contract_blocks(
        product, blocks, h_slots
    )
    reaction = (
        f"{_reverse_whole_pseudo_side(precursor_generic, precursor_labels)}>>"
        f"{_reverse_whole_pseudo_side(product_generic, product_labels)}"
    )
    domains = {
        f"R{index}": block.domain
        for index, block in enumerate(blocks, start=1)
    }
    domains.update(
        {
            f"R{len(blocks) + index}": "H"
            for index in range(1, len(h_slots) + 1)
        }
    )
    reaction, domains = _reverse_renumber_r_by_occurrence(reaction, domains)
    reaction, element_to_placeholder = _halogen_placeholders(reaction)
    domains.update(
        {
            placeholder: element
            for element, placeholder in element_to_placeholder.items()
        }
    )

    if re.search(r":\d+(?=\])", reaction):
        raise GenerationError("An atom map leaked into a reverse-generated reaction.")
    return reaction, domains


def _reverse_component_carbon_maps(
    molecule: Chem.Mol, atom_indices: Sequence[int]
) -> set[int]:
    return {
        molecule.GetAtomWithIdx(index).GetAtomMapNum()
        for index in atom_indices
        if molecule.GetAtomWithIdx(index).GetAtomicNum() == 6
    }


def _generic_grignard_halide_coupling(
    template: str,
) -> tuple[str, dict[str, str]] | None:
    """Represent a validated ``RMgX + R'X`` coupling with two whole R groups.

    Symmetric alkyl chains can receive different but equally valid atom maps.
    Building R groups from those maps may therefore split one rooted alkyl
    group into several methyl/H slots.  This recognizer instead uses the two
    precursor components and their Mg/X attachment atoms.  It is deliberately
    conservative: it accepts only one organomagnesium component, one organic
    halide, an all-carbon product, and conserved total carbon count.
    """
    witness = _forward_reaction_from_template(template, {})
    reactant_text, product_text = split_reaction(witness)
    precursor = Chem.MolFromSmiles(reactant_text)
    product = Chem.MolFromSmiles(product_text)
    if precursor is None or product is None:
        return None
    _reverse_assign_side_local_maps(precursor, product)

    organomagnesium: list[tuple[set[int], int, str]] = []
    organohalides: list[tuple[set[int], int, str]] = []
    for fragment in Chem.GetMolFrags(
        precursor, asMols=False, sanitizeFrags=False
    ):
        fragment_set = set(fragment)
        atoms = [precursor.GetAtomWithIdx(index) for index in fragment]
        carbon_maps = _reverse_component_carbon_maps(precursor, fragment)
        if not carbon_maps:
            continue
        magnesium = [atom for atom in atoms if atom.GetAtomicNum() == 12]
        halogens = [
            atom
            for atom in atoms
            if atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS
        ]
        allowed_atomic_numbers = {
            6,
            12,
            *HALOGEN_ATOMIC_NUMBERS,
        }
        if any(
            atom.GetAtomicNum() not in allowed_atomic_numbers for atom in atoms
        ):
            continue

        if len(magnesium) == 1:
            carbon_neighbours = [
                neighbour
                for neighbour in magnesium[0].GetNeighbors()
                if neighbour.GetIdx() in fragment_set
                and neighbour.GetAtomicNum() == 6
            ]
            x_neighbours = [
                neighbour
                for neighbour in magnesium[0].GetNeighbors()
                if neighbour.GetIdx() in fragment_set
                and neighbour.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS
            ]
            if (
                len(carbon_neighbours) == 1
                and len(x_neighbours) == 1
                and len(halogens) == 1
            ):
                organomagnesium.append(
                    (
                        carbon_maps,
                        carbon_neighbours[0].GetAtomMapNum(),
                        x_neighbours[0].GetSymbol(),
                    )
                )
            continue

        if magnesium or len(halogens) != 1:
            continue
        carbon_neighbours = [
            neighbour
            for neighbour in halogens[0].GetNeighbors()
            if neighbour.GetIdx() in fragment_set
            and neighbour.GetAtomicNum() == 6
        ]
        if len(carbon_neighbours) == 1:
            organohalides.append(
                (
                    carbon_maps,
                    carbon_neighbours[0].GetAtomMapNum(),
                    halogens[0].GetSymbol(),
                )
            )

    if len(organomagnesium) != 1 or len(organohalides) != 1:
        return None
    if len(Chem.GetMolFrags(product)) != 1:
        return None
    if any(atom.GetAtomicNum() != 6 for atom in product.GetAtoms()):
        return None

    first_maps, first_root, first_x = organomagnesium[0]
    second_maps, second_root, second_x = organohalides[0]
    if product.GetNumAtoms() != len(first_maps) + len(second_maps):
        return None
    domains = {
        "R1": _reverse_whole_r_domain(precursor, first_maps, first_root),
        "X1": first_x,
        "R2": _reverse_whole_r_domain(precursor, second_maps, second_root),
        "X2": second_x,
    }
    return "[R1][Mg][X1].[R2][X2]>>[R1][R2]", domains


def _reverse_single_variant_signature(
    reaction: str, domains: dict[str, str]
) -> tuple[str, str]:
    """Expand one proposed generic row for a lossless-abstraction check."""
    rows = expand_record(
        {
            "reaction_id": "reverse-qc",
            "reaction_name": "Reverse QC",
            "reaction": reaction,
            "domains": {name: [value] for name, value in domains.items()},
            "atom_sources": {},
            "condition": "",
            "source": "",
            "chapter": "",
            "mapping_mode": "auto",
            "allowed_variants": "",
            "template_derived": True,
        },
        max_combinations=1,
    )
    return rows[0]["reactant"], rows[0]["product"]


_REVERSE_R_PSEUDO = re.compile(
    r"\[(R(?:[1-9][0-9]*)?(?:[a-z])?)(?::([1-9][0-9]*))?\]"
)
_REVERSE_X_PSEUDO = re.compile(
    r"\[(X(?:[1-9][0-9]*)?)([Hh+\-0-9]*)(?::([1-9][0-9]*))?\]"
)
_REVERSE_DUMMY_PSEUDO = re.compile(r"\[(?:10[1-6])\*:(\d+)\]")


def _reverse_parse_pseudo_side(
    side: str, side_name: str
) -> tuple[Chem.Mol, dict[int, tuple[str, str]]]:
    occurrence = 0
    tags: dict[int, tuple[str, str]] = {}

    def replace_r(match: re.Match[str]) -> str:
        nonlocal occurrence
        occurrence += 1
        name, atom_map = match.group(1), match.group(2)
        local_map = int(atom_map) if atom_map else 980000 + occurrence
        tags[local_map] = ("R", name)
        return f"[101*:{local_map}]"

    def replace_x(match: re.Match[str]) -> str:
        nonlocal occurrence
        occurrence += 1
        name, decoration, atom_map = match.group(1), match.group(2), match.group(3)
        local_map = int(atom_map) if atom_map else 990000 + occurrence
        role = "XH" if "H" in decoration.upper() else "X"
        if "-" in decoration:
            role += "-"
        elif "+" in decoration:
            role += "+"
        isotope = {"X": 102, "XH": 103, "X-": 104, "X+": 105}.get(
            role, 106
        )
        tags[local_map] = (role, name)
        return f"[{isotope}*:{local_map}]"

    converted = _REVERSE_R_PSEUDO.sub(replace_r, side)
    converted = _REVERSE_X_PSEUDO.sub(replace_x, converted)
    molecule = Chem.MolFromSmiles(converted)
    if molecule is None:
        raise GenerationError(
            f"Could not parse reverse generic reaction side ({side_name}): {side}"
        )
    return molecule, tags


def _reverse_elide_h_side(
    side: str, h_names: set[str], side_name: str
) -> str:
    """Replace terminal, fixed-H R placeholders with ordinary implicit H."""
    molecule, tags = _reverse_parse_pseudo_side(side, side_name)
    found: set[str] = set()
    for atom in molecule.GetAtoms():
        role, name = tags.get(atom.GetAtomMapNum(), ("", ""))
        if role != "R" or name not in h_names:
            continue
        bonds = list(atom.GetBonds())
        if atom.GetDegree() != 1 or len(bonds) != 1:
            raise GenerationError(
                f"Fixed-H placeholder {name} is not terminal on {side_name}."
            )
        if bonds[0].GetBondType() != Chem.BondType.SINGLE:
            raise GenerationError(
                f"Fixed-H placeholder {name} is not single-bonded on {side_name}."
            )
        atom.SetIsotope(0)
        atom.SetAtomicNum(1)
        atom.SetAtomMapNum(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        found.add(name)

    if not h_names.issubset(found):
        absent = sorted(h_names - found, key=_reverse_placeholder_sort_key)
        raise GenerationError(
            f"Fixed-H placeholders were absent on {side_name}: {absent}."
        )
    molecule = Chem.RemoveHs(molecule, sanitize=True)
    result = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    def restore(match: re.Match[str]) -> str:
        atom_map = int(match.group(1))
        if atom_map not in tags:
            raise GenerationError(
                f"Unknown pseudo-atom map {atom_map} after H elision."
            )
        role, name = tags[atom_map]
        if role == "R":
            return f"[{name}]"
        decoration = "H" if role.startswith("XH") else ""
        if role.endswith("-"):
            decoration += "-"
        elif role.endswith("+"):
            decoration += "+"
        return f"[{name}{decoration}]"

    result = _REVERSE_DUMMY_PSEUDO.sub(restore, result)
    if "*" in result or re.search(r":\d+(?=\])", result):
        raise GenerationError(
            f"A pseudo atom remained after fixed-H elision: {result}"
        )
    return result


def _reverse_renumber_group_r(
    group: dict[str, object],
) -> None:
    """Renumber surviving R placeholders after fixed-H slots are removed."""
    reaction = str(group["reaction"])
    old_names: list[str] = []
    for match in R_PLACEHOLDER_PATTERN.finditer(reaction):
        name = _r_name(match.group("rgroup"), match.group("rsite"))
        if name not in old_names:
            old_names.append(name)
    renaming = {
        name: ("R" if len(old_names) == 1 else f"R{index}")
        for index, name in enumerate(old_names, start=1)
    }

    def replace(match: re.Match[str]) -> str:
        old_name = _r_name(match.group("rgroup"), match.group("rsite"))
        atom_map = match.group("rmap")
        map_suffix = f":{atom_map}" if atom_map else ""
        return f"[{renaming[old_name]}{map_suffix}]"

    group["reaction"] = R_PLACEHOLDER_PATTERN.sub(replace, reaction)
    domains = group["domains"]
    variants = group["variants"]
    assert isinstance(domains, dict)
    assert isinstance(variants, list)
    group["domains"] = {
        renaming.get(str(name), str(name)): values
        for name, values in domains.items()
    }
    group["variants"] = [
        {
            renaming.get(str(name), str(name)): value
            for name, value in variant.items()
        }
        for variant in variants
    ]


def _reverse_elide_fixed_h_group(group: dict[str, object]) -> int:
    """Remove only R positions whose aggregated domain is exactly ``H``.

    Replacing the pseudo atom by real H before ``RemoveHs`` preserves the
    correct implicit-H state, including charged carbon centres.  A placeholder
    whose domain also contains a carbon class is retained as genuine
    variability.
    """
    domains = group["domains"]
    variants = group["variants"]
    assert isinstance(domains, dict)
    assert isinstance(variants, list)
    h_names = {
        str(name)
        for name, values in domains.items()
        if str(name).startswith("R")
        and isinstance(values, list)
        and set(values) == {"H"}
    }
    if not h_names:
        return 0

    reactants, products = split_reaction(str(group["reaction"]))
    group["reaction"] = (
        f"{_reverse_elide_h_side(reactants, h_names, 'reactants')}>>"
        f"{_reverse_elide_h_side(products, h_names, 'products')}"
    )
    group["domains"] = {
        str(name): values
        for name, values in domains.items()
        if str(name) not in h_names
    }
    projected: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for variant in variants:
        kept = {
            str(name): str(value)
            for name, value in variant.items()
            if str(name) not in h_names
        }
        key = tuple(sorted(kept.items(), key=lambda item: _reverse_placeholder_sort_key(item[0])))
        if key not in seen:
            seen.add(key)
            projected.append(kept)
    group["variants"] = projected
    _reverse_renumber_group_r(group)
    return len(h_names)


def _reverse_generic_graph(reaction: str) -> nx.Graph:
    reactants, products = split_reaction(reaction)
    graph = nx.Graph()
    map_nodes: dict[int, list[str]] = defaultdict(list)
    choice_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
    # Placeholder identity is also global across the two sides.  This is
    # essential for map-free reverse output: the same [R1] still denotes the
    # same domain choice even though it deliberately has no atom-map number.
    global_choice_nodes: dict[str, list[str]] = defaultdict(list)
    for side_name, side in (("reactant", reactants), ("product", products)):
        molecule, tags = _reverse_parse_pseudo_side(side, side_name)
        for atom in molecule.GetAtoms():
            node = f"{side_name}:{atom.GetIdx()}"
            atom_map = atom.GetAtomMapNum()
            role, choice = tags.get(atom_map, ("atom", ""))
            if role == "atom":
                label = (
                    side_name,
                    role,
                    atom.GetAtomicNum(),
                    atom.GetFormalCharge(),
                    atom.GetIsAromatic(),
                    atom.GetTotalNumHs(),
                    str(atom.GetChiralTag()),
                )
            else:
                label = (side_name, role)
                choice_nodes[(side_name, choice)].append(node)
                global_choice_nodes[choice].append(node)
            graph.add_node(
                node,
                label=repr(label),
                placeholder_name=choice,
                placeholder_role=role,
            )
            if 0 < atom_map < 980000:
                map_nodes[atom_map].append(node)
        for bond in molecule.GetBonds():
            graph.add_edge(
                f"{side_name}:{bond.GetBeginAtomIdx()}",
                f"{side_name}:{bond.GetEndAtomIdx()}",
                label=(
                    f"bond:{bond.GetBondType()}:{int(bond.GetIsAromatic())}:"
                    f"{bond.GetStereo()}"
                ),
            )

    for index, atoms in enumerate(map_nodes.values(), start=1):
        if len(atoms) < 2:
            continue
        relation = f"map:{index}"
        graph.add_node(relation, label="map-relation")
        for node in atoms:
            graph.add_edge(relation, node, label="mapped-as")
    for index, atoms in enumerate(choice_nodes.values(), start=1):
        if len(atoms) < 2:
            continue
        relation = f"choice:{index}"
        graph.add_node(relation, label="same-placeholder-choice")
        for node in atoms:
            graph.add_edge(relation, node, label="shares-choice")
    for index, atoms in enumerate(global_choice_nodes.values(), start=1):
        if len(atoms) < 2:
            continue
        relation = f"global-choice:{index}"
        graph.add_node(relation, label="same-generic-placeholder")
        for node in atoms:
            graph.add_edge(relation, node, label="shares-global-choice")
    return graph


def _reverse_graph_hash(graph: nx.Graph) -> str:
    return nx.weisfeiler_lehman_graph_hash(
        graph, node_attr="label", edge_attr="label", iterations=5
    )


def _reverse_graph_matcher(
    reference: nx.Graph, candidate: nx.Graph
) -> nx.isomorphism.GraphMatcher:
    return nx.isomorphism.GraphMatcher(
        reference,
        candidate,
        node_match=lambda first, second: first["label"] == second["label"],
        edge_match=lambda first, second: first["label"] == second["label"],
    )


def _reverse_aligned_domains(
    reference: nx.Graph,
    candidate: nx.Graph,
    candidate_domains: dict[str, str],
) -> dict[str, str]:
    reference_names = sorted(
        {
            str(attributes.get("placeholder_name", ""))
            for _, attributes in reference.nodes(data=True)
            if attributes.get("placeholder_name", "")
        },
        key=_reverse_placeholder_sort_key,
    )
    aligned: dict[str, str] = {}

    # Find the lexicographically smallest *jointly feasible* domain assignment.
    # This avoids selecting an arbitrary first GraphMatcher mapping for symmetric
    # R positions, while also avoiding enumeration of a potentially large full
    # automorphism group.  At each step an isomorphism must still exist under all
    # assignments fixed so far, so the final tuple is guaranteed realizable.
    for reference_name in reference_names:
        kind = reference_name[0]
        choices = sorted(
            {
                value
                for name, value in candidate_domains.items()
                if name.startswith(kind)
            },
            key=_reverse_domain_sort_key,
        )
        selected: str | None = None
        for value in choices:
            constraints = {**aligned, reference_name: value}

            def node_match(
                reference_attributes: dict[str, object],
                candidate_attributes: dict[str, object],
            ) -> bool:
                if reference_attributes["label"] != candidate_attributes["label"]:
                    return False
                constrained_name = str(
                    reference_attributes.get("placeholder_name", "")
                )
                if constrained_name not in constraints:
                    return True
                candidate_name = str(
                    candidate_attributes.get("placeholder_name", "")
                )
                return (
                    candidate_name in candidate_domains
                    and candidate_domains[candidate_name]
                    == constraints[constrained_name]
                )

            matcher = nx.isomorphism.GraphMatcher(
                reference,
                candidate,
                node_match=node_match,
                edge_match=lambda first, second: first["label"]
                == second["label"],
            )
            if matcher.is_isomorphic():
                selected = value
                break
        if selected is None:
            raise GenerationError(
                f"Could not align symmetric placeholder {reference_name}."
            )
        aligned[reference_name] = selected
    return aligned


def _reverse_placeholder_sort_key(name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"([RX])([1-9][0-9]*)?([a-z])?", name)
    if match is None:
        return 9, 0, name
    kind, suffix, site = match.groups()
    return (0 if kind == "R" else 1), int(suffix or 0), site or ""


def _reverse_domain_sort_key(value: str) -> int:
    ordered = [*FRAGMENTS, *DEFAULT_X_DOMAINS]
    try:
        return ordered.index(value)
    except ValueError:
        return len(ordered)


def _reaction_family_name(value: object) -> str:
    family = str(value or "unclassified").strip()
    return family.replace("_", " ").title()


def _reverse_reaction_fingerprints(
    templates: Sequence[str],
) -> list[tuple[object, object]]:
    """Return reaction-centre and whole-reaction fingerprints for sorting."""
    fingerprints: list[tuple[object, object]] = []
    for template in templates:
        reaction = rdChemReactions.ReactionFromSmarts(template)
        if reaction is None:
            raise GenerationError(
                "Could not parse a source template for similarity sorting."
            )
        fingerprints.append(
            (
                rdChemReactions.CreateDifferenceFingerprintForReaction(reaction),
                rdChemReactions.CreateStructuralFingerprintForReaction(reaction),
            )
        )
    return fingerprints


def _reverse_group_similarity(
    first: dict[str, object], second: dict[str, object]
) -> float:
    """Compare groups, prioritizing the reaction centre over global structure."""
    first_fingerprints = first["similarity_fingerprints"]
    second_fingerprints = second["similarity_fingerprints"]
    assert isinstance(first_fingerprints, list)
    assert isinstance(second_fingerprints, list)
    similarities = (
        0.8 * DataStructs.TanimotoSimilarity(first_difference, second_difference)
        + 0.2 * DataStructs.TanimotoSimilarity(first_structure, second_structure)
        for first_difference, first_structure in first_fingerprints
        for second_difference, second_structure in second_fingerprints
    )
    return max(similarities, default=0.0)


def _reverse_group_stable_key(
    group: dict[str, object],
) -> tuple[str, bool, str, str]:
    hashes = group.get("hashes", [])
    first_hash = min((str(value) for value in hashes), default="")
    return (
        str(group["family"]),
        bool(group["fallback"]),
        str(group["reaction"]),
        first_hash,
    )


def _reverse_similarity_order(
    groups: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Place structurally similar reactions next to each other within a chapter.

    A deterministic nearest-neighbour path is started from the locally most
    connected reaction.  A 2-opt pass then improves adjacent-pair similarity.
    Source-template fingerprints are used so R/X pseudo-atoms do not distort
    the chemical comparison.
    """
    ordered_all: list[dict[str, object]] = []
    chapters: dict[int, list[dict[str, object]]] = defaultdict(list)
    for group in groups:
        source_templates = group["source_templates"]
        assert isinstance(source_templates, list)
        group["similarity_fingerprints"] = _reverse_reaction_fingerprints(
            [str(template) for template in source_templates]
        )
        chapters[int(group["chapter"])].append(group)

    for chapter in sorted(chapters):
        chapter_groups = sorted(chapters[chapter], key=_reverse_group_stable_key)
        count = len(chapter_groups)
        if count < 2:
            ordered_all.extend(chapter_groups)
            continue

        similarity: dict[tuple[int, int], float] = {}
        for first in range(count):
            similarity[first, first] = 1.0
            for second in range(first + 1, count):
                value = _reverse_group_similarity(
                    chapter_groups[first], chapter_groups[second]
                )
                similarity[first, second] = value
                similarity[second, first] = value

        # Starting inside the densest local neighbourhood avoids putting an
        # arbitrary structural outlier at the head of each chapter.
        neighbour_count = min(5, count - 1)
        connectivity = {
            index: sum(
                sorted(
                    (
                        similarity[index, other]
                        for other in range(count)
                        if other != index
                    ),
                    reverse=True,
                )[:neighbour_count]
            )
            for index in range(count)
        }
        start = min(
            range(count),
            key=lambda index: (
                -connectivity[index],
                _reverse_group_stable_key(chapter_groups[index]),
            ),
        )
        path = [start]
        remaining = set(range(count)) - {start}
        while remaining:
            current = path[-1]
            next_index = min(
                remaining,
                key=lambda index: (
                    -similarity[current, index],
                    -connectivity[index],
                    _reverse_group_stable_key(chapter_groups[index]),
                ),
            )
            path.append(next_index)
            remaining.remove(next_index)

        # Improve the linear path without changing its deterministic start.
        improved = True
        while improved:
            improved = False
            for left in range(1, count - 1):
                for right in range(left + 1, count):
                    old_score = similarity[path[left - 1], path[left]]
                    new_score = similarity[path[left - 1], path[right]]
                    if right + 1 < count:
                        old_score += similarity[path[right], path[right + 1]]
                        new_score += similarity[path[left], path[right + 1]]
                    if new_score > old_score + 1e-12:
                        path[left : right + 1] = reversed(path[left : right + 1])
                        improved = True

        ordered_all.extend(chapter_groups[index] for index in path)

    for group in ordered_all:
        group.pop("similarity_fingerprints", None)
    return ordered_all


def _reverse_clear_atom_maps(side: str) -> str:
    molecule = Chem.MolFromSmiles(side)
    if molecule is None:
        raise GenerationError(f"Could not parse materialized witness: {side}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _reverse_self_replay(template: str, product: str) -> tuple[bool, str]:
    """Check that a normalized template can act on its own witness product."""
    try:
        from rdchiral.initialization import rdchiralReaction, rdchiralReactants
        from rdchiral.main import rdchiralRun

        outcomes = rdchiralRun(
            rdchiralReaction(template),
            rdchiralReactants(product),
            keep_mapnums=False,
        )
    except Exception as exc:
        return False, f"self-replay raised {type(exc).__name__}: {exc}"
    if outcomes:
        return True, ""
    return False, "extracted template returned no precursors for its own product"


def _reverse_normalize_template_rows(
    template_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Remap materialized witnesses and extract a clean radius-1 template."""
    try:
        from joblib import Parallel, delayed
        from rxnmapper import BatchedMapper
    except ModuleNotFoundError as exc:
        raise GenerationError(
            f"Missing reverse-generation dependency '{exc.name}'. "
            "Run: pip install -r requirements.txt"
        ) from exc

    normalized_rows: list[dict[str, object]] = []
    pending_indices: list[int] = []
    unmapped_reactions: list[str] = []
    for row in template_rows:
        record = dict(row)
        try:
            witness = _forward_reaction_from_template(str(row["template"]), {})
            reactants, products = split_reaction(witness)
            unmapped = (
                f"{_reverse_clear_atom_maps(reactants)}>>"
                f"{_reverse_clear_atom_maps(products)}"
            )
            record["materialized_witness"] = witness
            record["unmapped_witness"] = unmapped
            pending_indices.append(len(normalized_rows))
            unmapped_reactions.append(unmapped)
        except Exception as exc:
            record["normalization_status"] = "materialization_error"
            record["exclusion_reason"] = str(exc)
        normalized_rows.append(record)

    if unmapped_reactions:
        mapped_reactions = list(
            BatchedMapper(batch_size=20).map_reactions(unmapped_reactions)
        )
        if len(mapped_reactions) != len(unmapped_reactions):
            raise GenerationError(
                "Atom mapper returned a different number of reverse witnesses."
            )

        def extract(mapped_reaction: str, index: int) -> str:
            reactants, products = split_reaction(mapped_reaction)
            return _reaction_smarts(
                _extract_one(
                    {
                        "_id": f"reverse-normalize-{index}",
                        "reactants": reactants,
                        "products": products,
                    },
                    radius=1,
                    timeout_seconds=30,
                )
            )

        templates = Parallel(n_jobs=min(8, len(mapped_reactions)), verbose=0)(
            delayed(extract)(reaction, index)
            for index, reaction in enumerate(mapped_reactions)
        )
        for row_index, mapped_reaction, template in zip(
            pending_indices, mapped_reactions, templates
        ):
            record = normalized_rows[row_index]
            record["normalized_mapped_reaction"] = mapped_reaction
            record["normalized_template"] = template
            if _usable_template(template):
                _, product = split_reaction(str(record["unmapped_witness"]))
                replayed, reason = _reverse_self_replay(template, product)
                record["normalization_status"] = (
                    "usable" if replayed else "replay_failed"
                )
                record["exclusion_reason"] = reason
            elif not template:
                record["normalization_status"] = "empty"
                reactants, products = split_reaction(
                    str(record["unmapped_witness"])
                )
                reason = (
                    "materialized witness is an identity reaction"
                    if reactants == products
                    else "radius-1 template extraction returned no template"
                )
                record["exclusion_reason"] = reason
            else:
                record["normalization_status"] = "unusable"
                record["exclusion_reason"] = (
                    "extracted template has multiple product components and "
                    "is not executable by this library"
                )
    return normalized_rows


def _write_reverse_exclusions(
    rows: Sequence[dict[str, object]], output_path: Path
) -> None:
    fieldnames = [
        "template_hash",
        "chapter",
        "reaction_family",
        "normalization_status",
        "reason",
        "unmapped_witness",
        "normalized_mapped_reaction",
        "normalized_template",
        "source_template",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "template_hash": str(row.get("template_hash", "")),
                    "chapter": str(row.get("intro_chapter", "")),
                    "reaction_family": str(row.get("reaction_family", "")),
                    "normalization_status": str(
                        row.get("normalization_status", "")
                    ),
                    "reason": str(row.get("exclusion_reason", "")),
                    "unmapped_witness": str(row.get("unmapped_witness", "")),
                    "normalized_mapped_reaction": str(
                        row.get("normalized_mapped_reaction", "")
                    ),
                    "normalized_template": str(
                        row.get("normalized_template", "")
                    ),
                    "source_template": str(row.get("template", "")),
                }
            )


def _reverse_exclusions_output_path(output_path: Path) -> Path:
    """Derive a non-colliding exclusions filename beside the main CSV."""
    stem = output_path.stem
    if stem.endswith("_reactions"):
        stem = stem[: -len("_reactions")]
    suffix = output_path.suffix or ".csv"
    exclusions_path = output_path.with_name(f"{stem}_exclusions{suffix}")
    if exclusions_path.resolve() == output_path.resolve():
        raise GenerationError("Main and exclusions CSV paths must be different.")
    return exclusions_path


def reverse_template_library(input_path: Path, output_path: Path) -> dict[str, object]:
    """Compress SMARTS into map-free, whole-R reactions for human review.

    Each source SMARTS is first materialized, stripped of its historical atom
    maps, remapped, and re-extracted.  Only normalized executable templates are
    generalized.  Correlated R/X choices are retained in ``allowed_variants``.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    template_rows = payload.get("templates")
    if not isinstance(template_rows, list) or not template_rows:
        raise GenerationError("Template library must contain a non-empty templates list.")

    sorted_templates = sorted(
        template_rows,
        key=lambda row: (
            int(row.get("intro_chapter", 0)),
            str(row.get("reaction_family", "")),
            str(row.get("template_hash", "")),
        ),
    )
    for row in sorted_templates:
        if not row.get("template") or not row.get("template_hash"):
            raise GenerationError("Every template requires template and template_hash.")

    normalized_rows = _reverse_normalize_template_rows(sorted_templates)
    usable_rows = [
        row
        for row in normalized_rows
        if row.get("normalization_status") == "usable"
    ]
    excluded_rows = [
        row
        for row in normalized_rows
        if row.get("normalization_status") != "usable"
    ]
    if not usable_rows:
        raise GenerationError("No executable templates remained after normalization.")

    buckets: dict[tuple[int, str, str], list[dict[str, object]]] = defaultdict(list)
    grignard_whole_r_count = 0
    for row in usable_rows:
        source_template = str(row["template"])
        normalized_template = str(row["normalized_template"])
        template_hash = str(row["template_hash"])
        chapter = int(row.get("intro_chapter", 0))
        family = str(row.get("reaction_family", "") or "unclassified")
        base_result = _generic_whole_r_forward_from_template(
            normalized_template
        )
        grignard_result = None
        if family.strip().lower() == "organometallic_coupling":
            grignard_result = _generic_grignard_halide_coupling(
                normalized_template
            )
        if (
            grignard_result is not None
            and _reverse_single_variant_signature(*grignard_result)
            == _reverse_single_variant_signature(*base_result)
        ):
            reaction, candidate_domains = grignard_result
            grignard_whole_r_count += 1
        else:
            reaction, candidate_domains = base_result
        graph = _reverse_generic_graph(reaction)
        key = (chapter, family, _reverse_graph_hash(graph))
        group: dict[str, object] | None = None
        aligned_domains: dict[str, str] = {}
        for possible_group in buckets[key]:
            reference_graph = possible_group["graph"]
            assert isinstance(reference_graph, nx.Graph)
            if _reverse_graph_matcher(reference_graph, graph).is_isomorphic():
                group = possible_group
                aligned_domains = _reverse_aligned_domains(
                    reference_graph, graph, candidate_domains
                )
                if set(aligned_domains) != set(group["domains"]):
                    raise GenerationError(
                        f"Incomplete R/X alignment for template {template_hash}."
                    )
                break
        if group is None:
            group = {
                "chapter": chapter,
                "family": family,
                "reaction": reaction,
                "graph": graph,
                "domains": {
                    name: [value] for name, value in candidate_domains.items()
                },
                "variants": [],
                "conditions": [],
                "hashes": [],
                "source_templates": [],
                "fallback": False,
            }
            buckets[key].append(group)
            aligned_domains = dict(candidate_domains)

        domains = group["domains"]
        variants = group["variants"]
        conditions = group["conditions"]
        hashes = group["hashes"]
        source_templates = group["source_templates"]
        assert isinstance(domains, dict)
        assert isinstance(variants, list)
        assert isinstance(conditions, list)
        assert isinstance(hashes, list)
        assert isinstance(source_templates, list)
        for name, value in aligned_domains.items():
            values = domains.setdefault(name, [])
            if value not in values:
                values.append(value)
        names = placeholder_names(str(group["reaction"]))
        variant_key = tuple((name, aligned_domains[name]) for name in names)
        if variant_key not in {
            tuple((name, variant[name]) for name in names) for variant in variants
        }:
            variants.append(dict(aligned_domains))
        condition = str(row.get("conditions", "")).strip()
        if condition and condition not in conditions:
            conditions.append(condition)
        hashes.append(template_hash)
        source_templates.append(source_template)

    grouped = _reverse_similarity_order(
        [group for groups in buckets.values() for group in groups]
    )
    fixed_h_slots_elided = sum(
        _reverse_elide_fixed_h_group(group) for group in grouped
    )
    max_r = max(
        2,
        max(
            (
                int(match.group(1))
                for group in grouped
                for name in group["domains"]
                if (match := re.fullmatch(r"R([1-9][0-9]*)", str(name)))
            ),
            default=2,
        ),
    )
    max_x = max(
        2,
        max(
            (
                int(match.group(1))
                for group in grouped
                for name in group["domains"]
                if (match := re.fullmatch(r"X([1-9][0-9]*)", str(name)))
            ),
            default=2,
        ),
    )
    r_columns = [f"R{index}" for index in range(1, max_r + 1)]
    x_columns = ["X", *[f"X{index}" for index in range(1, max_x + 1)]]
    fieldnames = [
        "reaction_id",
        "reaction_name",
        "reaction",
        *r_columns,
        *x_columns,
        "allowed_variants",
        "atom_source",
        "condition",
        "source",
        "chapter",
        "mapping_mode",
        "template_hash",
        "source_template",
    ]
    counters: Counter[int] = Counter()
    output_rows: list[dict[str, str]] = []
    for group in grouped:
        chapter = int(group["chapter"])
        counters[chapter] += 1
        reaction = str(group["reaction"])
        if re.search(r":\d+(?=\])", reaction):
            raise GenerationError("Reverse output must not contain atom maps.")
        names = placeholder_names(reaction)
        domains = group["domains"]
        variants = group["variants"]
        assert isinstance(domains, dict)
        assert isinstance(variants, list)
        ordered_variants = sorted(
            variants,
            key=lambda variant: tuple(
                _reverse_domain_sort_key(variant[name]) for name in names
            ),
        )
        output_row = {column: "" for column in fieldnames}
        output_row.update(
            {
                "reaction_id": f"{chapter}-{counters[chapter]:03d}",
                "reaction_name": _reaction_family_name(group["family"]),
                "reaction": reaction,
                "allowed_variants": " | ".join(
                    ",".join(f"{name}={variant[name]}" for name in names)
                    for variant in ordered_variants
                )
                if names
                else "",
                "atom_source": "",
                "condition": " | ".join(group["conditions"]),
                "source": input_path.name,
                "chapter": str(chapter),
                "mapping_mode": "auto",
                "template_hash": ";".join(group["hashes"]),
                "source_template": " || ".join(group["source_templates"]),
            }
        )
        for name, values in domains.items():
            column = "R1" if name == "R" else str(name)
            output_row[column] = ";".join(
                sorted(values, key=_reverse_domain_sort_key)
            )
        output_rows.append(output_row)

    covered_hashes = [
        template_hash for group in grouped for template_hash in group["hashes"]
    ]
    expected_hashes = [str(row["template_hash"]) for row in usable_rows]
    if len(covered_hashes) != len(usable_rows) or set(covered_hashes) != set(
        expected_hashes
    ):
        raise GenerationError(
            "Reverse grouping did not retain every normalized usable template."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclusions_path = _reverse_exclusions_output_path(output_path)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    _write_reverse_exclusions(excluded_rows, exclusions_path)
    return {
        "source_template_count": len(template_rows),
        "normalized_usable_template_count": len(usable_rows),
        "excluded_template_count": len(excluded_rows),
        "reverse_generated_reaction_count": len(output_rows),
        "template_coverage_count": len(covered_hashes),
        "collapsed_template_count": len(usable_rows) - len(output_rows),
        "allowed_variant_count": sum(len(group["variants"]) for group in grouped),
        "grignard_whole_r_template_count": grignard_whole_r_count,
        "fixed_h_slots_elided": fixed_h_slots_elided,
        "reactions_by_chapter": dict(sorted(counters.items())),
        "exclusions_output": str(exclusions_path),
    }


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

            mapping_mode = row.get("mapping_mode", "auto").strip().lower() or "auto"
            if mapping_mode not in {"auto", "preserve"}:
                raise GenerationError(
                    f"{reaction_id}: mapping_mode must be 'auto' or 'preserve'."
                )

            placeholders = placeholder_names(reaction)
            domains: dict[str, list[str]] = {}
            for column, value in row.items():
                if not value or re.fullmatch(r"[RX](?:[1-9][0-9]*)?", column) is None:
                    continue
                placeholder = column
                if column == "R1" and "R" in placeholders and "R1" not in placeholders:
                    placeholder = "R"
                elif column == "X1" and "X" in placeholders and "X1" not in placeholders:
                    placeholder = "X"
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
                    "mapping_mode": mapping_mode,
                    "allowed_variants": row.get("allowed_variants", ""),
                    # Reverse-generated rows already come from executable
                    # templates.  Radius-limited witnesses can be formally
                    # atom-unbalanced, so do not invent atom-source reagents.
                    "template_derived": bool(row.get("source_template", "")),
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
    allowed_variants = str(record.get("allowed_variants", "")).strip()
    if allowed_variants:
        selections = _allowed_variant_selections(
            allowed_variants, placeholders, domains
        )
    else:
        selections = [
            dict(zip(placeholders, values))
            for values in itertools.product(*(domains[name] for name in placeholders))
        ]
    if len(selections) > max_combinations:
        raise GenerationError(
            f"{record['reaction_id']} expands to {len(selections)} combinations; "
            f"the limit is {max_combinations}."
        )

    r_placeholders = [name for name in placeholders if name.startswith("R")]
    dummy_maps = {name: 8000 + index for index, name in enumerate(r_placeholders)}
    declared_r_maps = _r_placeholder_maps(generic_reaction)
    rows: list[dict[str, str]] = []
    for index, selection in enumerate(selections, start=1):
        concrete_reactants = _replace_x_placeholders(reactant_side, selection)
        concrete_products = _replace_x_placeholders(product_side, selection)
        mapping_mode = str(record.get("mapping_mode", "auto"))
        preserve_atom_maps = mapping_mode == "preserve"
        fragment_atom_maps: dict[str, Sequence[int]] | None = None
        if preserve_atom_maps and r_placeholders:
            if any(declared_r_maps.get(name) is None for name in r_placeholders):
                raise GenerationError(
                    f"{record['reaction_id']}: every [R] placeholder requires an "
                    "explicit atom map in preserve mode, e.g. [R1:70001]."
                )
            root_maps = [int(declared_r_maps[name]) for name in r_placeholders]
            if len(set(root_maps)) != len(root_maps):
                raise GenerationError(
                    f"{record['reaction_id']}: mapped [R] placeholders must use "
                    "different atom maps."
                )
            used_maps = {
                int(value)
                for value in re.findall(r":([1-9][0-9]*)(?=\])", generic_reaction)
            }
            next_map = max([70000, *used_maps]) + 1
            fragment_atom_maps = {}
            for name, root_map in zip(r_placeholders, root_maps):
                maps = [root_map]
                while len(maps) < 6:
                    while next_map in used_maps:
                        next_map += 1
                    maps.append(next_map)
                    used_maps.add(next_map)
                    next_map += 1
                fragment_atom_maps[name] = maps
        reactants = expand_side(
            reactant_side,
            selection,
            dummy_maps,
            preserve_atom_maps,
            fragment_atom_maps,
        )
        products = expand_side(
            product_side,
            selection,
            dummy_maps,
            preserve_atom_maps,
            fragment_atom_maps,
        )
        if preserve_atom_maps and not _valid_preserved_mapping(reactants, products):
            raise GenerationError(
                f"{record['reaction_id']}: preserve mode requires a valid mapped "
                "product and consistent atom maps."
            )
        if not preserve_atom_maps and not bool(record.get("template_derived")):
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
                "mapping_mode": mapping_mode,
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
        "mapping_mode",
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
    if "mapping_mode" not in data.columns:
        data["mapping_mode"] = "auto"
    data["mapping_mode"] = (
        data["mapping_mode"].astype(str).str.strip().str.lower().replace("", "auto")
    )
    invalid_modes = sorted(set(data["mapping_mode"]) - {"auto", "preserve"})
    if invalid_modes:
        raise GenerationError(
            f"Unknown mapping_mode value(s): {', '.join(invalid_modes)}"
        )
    if any(not value.strip() for value in data["_id"]):
        raise GenerationError("Every row requires a non-empty _id.")
    if data["_id"].duplicated().any():
        raise GenerationError("Every _id must be unique.")

    reactions = [
        f"{reactant}>>{product}"
        for reactant, product in zip(data["reactant"], data["product"])
    ]
    mapped_reactions = [""] * len(reactions)
    automatic_indices: list[int] = []
    for index, (reaction, mode) in enumerate(
        zip(reactions, data["mapping_mode"])
    ):
        reactants, products = split_reaction(reaction)
        if mode == "preserve":
            if not _valid_preserved_mapping(reactants, products):
                raise GenerationError(
                    f"{data.iloc[index]['_id']}: invalid atom maps for preserve mode."
                )
            mapped_reactions[index] = reaction
        else:
            automatic_indices.append(index)

    if automatic_indices:
        mapper = BatchedMapper(batch_size=20)
        automatic_reactions = [reactions[index] for index in automatic_indices]
        automatic_mapped = list(mapper.map_reactions(automatic_reactions))
        if len(automatic_mapped) != len(automatic_indices):
            raise GenerationError("Atom mapper returned a different number of reactions.")
        for index, mapped_reaction in zip(automatic_indices, automatic_mapped):
            mapped_reactions[index] = mapped_reaction

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
        "preserved_mapping_count": len(data) - len(automatic_indices),
        "template_instance_count": len(templates),
        "unique_template_count": len(set(templates)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reverse a template library, expand [R]/[X] reactions, and extract "
            "executable templates."
        )
    )
    commands = parser.add_subparsers(dest="command")

    reverse = commands.add_parser(
        "reverse",
        help="Create a map-free, human-checkable CSV directly from a template library.",
    )
    reverse.add_argument(
        "--input", type=Path, required=True, help="Chapter template library JSON."
    )
    reverse.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_DIR / "template_derived_reactions.csv",
        help="Output generic-reaction CSV path.",
    )

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
        if args.command == "reverse":
            summary = reverse_template_library(
                args.input.resolve(), args.output.resolve()
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            print(f"Check this file before expansion: {args.output.resolve()}")
        elif args.command == "expand":
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
            raise GenerationError(
                "Choose the 'reverse', 'expand', or 'extract' command."
            )
        return 0
    except (GenerationError, FileNotFoundError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
