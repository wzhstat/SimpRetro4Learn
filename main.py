import os
import re
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw

from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants


def CDScore(p_mol, r_mols):
    p_atom_count = p_mol.GetNumAtoms()
    n_r_mols = len(r_mols)
    if len(r_mols) == 1:
        return 0
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    main_r = r_mols[np.argmax(r_atom_count)]
    if len(Chem.MolFromSmiles(main_r).GetAtoms()) >= p_atom_count:
        return 0
    MAE =  1 / n_r_mols * sum([abs(p_atom_count / n_r_mols - r_atom_count[i]) for i in range(n_r_mols)])
    return 1 / (1 + MAE) * p_atom_count

def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

def ASScore(p_mol, r_mol_dict, in_stock):
    p_atom_count = p_mol.GetNumAtoms()
    r_mols = list(r_mol_dict.keys())
    r_atom_count = [len([int(num[1:]) for num in re.findall(r':\d+', r_mol) if int(num[1:]) < 900]) for r_mol in r_mols]
    main_r = r_mols[np.argmax(r_atom_count)]
    asscore = 0
    for k, v in r_mol_dict.items():
        if v in in_stock:
            add = len([int(num[1:]) for num in re.findall(r':\d+', k) if int(num[1:]) < 900])
            if len(Chem.MolFromSmiles(main_r).GetAtoms()) < p_atom_count:
                asscore += add
            else:
                asscore += add if add > 2 else 0
        if ('Mg' in v or 'Li' in v or 'Zn' in v) and v not in in_stock:
            asscore -= 5
    return asscore

def RDScore(p_mol, r_mols):
    p_ring_count = p_mol.GetRingInfo().NumRings()
    r_rings_s =[r_mol.GetRingInfo().AtomRings() for r_mol in r_mols]
    r_ring_count = 0
    for r_rings, r_mol in zip(r_rings_s, r_mols):
        for r_ring in r_rings:
            mapnums =[r_mol.GetAtomWithIdx(i).GetAtomMapNum() for i in r_ring]
            symbols =[r_mol.GetAtomWithIdx(i).GetSymbol() for i in r_ring]
            if 'B' in symbols or 'Si' in symbols:
                continue
            if min(mapnums) < 900:
                r_ring_count += 1
    if p_ring_count > r_ring_count:
        return 1
    else:
        return 0

# Global variables
in_stock = None
templates_raw = None
template_list = None
tpl_condition = None

def load_global_data(data):
    """Load stock molecules and reaction templates."""
    global in_stock, templates_raw, template_list, tpl_condition
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Parse database path
    if data.get('databaseName') in ['emol_under_0_carbons', 'unrestricted']:
        emolecules_path = None
    else:
        emolecules_path = os.path.join(base_dir, data.get('databaseName', '') + '.txt')

    # 2. Parse template and condition file paths
    tpl_filename = data.get('template_file', 'reaction_template.json')
    tc_filename = data.get('condition_file', 'template_condition.json')
    
    template_path = os.path.join(base_dir, tpl_filename) if not os.path.isabs(tpl_filename) else tpl_filename
    tpl_c_path = os.path.join(base_dir, tc_filename) if not os.path.isabs(tc_filename) else tc_filename
    
    # Load in-stock molecules
    if emolecules_path is None:
        in_stock = set()
        print("[Load] Unrestricted in-stock molecules.")
    elif os.path.exists(emolecules_path):
        with open(emolecules_path, 'r', encoding='utf-8') as f:
            in_stock = set(f.read().splitlines())
        print(f"[Load] Loaded {len(in_stock)} in-stock molecules.")
    else:
        raise FileNotFoundError(f"In-stock molecules file not found: {emolecules_path}")
    
    # add reactants from request data to in_stock for scoring purposes
    reactants_from_request = data.get('reactants', [])
    if reactants_from_request:
        in_stock.update(reactants_from_request)
        print(f"[Load] Added {len(reactants_from_request)} reactants from request data to in-stock set.")
    
    # canonicalize in-stock molecules for consistent matching
    in_stock = set(canonical_smiles(smiles) for smiles in in_stock if canonical_smiles(smiles) is not None)
    print(f"[Load] Canonicalized in-stock molecules. Total unique in-stock molecules: {len(in_stock)}")
    
    # Load reaction templates
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            templates_raw = json.load(f)
        print(f"[Load] Loaded {len(templates_raw)} templates from '{tpl_filename}'.")
        
        template_list =[]
        for template in tqdm(templates_raw, desc="Init templates"):
            template_list.append(rdchiralReaction(template))
    else:
        raise FileNotFoundError(f"Reaction template file not found: {template_path}")
    
    # Load template conditions
    if os.path.exists(tpl_c_path):
        with open(tpl_c_path, 'r', encoding='utf-8') as f:
            tpl_condition = json.load(f)
        print(f"[Load] Loaded reaction conditions from '{tc_filename}'.")
    else:
        raise FileNotFoundError(f"Reaction condition file not found: {tpl_c_path}")


def format_results_for_llm(target_smiles, results):
    global in_stock
    target_mol = Chem.MolFromSmiles(target_smiles)
    target_mw = rdMolDescriptors.CalcExactMolWt(target_mol) if target_mol else None
    
    output = {
        "target_molecule": {
            "smiles": target_smiles,
            "molecular_weight": round(target_mw, 2) if target_mw else None,
            "in_stock": target_smiles in in_stock
        },
        "retrosynthesis_routes":[]
    }
    
    if not results:
        return output

    for rank, (reactants_smiles_str, data) in enumerate(results.items(), start=1):
        reactants_list =[]
        for r_smiles in reactants_smiles_str.split('.'):
            r_mol = Chem.MolFromSmiles(r_smiles)
            r_mw = rdMolDescriptors.CalcExactMolWt(r_mol) if r_mol else None
            reactants_list.append({
                "smiles": r_smiles,
                "molecular_weight": round(r_mw, 2) if r_mw else None,
                "in_stock": r_smiles in in_stock
            })
            
        route = {
            "route_rank": rank,                   
            "score": round(data['Score'], 4),     
            "reaction_template": data['Template'],
            "reaction_condition": data['Condition'], 
            "all_reactants_in_stock": all(r['in_stock'] for r in reactants_list), 
            "reactants": reactants_list           
        }
        output["retrosynthesis_routes"].append(route)

    return output


def perform_retrosynthesis(data):
    """Execute single-step retrosynthesis."""
    global in_stock, templates_raw, template_list, tpl_condition
    product = data['smiles']
    
    # Get custom weights
    w1, w2, w3, w4 = data.get('weights',[0.1, 0.2, 0.5, 0.0])
    print(f"[Config] Scoring weights applied: w1={w1}, w2={w2}, w3={w3}, w4={w4}")

    results = {}
    result_set = set()
    p_mol_rdchiral = rdchiralReactants(product)
    p_mol = Chem.MolFromSmiles(product)
    
    valid_template_id =[]

    for idx, (template, template_raw) in enumerate(zip(template_list, templates_raw)):
        mapped_curr_results = rdchiralRun(template, p_mol_rdchiral, keep_mapnums=True)
        for r in mapped_curr_results:
            if idx not in valid_template_id:
                valid_template_id.append(idx)
            canonical_r = canonical_smiles(r)
            canonical_r_dict = {r_: canonical_smiles(r_) for r_ in r.split('.')}
            
            if canonical_r in result_set:
                continue
            result_set.add(canonical_r)
            
            r_mols =[Chem.MolFromSmiles(r_) for r_ in r.split('.')]
            rdscore = RDScore(p_mol, r_mols)
            
            # Compute score
            score = 1 * (w1 * CDScore(p_mol, r.split('.')) + 
                         w2 * ASScore(p_mol, canonical_r_dict, in_stock) + 
                         w3 * rdscore + 
                         w4 * 1 / len(mapped_curr_results))
                         
            results[canonical_r] = {
                'Score': score, 
                'Template': template_raw, 
                'Template_id': idx, 
                'Condition': tpl_condition.get(template_raw, None)
            }

            # Sort dictionary by score descending
            results = dict(sorted(results.items(), key=lambda x: x[1]['Score'], reverse=True))
            
    # Keep top 5 results
    results = dict(list(results.items())[:5])

    return format_results_for_llm(product, results)


def main():
    parser = argparse.ArgumentParser(description="Single-step Retrosynthesis CLI Tool")
    
    # Basic parameters
    parser.add_argument("-s", "--smiles", type=str, default="CC(=O)C=C(C)C", help="Target SMILES string (default: CC(=O)C=C(C)C)")
    parser.add_argument("-db", "--database", type=str, default="emol_under_1_carbons", help="In-stock database name (default: emol_under_1_carbons)")
    parser.add_argument("-o", "--output", type=str, default="retro_result.json", help="Output JSON file path (default: retro_result.json)")
    
    # Custom templates
    parser.add_argument("-tpl", "--template", type=str, default="reaction_template.json", help="Reaction template file path (default: reaction_template.json)")
    parser.add_argument("-cond", "--condition", type=str, default="template_condition.json", help="Reaction condition file path (default: template_condition.json)")
    
    # Custom weights
    parser.add_argument("-w", "--weights", type=float, nargs=4, default=[0.1, 0.2, 0.5, 0.0], 
                        help="Four weights for the scoring function separated by space (default: 0.1 0.2 0.5 0.0)")
    
    parser.add_argument("-r","--reactants", type=list, default=[], help="Optional: Provide reactants SMILES to compute scores directly without running retrosynthesis (format: 'reactant1.reactant2', e.g., 'CCO.CN')")
    # 

    args = parser.parse_args()

    request_data = {
        'smiles': args.smiles,
        'databaseName': args.database,
        'template_file': args.template,
        'condition_file': args.condition,
        'weights': args.weights,
        'reactants': args.reactants
    }

    try:
        # 1. Validate SMILES
        if not Chem.MolFromSmiles(args.smiles):
            print(f"[Error] Invalid SMILES format: {args.smiles}")
            return

        print("=========================================")
        print(f"Target Molecule: {args.smiles}")
        print("=========================================")

        # 2. Load globally needed data
        load_global_data(request_data)

        # 3. Perform retrosynthesis
        print("\nRunning retrosynthesis analysis, please wait...")
        result_for_llm = perform_retrosynthesis(request_data)

        # 4. Save results to JSON
        response = {
            "status": "success",
            "message": "Retrosynthesis completed.",
            "data": result_for_llm
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=4)
        
        print(f"\n[Success] Retrosynthesis analysis completed!")
        print(f"JSON result saved to: {args.output}\n")

    except Exception as e:
        print(f"\n[Error] An exception occurred: {str(e)}\n")


if __name__ == '__main__':
    main()
