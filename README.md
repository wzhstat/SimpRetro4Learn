# SimpRetro4Learn
A single-step retrosynthesis algorithm specifically tailored for teaching purposes.

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/SimpRetro4OrganicChemistryB.git
cd SimpRetro4OrganicChemistryB
```

**2. Create environment**
```bash
conda create -n retro_env python=3.9
conda activate retro_env
```

**3. Install dependencies**
Install RDKit, `rdchiral`, and other required packages:
```bash
pip install -r requirements.txt
```

**4. Data Preparation**
Ensure the following necessary data files are placed in the root directory of the project (or specify their paths during inference):
- `reaction_template.json`
- `template_condition.json`
- `<database>.txt` (e.g., `emol_under_0_carbons.txt`)

---

## Inference

Run the python script from your terminal. The basic usage only requires a target SMILES string.

### Basic Run
```bash
python main.py -s "CC(=O)C=C(C)C"
```
*The result will be automatically saved to `retro_result.json`.*

### Chapter-Restricted Prediction

Use only the library-1270 templates assigned to the chapters listed after `-c`:

```bash
python chapter_single_step/predict.py -p "CCOC(=O)CC" -c 5,6,7 -k 10 -o result.json
```

For example, `-c 1,2,3` uses Chapters 1, 2, and 3, whereas `-c 5,6,7` uses only Chapters 5, 6, and 7. It does not automatically include earlier chapters.

| Parameter | Meaning |
|---|---|
| `-p`, `--product` | Product SMILES. The script asks for it if omitted. |
| `-c`, `--chapters` | Comma-separated chapters from 1 to 14. Only the listed chapters are used. |
| `-k`, `--top-k` | Number of predictions to return; default is 10. |
| `-o`, `--output` | Optional JSON output file. |
| `--dataset` | Optional chapter-template dataset path. |
| `--stock` | Optional starting-material file path. |
| `--weights CD AS RD SS` | Four ranking weights; default is `0.1 0.2 0.5 0.0`. |

Chapter-by-reaction-type counts are provided in `chapter_single_step/data/chapter_reaction_template_summary.csv`; `reactions` counts curriculum core reactions, `templates` counts extracted template records, and `unique_templates` counts unique executable SMARTS.

### Advanced Run (Customizing Weights & Templates)
You can customize the scoring weights ($w_1, w_2, w_3, w_4$) and specify your own template/condition files:
```bash
python main.py \
    -s "CC(=O)C=C(C)C" \
    -w 0.2 0.3 0.5 0.0 \
    -tpl "custom_templates.json" \
    -cond "custom_conditions.json" \
    -o "my_output.json" \
    -r ["CC(C)=O"]
```

### Command Line Arguments
| Argument | Short | Default | Description |
| :--- | :---: | :--- | :--- |
| `--smiles` | `-s` | `CC(=O)C=C(C)C` | The target molecule's SMILES string. |
| `--weights` | `-w` | `0.1 0.2 0.5 0.0` | 4 float numbers for the scoring function weights. |
| `--template` | `-tpl` | `reaction_template.json` | Path to the reaction templates JSON file. |
| `--condition`| `-cond`| `template_condition.json`| Path to the reaction conditions JSON file. |
| `--database` | `-db` | `emol_under_0_carbons` | In-stock molecules database prefix name. |
| `--output` | `-o` | `retro_result.json` | Path to save the final JSON output. |
| `--reactants` | `-r` | `[]` | List of reactant SMILES to add to in-stock set for scoring (default: empty list) |

### Scoring Weights Explanation ($w_1, w_2, w_3, w_4$)
* **$w_1$ (Complexity Reduction / CDScore):**
  Favors **convergent synthesis**. It calculates the size disparity between the generated reactants. A higher score is given to reactions that cleave the target molecule into roughly equal-sized fragments, and penalizes linear, one-atom-at-a-time disconnections.
* **$w_2$ (Availability Score / ASScore):**
  Favors **commercially available materials**. It provides a significant bonus if the suggested reactants are found in your `in-stock` database (scaled by the size of the matched fragment). It also penalizes the generation of highly reactive or unstable organometallics (like Mg, Li, Zn) if they are not strictly in-stock.
* **$w_3$ (Ring Disconnection / RDScore):**
  Favors **ring-breaking reactions**. It acts as a binary bonus (1 or 0) given to disconnections that open a ring (meaning the forward reaction is a ring-forming step, like Diels-Alder or macrocyclization), which is a highly strategic move in organic synthesis.
* **$w_4$ (Site Selectivity / Specificity):**
  Favors **chemoselective/regioselective templates**. It is inversely proportional to the number of possible reactive sites found by `rdchiral` for a given template (`1 / len(mapped_results)`). Templates that produce multiple isomeric products (indicating potential selectivity issues) will receive a lower score.

## Preprocessing

### ChemDraw `[R]` Template Workflow

All dependencies are installed once with `pip install -r requirements.txt` during installation.

First, add one generic forward reaction per row to `template_generator/demo_reactions.csv`.

| CSV column | Meaning |
|---|---|
| `reaction_id` | Unique reaction ID. |
| `reaction_name` | Reaction name. |
| `reaction` | ChemDraw reaction, e.g. `[R]C(Cl)=O>>[R]C(O)=O`. |
| `R1` | Allowed classes for `[R]` or `[R1]`, separated by semicolons. |
| `R2` | Allowed classes for `[R2]`; leave blank if unused. |
| `atom_source` | Source of a newly added heavy atom, e.g. `O` for water. |
| `condition` | Reaction conditions. |
| `source` | Textbook or other reaction source. |
| `chapter` | Course chapter. |

Allowed R classes are `methyl`, `primary`, `secondary`, `tertiary`, and `aryl`.

1. Expand the generic reactions:

```bash
python template_generator/generate_templates.py expand \
    --input template_generator/demo_reactions.csv \
    --output template_generator/output/demo/preprocessed_data.csv
```

2. Open `preprocessed_data.csv` and check the generated reactants, products, R-group selections, conditions, sources, and chapters. Correct or remove unsuitable rows before continuing. Keep every `_id` unique.

3. Atom-map the checked reactions and extract radius-1 templates:

```bash
python template_generator/generate_templates.py extract \
    --input template_generator/output/demo/preprocessed_data.csv \
    --template-output reaction_template.json \
    --condition-output template_condition.json
```

The extraction stage adds the mapped reactions and extracted SMARTS to `preprocessed_data.csv`. It also creates:

- `reaction_template.json`: executable retrosynthesis templates.
- `template_condition.json`: reaction conditions associated with each template.

Add `--add-radius-0` to the extraction command when both radius-0 and radius-1 templates are required. Use `--jobs N` to set the number of parallel extraction workers.
