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
If you want to extract the template by yourself, the following packages need to be installed：
```
pip install rxnmapper
pip install func_timeout
```
When extracting the template, please run:
```
python ./template/preprocessing.py \
    --input_path ./template/preprocessed_data.csv \
    --output_path ./reaction_template.json \
    --condition_path ./template_condition.json
```
