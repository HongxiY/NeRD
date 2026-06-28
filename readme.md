# NeRD: Neuro-Symbolic Rule Distillation for Efficient Ontology-Grounded Chain-of-Thought in Medical Image Diagnosis

Official implementation of the paper
**"NeRD: Neuro-Symbolic Rule Distillation for Efficient Ontology-Grounded Chain-of-Thought in Medical Image Diagnosis"**.

> 🎉 **Accepted at MICCAI 2026.**
> 📄 Paper: [arXiv:2606.15617](https://arxiv.org/abs/2606.15617)

<p align="center">
  <img src="assets/framework.pdf" width="90%" alt="NeRD framework">
</p>

> If the figure does not render, place the framework figure (Fig. 2 of the paper) at `assets/framework.png`.

---

## Overview

Interpretability is essential for trustworthy medical image diagnosis. Existing concept-driven
methods face two limitations: **Concept Bottleneck Models (CBMs)** score *all* predefined concepts
at inference time, while **rationale-based generative approaches** select concepts purely by class
discriminability and can drift away from clinical diagnostic ontologies.

**NeRD** is a neuro-symbolic framework that produces **efficient, ontology-grounded
Multimodal Chain-of-Thought (MCoT) rationales** for medical image diagnosis, *without manually
crafting diagnostic rules*. Instead of prioritizing class discriminability, NeRD first induces an
explicit logical rule set over clinical concepts, then **distills** the activated rules into compact,
case-specific reasoning chains through rule selection, logic simplification, and concept grounding.

The method operates in three steps (see figure above):

1. **Diagnostic Rule Set Construction** — induce a class-conditioned logical rule set over concept
   pairs from data.
2. **Neuro-Symbolic Rule Distillation (NeRD)** — for each case, select the activated supporting
   rules, simplify them into a compact logical expression (CNF), and ground the literals against the
   case's concept annotations.
3. **MCoT Construction** — organize the grounded concepts into supportive / refutational groups and
   render them into an ontology-aligned reasoning chain to fine-tune an MLLM.

**This repository covers Steps 1 and 2.** Step 3 (MCoT templating and MLLM fine-tuning) is described
in the paper and is performed with standard MLLM fine-tuning tooling.

---

## Relationship to prior work

The **rule-extraction component (Step 1)** builds directly on **LogicCBM**:

> Vemuri, D.S., Bellamkonda, G., Pola, A., Balasubramanian, V.N.
> *LogicCBMs: Logic-enhanced concept-based learning.* WACV.

The differentiable logic-gate layer under `difflogic/` and the logic classifier in `nerd_main.py`
are adapted from the official LogicCBM implementation:
**https://github.com/deepikavemuri/LogicCBMs**

---

## Repository structure

```
NeRD-main/
├── nerd_main.py          # Step 1: train the differentiable logic classifier, extract rules
│                         #         and per-case activated rules (adapted from LogicCBM)
├── develop_rule_set.py   # Step 2: rule selection + logic simplification (CNF) + concept grounding
├── difflogic/            # Differentiable logic-gate layer (adapted from LogicCBM / difflogic)
│   ├── difflogic.py
│   ├── functional.py
│   └── __init__.py
├── assets/               # Place framework.png here
└── readme.md
```

---

## Installation

```bash
# Python 3.9+ recommended
pip install torch numpy scikit-learn sympy
```

A CUDA-capable GPU is recommended for Step 1 but not required (the code falls back to CPU).

---

## Data format

Both steps consume a single dataset JSON with `train` / `val` / `test` splits. Each item provides
the concept annotations for one image:

```json
{
  "train": [
    {
      "image": "path/to/image.jpg",
      "binary_label": "malignant",
      "concept": {
        "Erythema": 1,
        "Nodule": 0,
        "Scale": 1
      }
    }
  ],
  "val":  [ ... ],
  "test": [ ... ]
}
```

The 32 clinical concepts used (SkinCon-style) are defined in `CONCEPTS` in `nerd_main.py`.
Concepts absent from the `concept` dict default to `0`.

---

## Usage

### Step 1 — Construct the diagnostic rule set

Trains the differentiable logic-gate classifier, evaluates it, extracts the learned rules, and emits
per-case activated rules.

```bash
python nerd_main.py \
    --data_path path/to/dataset.json \
    --output_dir outputs/ \
    --n_logic_neurons 64 \
    --n_logic_layers 1 \
    --epochs 200
```

Outputs (written to `--output_dir`):

| File | Description |
|------|-------------|
| `best_model.pth`          | Best checkpoint (lowest validation loss) |
| `test_metrics.json`       | Test-set diagnostic metrics |
| `learned_rules.json`      | All extracted logic rules with class-wise weights |
| `case_explanations.json`  | Per-case activated rules and contributions |
| `training_log.txt`        | Full training log |

### Step 2 — Neuro-Symbolic Rule Distillation

Takes the per-case activated rules from Step 1 and distills them into compact, grounded logical
expressions via **rule selection → logic simplification (CNF) → concept grounding**.

```bash
python develop_rule_set.py \
    --input outputs/case_explanations.json \
    --concepts path/to/dataset.json \
    --output outputs/case_simplified.json
```

For each case, the output contains:

- `step1_selected_rules` — rules selected as supporting the ground-truth diagnosis
- `step2_simplified` — the combined, CNF-simplified logical expression
- `step3_grounded` — the final expression after grounding against the case's concept values

The grounded expressions form the symbolic backbone for the **MCoT construction (Step 3)**.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yang2026nerdneurosymbolicruledistillation,
      title={NeRD: Neuro-Symbolic Rule Distillation for Efficient Ontology-Grounded Chain-of-Thought in Medical Image Diagnosis}, 
      author={Hongxi Yang and Yiwen Jiang and Siyuan Yan and Jamie Chow and Eunis Li and Charlotte Poon and Stephanie Fong and Xiangyu Zhao and Deval Mehta and Yasmeen George and Zongyuan Ge},
      year={2026},
      eprint={2606.15617},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.15617}, 
}
```

---

## Acknowledgements

- The rule-extraction component is adapted from
  [LogicCBMs](https://github.com/deepikavemuri/LogicCBMs).
