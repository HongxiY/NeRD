import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional

from sympy import symbols, simplify_logic, Or, And, Not, Implies, true, false
from sympy.logic.boolalg import to_cnf

CONCEPT_NAMES = [
    "Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule",
    "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy", "Exudate", "Fissure",
    "Induration", "Xerosis", "Telangiectasia", "Scale", "Scar", "Friable",
    "Pedunculated", "Exophytic/Fungating", "Warty/Papillomatous", "Dome-shaped",
    "Brown(Hyperpigmentation)", "White(Hypopigmentation)", "Purple", "Yellow",
    "Black", "Erythema", "Umbilicated"
]


def to_safe_name(name: str) -> str:
    safe = name.replace("/", "_").replace("(", "_").replace(")", "").replace("-", "_")
    return safe

SAFE_TO_ORIGINAL = {to_safe_name(name): name for name in CONCEPT_NAMES}
ORIGINAL_TO_SAFE = {name: to_safe_name(name) for name in CONCEPT_NAMES}

ALL_SYMBOLS = {name: symbols(to_safe_name(name)) for name in CONCEPT_NAMES}

SAFE_NAME_TO_SYMBOL = {to_safe_name(name): ALL_SYMBOLS[name] for name in CONCEPT_NAMES}



def parse_rule_to_sympy(rule_str: str):
    rule_str = rule_str.strip()
    concepts = []

    if rule_str == 'TRUE':
        return True, []

    concept_pattern = r'[A-Za-z][A-Za-z0-9_/\(\)\-]*'

    match = re.match(f'^({concept_pattern})$', rule_str)
    if match:
        concept = match.group(1)
        if concept in ALL_SYMBOLS:
            concepts.append(concept)
            return ALL_SYMBOLS[concept], concepts

    match = re.match(f'^NOT ({concept_pattern})$', rule_str)
    if match:
        concept = match.group(1)
        if concept in ALL_SYMBOLS:
            concepts.append(concept)
            return Not(ALL_SYMBOLS[concept]), concepts

    match = re.match(f'^\\((.+) AND (.+)\\)$', rule_str)
    if match:
        left_str, right_str = match.group(1).strip(), match.group(2).strip()

        if left_str.startswith('NOT '):
            left_concept = left_str[4:]
            left_expr = Not(ALL_SYMBOLS[left_concept])
        else:
            left_concept = left_str
            left_expr = ALL_SYMBOLS[left_concept]
        concepts.append(left_concept)

        if right_str.startswith('NOT '):
            right_concept = right_str[4:]
            right_expr = Not(ALL_SYMBOLS[right_concept])
        else:
            right_concept = right_str
            right_expr = ALL_SYMBOLS[right_concept]
        concepts.append(right_concept)

        return And(left_expr, right_expr), concepts

    match = re.match(f'^\\(({concept_pattern}) OR ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Or(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept]), concepts

    match = re.match(f'^\\(({concept_pattern}) -> ({concept_pattern})\\)$', rule_str)
    if match:
        ante, cons = match.group(1), match.group(2)
        concepts = [ante, cons]
        return Implies(ALL_SYMBOLS[ante], ALL_SYMBOLS[cons]), concepts

    match = re.match(f'^NOT \\(({concept_pattern}) OR ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Not(Or(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept])), concepts

    match = re.match(f'^NOT \\(({concept_pattern}) AND ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Not(And(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept])), concepts

    print(f"Warning: Cannot deal with '{rule_str}'")
    return None, []



def select_supporting_rules(activated_rules: List[dict], true_label: int) -> List[dict]:
    selected = []

    for rule in activated_rules:
        if rule['rule'] == 'TRUE':
            continue

        w_benign = rule['weight_benign']
        w_malignant = rule['weight_malignant']

        if true_label == 0:
            if w_benign > 0 and w_malignant < 0:
                selected.append(rule)
            elif w_benign < 0 and w_malignant > 0:
                pass
            else:
                if abs(w_benign) > abs(w_malignant):
                    if w_benign > 0:
                        selected.append(rule)
                else:
                    if w_malignant < 0:
                        selected.append(rule)
        else: 
            if w_malignant > 0 and w_benign < 0:
                selected.append(rule)
            elif w_malignant < 0 and w_benign > 0:
                pass
            else:
                if abs(w_malignant) > abs(w_benign):
                    if w_malignant > 0:
                        selected.append(rule)
                else:
                    if w_benign < 0:
                        selected.append(rule)

    return selected



def sympy_simplify(rules: List[dict], verbose: bool = True):

    if verbose:
        print("\n To CNF")
        print("-" * 60)

    sympy_exprs = []
    for rule in rules:
        rule_str = rule['rule']
        expr, concepts = parse_rule_to_sympy(rule_str)

        if expr is not None and expr is not True:
            sympy_exprs.append(expr)
            if verbose:
                print(f"  {rule_str} → {expr}")

    if not sympy_exprs:
        if verbose:
            print("No valid rule")
        return None

    if verbose:
        print(f"\n  Deal with ({len(sympy_exprs)}):")

    combined = to_cnf(sympy_exprs[0])
    combined = simplify_logic(combined, form='cnf')

    if verbose:
        print(f"    [1] {combined}")

    for i, expr in enumerate(sympy_exprs[1:], 2):
        expr_cnf = to_cnf(expr)
        combined = And(combined, expr_cnf)
        combined = simplify_logic(combined, form='cnf')

        if verbose:
            if combined.func == And:
                num_clauses = len(combined.args)
            elif combined is True or combined is False:
                num_clauses = 0
            else:
                num_clauses = 1
            print(f"    [{i}] Sub number: {num_clauses}")

    if verbose:
        print(f"\n  CNF:")
        print(f"    {combined}")

    return combined


def get_concept_truth_values(case_concepts: dict) -> dict:
 
    truth_values = {}

    for concept_name in CONCEPT_NAMES:
        if concept_name in case_concepts:
            truth_values[concept_name] = (case_concepts[concept_name] == 1)
        else:
            truth_values[concept_name] = False

    return truth_values


def evaluate_literal(lit, truth_values: dict) -> bool:

    if lit.is_Symbol:

        concept_name = str(lit)
        if concept_name in SAFE_TO_ORIGINAL:
            concept_name = SAFE_TO_ORIGINAL[concept_name]
        return truth_values.get(concept_name, False)
    elif lit.func == Not and lit.args[0].is_Symbol:
        concept_name = str(lit.args[0])
        if concept_name in SAFE_TO_ORIGINAL:
            concept_name = SAFE_TO_ORIGINAL[concept_name]
        return not truth_values.get(concept_name, False)
    else:
        return None


def resolve_or_clause(clause, truth_values: dict, verbose: bool = False):
    if clause.func != Or:
        return clause

    for lit in clause.args:
        val = evaluate_literal(lit, truth_values)
        if val is True:
            if verbose:
                print(f"    OR {clause} → choose {lit}")
            return lit

    if verbose:
        print(f"    Warning: OR {clause} all are False")
    return None


def ground_expression(expr, truth_values: dict, verbose: bool = False):

    if verbose:
        print("\n Grounding")
        print("-" * 60)
        print(f"  Input set: {expr}")

    if expr is None or expr is True:
        if verbose:
            print("  TRUE")
        return [], []

    if expr is False:
        if verbose:
            print("  FALSE")
        return [false], []

    anomalies = []
    result_literals = []

    if expr.func == And:
        clauses = list(expr.args)
    else:

        clauses = [expr]

    if verbose:
        print(f"  total {len(clauses)} ")

    for clause in clauses:
        if clause.func == Or:
            selected = resolve_or_clause(clause, truth_values, verbose=verbose)
            if selected is not None:
                result_literals.append(selected)
            else:
                anomalies.append(f"OR all are False: {clause}")
        else:
            result_literals.append(clause)
            if verbose:
                print(f"    keep literal: {clause}")

    seen = set()
    unique_literals = []
    for lit in result_literals:
        lit_str = str(lit)
        if lit_str not in seen:
            seen.add(lit_str)
            unique_literals.append(lit)

    if verbose:
        if len(result_literals) != len(unique_literals):
            print(f"\n remove : {len(result_literals)} → {len(unique_literals)} 个")
        print(f"\n  result literals ({len(unique_literals)} 个):")
        for lit in unique_literals:
            print(f"    {lit}")

    return unique_literals, anomalies


def build_final_expression(literals: List):
    if not literals:
        return True

    if len(literals) == 1:
        return literals[0]

    return And(*literals)


def format_expr_string(expr) -> str:
    if expr is None:
        return "TRUE"
    if expr is True:
        return "TRUE"
    if expr is False:
        return "FALSE"

    s = str(expr)
    s = s.replace(' & ', ' AND ')
    s = s.replace(' | ', ' OR ')
    s = s.replace('~', 'NOT ')

    for safe_name, original_name in SAFE_TO_ORIGINAL.items():
        if safe_name != original_name:
            s = s.replace(safe_name, original_name)

    return s



def process_case(case: dict, case_concepts: dict = None, verbose: bool = False) -> dict:
    true_label = case['true_label']
    activated_rules = case['activated_rules']
    label_name = "malignant" if true_label == 1 else "benign"

    selected_rules = select_supporting_rules(activated_rules, true_label)
    step1_rules = [r['rule'] for r in selected_rules]

    if verbose:
        print("\n Rule selection")
        print("-" * 60)
        print(f"  Before: {len(activated_rules)}")
        print(f"  After: {len(selected_rules)}")
        for r in step1_rules:
            print(f"    - {r}")

    step2_expr = sympy_simplify(selected_rules, verbose=verbose)
    step2_str = format_expr_string(step2_expr)

    step3_literals = []
    step3_str = step2_str
    anomalies = []

    if case_concepts is not None:
        truth_values = get_concept_truth_values(case_concepts)
        step3_literals, anomalies = ground_expression(step2_expr, truth_values, verbose=verbose)
        step3_expr = build_final_expression(step3_literals)
        step3_str = format_expr_string(step3_expr)

    result = {
        'image': case['image'],
        'true_label': true_label,
        'true_label_name': label_name,
        'pred_label': case['pred_label'],
        'pred_prob_benign': case['pred_prob_benign'],
        'pred_prob_malignant': case['pred_prob_malignant'],

        'step1_selected_rules': step1_rules,

        'step2_simplified': step2_str,

        'step3_grounded': step3_str,
    }

    if anomalies:
        result['step3_anomalies'] = anomalies

    return result


def load_concept_data(concept_path: str) -> dict:

    print(f"Load: {concept_path}")

    with open(concept_path, 'r') as f:
        data = json.load(f)

    image_to_concepts = {}

    for split in ['train', 'val', 'test']:
        if split not in data:
            continue

        for item in data[split]:
            image_path = item['image']
            concepts = item.get('concept', {})
            image_to_concepts[image_path] = concepts

    print(f"  Load {len(image_to_concepts)} concepts")

    return image_to_concepts


def process_file(input_path: str, output_path: str, concept_path: str = None, verbose_first_n: int = 2):
    print(f"Read: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    image_to_concepts = {}
    if concept_path is not None:
        image_to_concepts = load_concept_data(concept_path)

    result = {}
    case_count = 0
    total_anomalies = []

    for split in ['train', 'val', 'test']:
        if split not in data:
            continue

        print(f"\n{'=' * 60}")
        print(f"{split}  ({len(data[split])} )")
        print(f"{'=' * 60}")

        result[split] = []

        for i, case in enumerate(data[split]):
            verbose = (case_count < verbose_first_n)

            if verbose:
                print(f"\n\n{'#' * 60}")
                print(f"Case {i + 1}: {case['image'].split('/')[-1]}")
                print(f"true_label={case['true_label']}, pred_label={case['pred_label']}")
                print(f"{'#' * 60}")

            case_concepts = image_to_concepts.get(case['image'], None)

            if case_concepts is None and image_to_concepts:
                print(f"  Warning: cannot find {case['image']} concepts")

            processed = process_case(case, case_concepts=case_concepts, verbose=verbose)
            result[split].append(processed)

            if 'step3_anomalies' in processed:
                total_anomalies.append({
                    'split': split,
                    'image': case['image'],
                    'anomalies': processed['step3_anomalies']
                })

            case_count += 1

            if (i + 1) % 100 == 0:
                print(f"  Done {i + 1}/{len(data[split])}")

        total = len(result[split])
        if total > 0:
            avg_rules = sum(len(c['step1_selected_rules']) for c in result[split]) / total

            print(f"\n{split} :")
            print(f"  Avg rule number: {avg_rules:.1f}")

    print(f"\n Save to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if total_anomalies:
        print(f"\n{'=' * 60}")
        print(f"Warning ({len(total_anomalies)} 个 case)")
        print(f"{'=' * 60}")
        for anomaly in total_anomalies:
            print(f"  [{anomaly['split']}] {anomaly['image']}")
            for a in anomaly['anomalies']:
                print(f"    - {a}")

    print("\n Done")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Experiment xxx')
    parser.add_argument('--input', type=str,
                        default=None,
                        help='Path to input data')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output data')
    parser.add_argument('--concepts', type=str,
                        default=None,
                        help='Path to concept annotation file')
    parser.add_argument('--verbose_n', type=int, default=2,
                        help='Detailed output for first xx cases')

    args = parser.parse_args()

    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, 'case_explanations_simplified.json')

    process_file(args.input, args.output, concept_path=args.concepts, verbose_first_n=args.verbose_n)
