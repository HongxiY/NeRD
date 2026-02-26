"""
Case-level 规则化简工具 - SkinCon 版本

适配 SkinCon 数据集的 binary concept 结构

流程：
1. 筛选支持正确预测的规则 → 保存规则列表
2. to_cnf + simplify_logic → 保存化简后表达式
3. 代入 concept 真值，展开 OR 子句 → 只留下 AND 和 NOT，并化简已知真值
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple, Optional

from sympy import symbols, simplify_logic, Or, And, Not, Implies, true, false
from sympy.logic.boolalg import to_cnf

# ============== Concept 定义 ==============

# SkinCon 的 32 个 binary concepts
CONCEPT_NAMES = [
    "Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule",
    "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy", "Exudate", "Fissure",
    "Induration", "Xerosis", "Telangiectasia", "Scale", "Scar", "Friable",
    "Pedunculated", "Exophytic/Fungating", "Warty/Papillomatous", "Dome-shaped",
    "Brown(Hyperpigmentation)", "White(Hypopigmentation)", "Purple", "Yellow",
    "Black", "Erythema", "Umbilicated"
]

# 原始名称 -> SymPy 安全名称
def to_safe_name(name: str) -> str:
    """将 concept 名称转换为 SymPy 安全的变量名"""
    safe = name.replace("/", "_").replace("(", "_").replace(")", "").replace("-", "_")
    return safe

# SymPy 安全名称 -> 原始名称
SAFE_TO_ORIGINAL = {to_safe_name(name): name for name in CONCEPT_NAMES}
ORIGINAL_TO_SAFE = {name: to_safe_name(name) for name in CONCEPT_NAMES}

# 构建所有符号（使用安全名称）
ALL_SYMBOLS = {name: symbols(to_safe_name(name)) for name in CONCEPT_NAMES}

# 安全名称到符号的映射（用于从表达式中提取）
SAFE_NAME_TO_SYMBOL = {to_safe_name(name): ALL_SYMBOLS[name] for name in CONCEPT_NAMES}


# ============== 规则解析 ==============

def parse_rule_to_sympy(rule_str: str):
    """
    将规则字符串解析为 SymPy 表达式

    Returns:
        (sympy_expr, involved_concepts)
    """
    rule_str = rule_str.strip()
    concepts = []

    # TRUE 常量
    if rule_str == 'TRUE':
        return True, []

    # 构建 concept 匹配模式（支持特殊字符）
    # 匹配如: Vesicle, Exophytic/Fungating, White(Hypopigmentation), Dome-shaped
    concept_pattern = r'[A-Za-z][A-Za-z0-9_/\(\)\-]*'

    # 单个 concept: concept_name
    match = re.match(f'^({concept_pattern})$', rule_str)
    if match:
        concept = match.group(1)
        if concept in ALL_SYMBOLS:
            concepts.append(concept)
            return ALL_SYMBOLS[concept], concepts

    # NOT 单个 concept: NOT concept_name
    match = re.match(f'^NOT ({concept_pattern})$', rule_str)
    if match:
        concept = match.group(1)
        if concept in ALL_SYMBOLS:
            concepts.append(concept)
            return Not(ALL_SYMBOLS[concept]), concepts

    # AND 规则: (A AND B) 或 (A AND NOT B) 或 (NOT A AND B) 或 (NOT A AND NOT B)
    match = re.match(f'^\\((.+) AND (.+)\\)$', rule_str)
    if match:
        left_str, right_str = match.group(1).strip(), match.group(2).strip()

        # 解析左边
        if left_str.startswith('NOT '):
            left_concept = left_str[4:]
            left_expr = Not(ALL_SYMBOLS[left_concept])
        else:
            left_concept = left_str
            left_expr = ALL_SYMBOLS[left_concept]
        concepts.append(left_concept)

        # 解析右边
        if right_str.startswith('NOT '):
            right_concept = right_str[4:]
            right_expr = Not(ALL_SYMBOLS[right_concept])
        else:
            right_concept = right_str
            right_expr = ALL_SYMBOLS[right_concept]
        concepts.append(right_concept)

        return And(left_expr, right_expr), concepts

    # OR 规则: (A OR B)
    match = re.match(f'^\\(({concept_pattern}) OR ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Or(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept]), concepts

    # IMPLIES 规则: (A -> B)
    match = re.match(f'^\\(({concept_pattern}) -> ({concept_pattern})\\)$', rule_str)
    if match:
        ante, cons = match.group(1), match.group(2)
        concepts = [ante, cons]
        return Implies(ALL_SYMBOLS[ante], ALL_SYMBOLS[cons]), concepts

    # NOT (A OR B)
    match = re.match(f'^NOT \\(({concept_pattern}) OR ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Not(Or(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept])), concepts

    # NOT (A AND B)
    match = re.match(f'^NOT \\(({concept_pattern}) AND ({concept_pattern})\\)$', rule_str)
    if match:
        left_concept, right_concept = match.group(1), match.group(2)
        concepts = [left_concept, right_concept]
        return Not(And(ALL_SYMBOLS[left_concept], ALL_SYMBOLS[right_concept])), concepts

    # 无法解析
    print(f"Warning: 无法解析规则 '{rule_str}'")
    return None, []


# ============== 规则筛选 ==============

def select_supporting_rules(activated_rules: List[dict], true_label: int) -> List[dict]:
    """根据 true_label 筛选支持正确预测的规则"""
    selected = []

    for rule in activated_rules:
        if rule['rule'] == 'TRUE':
            continue

        w_benign = rule['weight_benign']
        w_malignant = rule['weight_malignant']

        if true_label == 0:  # benign
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
        else:  # malignant
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


# ============== Step 2: SymPy 化简 ==============

def sympy_simplify(rules: List[dict], verbose: bool = True):
    """
    Step 2: 逐步添加规则并化简

    Returns:
        simplified_expr (SymPy expression or None)
    """
    if verbose:
        print("\n【Step 2】SymPy 化简 (CNF) - 逐步化简版")
        print("-" * 60)

    # 解析规则
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
            print("  无有效规则")
        return None

    # 逐步添加并化简
    if verbose:
        print(f"\n  逐步添加并化简 ({len(sympy_exprs)} 个表达式):")

    # 第一个表达式先转 CNF
    combined = to_cnf(sympy_exprs[0])
    combined = simplify_logic(combined, form='cnf')

    if verbose:
        print(f"    [1] {combined}")

    # 逐步添加剩余表达式
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
            print(f"    [{i}] 子句数: {num_clauses}")

    if verbose:
        print(f"\n  化简后 (CNF):")
        print(f"    {combined}")

    return combined


# ============== Step 3: 代入真值展开 OR ==============

def get_concept_truth_values(case_concepts: dict) -> dict:
    """
    从 case 的 concept 数据中生成所有 concept 的真值映射

    Args:
        case_concepts: {"Vesicle": 0, "Papule": 1, ...}

    Returns:
        {concept_name: True/False}
    """
    truth_values = {}

    for concept_name in CONCEPT_NAMES:
        if concept_name in case_concepts:
            truth_values[concept_name] = (case_concepts[concept_name] == 1)
        else:
            # 如果缺失，默认 False
            truth_values[concept_name] = False

    return truth_values


def evaluate_literal(lit, truth_values: dict) -> bool:
    """
    计算单个 literal 的真值

    Args:
        lit: SymPy 符号或 Not(符号)
        truth_values: {concept_name: True/False}

    Returns:
        True/False
    """
    if lit.is_Symbol:
        # 正向 literal
        concept_name = str(lit)
        # 需要从安全名称转回原始名称
        if concept_name in SAFE_TO_ORIGINAL:
            concept_name = SAFE_TO_ORIGINAL[concept_name]
        return truth_values.get(concept_name, False)
    elif lit.func == Not and lit.args[0].is_Symbol:
        # 负向 literal: NOT A
        concept_name = str(lit.args[0])
        if concept_name in SAFE_TO_ORIGINAL:
            concept_name = SAFE_TO_ORIGINAL[concept_name]
        return not truth_values.get(concept_name, False)
    else:
        # 复杂情况，不应该出现
        return None


def resolve_or_clause(clause, truth_values: dict, verbose: bool = False):
    """
    解析 OR 子句，根据真值选择一个为 True 的 literal

    Args:
        clause: OR 子句 (A | B | ...)
        truth_values: {concept_name: True/False}

    Returns:
        选中的 literal，或 None（如果整个 OR 为 False）
    """
    if clause.func != Or:
        return clause

    # 遍历 OR 的所有子项
    for lit in clause.args:
        val = evaluate_literal(lit, truth_values)
        if val is True:
            if verbose:
                print(f"    OR 子句 {clause} → 选择 {lit}")
            return lit

    # 如果都是 False，返回 None（异常情况）
    if verbose:
        print(f"    Warning: OR 子句 {clause} 所有项都为 False")
    return None


def ground_expression(expr, truth_values: dict, verbose: bool = False):
    """
    Step 3: 只展开 OR 子句，其他 literal 保持原样

    Args:
        expr: SymPy 表达式 (CNF 形式)
        truth_values: {concept_name: True/False}

    Returns:
        (literals_list, anomalies_list)
    """
    if verbose:
        print("\n【Step 3】展开 OR 子句")
        print("-" * 60)
        print(f"  输入表达式: {expr}")

    if expr is None or expr is True:
        if verbose:
            print("  表达式为 TRUE，无需处理")
        return [], []

    if expr is False:
        if verbose:
            print("  表达式为 FALSE")
        return [false], []

    anomalies = []
    result_literals = []

    # 获取所有子句（CNF 是 AND 连接的子句）
    if expr.func == And:
        clauses = list(expr.args)
    else:
        # 单个子句
        clauses = [expr]

    if verbose:
        print(f"  共 {len(clauses)} 个子句")

    for clause in clauses:
        if clause.func == Or:
            # OR 子句：根据真值选择一个
            selected = resolve_or_clause(clause, truth_values, verbose=verbose)
            if selected is not None:
                result_literals.append(selected)
            else:
                anomalies.append(f"OR 子句全为 False: {clause}")
        else:
            # 非 OR 子句（单个 literal）：保持原样
            result_literals.append(clause)
            if verbose:
                print(f"    保留 literal: {clause}")

    # 去重：将 literal 转为字符串比较，保持顺序
    seen = set()
    unique_literals = []
    for lit in result_literals:
        lit_str = str(lit)
        if lit_str not in seen:
            seen.add(lit_str)
            unique_literals.append(lit)

    if verbose:
        if len(result_literals) != len(unique_literals):
            print(f"\n  去重: {len(result_literals)} → {len(unique_literals)} 个")
        print(f"\n  结果 literals ({len(unique_literals)} 个):")
        for lit in unique_literals:
            print(f"    {lit}")

    return unique_literals, anomalies


def build_final_expression(literals: List):
    """将 literals 列表构建为 SymPy AND 表达式"""
    if not literals:
        return True

    if len(literals) == 1:
        return literals[0]

    return And(*literals)


def format_expr_string(expr) -> str:
    """将 SymPy 表达式格式化为可读字符串"""
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

    # 将安全名称转回原始名称
    for safe_name, original_name in SAFE_TO_ORIGINAL.items():
        if safe_name != original_name:
            s = s.replace(safe_name, original_name)

    return s


# ============== 处理单个 case ==============

def process_case(case: dict, case_concepts: dict = None, verbose: bool = False) -> dict:
    """处理单个 case，返回三步的结果"""
    true_label = case['true_label']
    activated_rules = case['activated_rules']
    label_name = "malignant" if true_label == 1 else "benign"

    # ========== Step 1: 筛选规则 ==========
    selected_rules = select_supporting_rules(activated_rules, true_label)
    step1_rules = [r['rule'] for r in selected_rules]

    if verbose:
        print("\n【Step 1】筛选支持正确预测的规则")
        print("-" * 60)
        print(f"  原始规则数: {len(activated_rules)}")
        print(f"  筛选后规则数: {len(selected_rules)}")
        for r in step1_rules:
            print(f"    - {r}")

    # ========== Step 2: SymPy 化简 ==========
    step2_expr = sympy_simplify(selected_rules, verbose=verbose)
    step2_str = format_expr_string(step2_expr)

    # ========== Step 3: 代入真值展开 OR ==========
    step3_literals = []
    step3_str = step2_str
    anomalies = []

    if case_concepts is not None:
        truth_values = get_concept_truth_values(case_concepts)
        step3_literals, anomalies = ground_expression(step2_expr, truth_values, verbose=verbose)
        step3_expr = build_final_expression(step3_literals)
        step3_str = format_expr_string(step3_expr)

    # 组装结果
    result = {
        'image': case['image'],
        'true_label': true_label,
        'true_label_name': label_name,
        'pred_label': case['pred_label'],
        'pred_prob_benign': case['pred_prob_benign'],
        'pred_prob_malignant': case['pred_prob_malignant'],

        # Step 1: 筛选后的规则
        'step1_selected_rules': step1_rules,

        # Step 2: SymPy 化简后的表达式
        'step2_simplified': step2_str,

        # Step 3: 代入真值后的表达式 (展开 OR, 化简 NOT)
        'step3_grounded': step3_str,
    }

    # 如果有异常，记录
    if anomalies:
        result['step3_anomalies'] = anomalies

    return result


# ============== 处理文件 ==============

def load_concept_data(concept_path: str) -> dict:
    """
    加载 concept 真值数据，建立 image -> concepts 的映射
    """
    print(f"加载 concept 数据: {concept_path}")

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

    print(f"  加载了 {len(image_to_concepts)} 个样本的 concept 数据")

    return image_to_concepts


def process_file(input_path: str, output_path: str, concept_path: str = None, verbose_first_n: int = 2):
    """处理整个 JSON 文件"""
    print(f"读取文件: {input_path}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    # 加载 concept 真值数据
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
        print(f"处理 {split} 集 ({len(data[split])} 个样本)")
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
                print(f"  Warning: 找不到 {case['image']} 的 concept 数据")

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
                print(f"  已处理 {i + 1}/{len(data[split])}")

        # 统计
        total = len(result[split])
        if total > 0:
            avg_rules = sum(len(c['step1_selected_rules']) for c in result[split]) / total

            print(f"\n{split} 统计:")
            print(f"  平均筛选后规则数: {avg_rules:.1f}")

    # 保存结果
    print(f"\n保存结果到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 报告异常
    if total_anomalies:
        print(f"\n{'=' * 60}")
        print(f"异常报告 ({len(total_anomalies)} 个 case)")
        print(f"{'=' * 60}")
        for anomaly in total_anomalies:
            print(f"  [{anomaly['split']}] {anomaly['image']}")
            for a in anomaly['anomalies']:
                print(f"    - {a}")

    print("\n完成!")


# ============== 主函数 ==============

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Case-level 规则化简工具 - SkinCon 版本')
    parser.add_argument('--input', type=str,
                        default='/home/user01/storage/result/DCR/f17k_binary/logicCBM_noxor_2class_new/neuron32_best_val_loss/case_explanations.json',
                        help='输入 JSON 文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径')
    parser.add_argument('--concepts', type=str,
                        default='/home/user01/storage/data/MAKE/MAKE_Downstreams/skincon/resplit_skincon_fitz_only.json',
                        help='Concept 真值 JSON 文件路径')
    parser.add_argument('--verbose_n', type=int, default=2,
                        help='详细输出前 N 个 case 的化简过程')

    args = parser.parse_args()

    if args.output is None:
        input_dir = os.path.dirname(args.input)
        args.output = os.path.join(input_dir, 'case_explanations_simplified.json')

    process_file(args.input, args.output, concept_path=args.concepts, verbose_first_n=args.verbose_n)
