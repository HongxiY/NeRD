import os
import sys
import json
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, classification_report
)



def setup_logging(output_dir):
    log_path = os.path.join(output_dir, 'training_log.txt')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def log_print(msg=""):
    """Print and log message."""
    logging.info(msg)


CONCEPTS = [
    "Vesicle", "Papule", "Macule", "Plaque", "Pustule", "Bulla", "Patch", "Nodule",
    "Ulcer", "Crust", "Erosion", "Excoriation", "Atrophy", "Exudate", "Fissure",
    "Induration", "Xerosis", "Telangiectasia", "Scale", "Scar", "Friable",
    "Pedunculated", "Exophytic/Fungating", "Warty/Papillomatous", "Dome-shaped",
    "Brown(Hyperpigmentation)", "White(Hypopigmentation)", "Purple", "Yellow",
    "Black", "Erythema", "Umbilicated"
]

N_CONCEPTS = len(CONCEPTS)



class SkinConConceptDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        concepts = torch.zeros(N_CONCEPTS, dtype=torch.float32)
        for i, concept_name in enumerate(CONCEPTS):
            concepts[i] = float(item['concept'].get(concept_name, 0))

        label = 1 if item["binary_label"] == "malignant" else 0
        label = torch.tensor(label, dtype=torch.long)

        image_path = item.get("image", f"case_{idx}")

        return concepts, label, image_path


def load_dataset(json_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    train_dataset = SkinConConceptDataset(dataset['train'])
    val_dataset = SkinConConceptDataset(dataset['val'])
    test_dataset = SkinConConceptDataset(dataset['test'])

    return train_dataset, val_dataset, test_dataset



def generate_concept_pairs(n_pairs, device='cuda'):

    valid_pairs = []

    for i in range(N_CONCEPTS):
        for j in range(N_CONCEPTS):
            if i != j:  # Exclude self-pairs
                valid_pairs.append((i, j))

    log_print(f"Total valid pairs (excluding self-pairs): {len(valid_pairs)}")

    if n_pairs > len(valid_pairs):
        log_print(f"Warning: n_pairs ({n_pairs}) > valid pairs ({len(valid_pairs)}), using all valid pairs")
        selected_pairs = valid_pairs
    else:
        selected_pairs = random.sample(valid_pairs, n_pairs)

    concept_pairs = torch.tensor(selected_pairs, dtype=torch.int64, device=device)
    return concept_pairs


class LogicClassifier(nn.Module):

    def __init__(self, n_concepts, n_logic_neurons, n_logic_layers=1,
                 concept_pairs=None, device='cuda', fixed_gates=False):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_logic_neurons = n_logic_neurons
        self.device = device

        from difflogic.difflogic_noxor import LogicLayer
        layers = []

        layers.append(LogicLayer(
            in_dim=n_concepts,
            out_dim=n_logic_neurons,
            device=device,
            connections='correlated',
            concept_pairs=concept_pairs,
            fixed_gates=fixed_gates
        ))

        for _ in range(n_logic_layers - 1):
            layers.append(LogicLayer(
                in_dim=n_logic_neurons,
                out_dim=n_logic_neurons,
                device=device,
                connections='random',
                fixed_gates=fixed_gates
            ))

        self.logic_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(n_logic_neurons, 2)

    def forward(self, x):
        x = self.logic_layers(x)
        logits = self.classifier(x)
        return logits

    def get_logic_outputs(self, x):
        return self.logic_layers(x)



def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    for concepts, labels, _ in dataloader:
        concepts = concepts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(concepts)  # (batch, 2)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss

    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for concepts, labels, _ in dataloader:
            concepts = concepts.to(device)
            labels = labels.to(device)

            logits = model(concepts)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs.extend(probs[:, 1].cpu().numpy()) 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = avg_loss

    return metrics, all_labels, all_preds, all_probs


def compute_metrics(labels, preds, probs):

    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    metrics = {
        'acc': accuracy_score(labels, preds),
        'balanced_acc': balanced_accuracy_score(labels, preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,  # Recall for positive class
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'precision': precision_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

    try:
        metrics['auc'] = roc_auc_score(labels, probs)
    except:
        metrics['auc'] = 0.0

    return metrics



def gate_to_str(concept_a, concept_b, gate_idx):
    gate_names = {
        0: "FALSE",
        1: f"({concept_a} AND {concept_b})",
        2: f"({concept_a} AND NOT {concept_b})",
        3: f"{concept_a}",
        4: f"(NOT {concept_a} AND {concept_b})",
        5: f"{concept_b}",
        6: f"({concept_a} XOR {concept_b})",
        7: f"({concept_a} OR {concept_b})",
        8: f"NOT ({concept_a} OR {concept_b})",
        9: f"({concept_a} XNOR {concept_b})",
        10: f"NOT {concept_b}",
        11: f"({concept_b} -> {concept_a})",
        12: f"NOT {concept_a}",
        13: f"({concept_a} -> {concept_b})",
        14: f"NOT ({concept_a} AND {concept_b})",
        15: "TRUE"
    }
    return gate_names.get(gate_idx, f"UNKNOWN({gate_idx})")


def get_all_rules(model):
    logic_layer = model.logic_layers[0]

    gate_types = torch.argmax(logic_layer.weights, dim=-1).cpu().numpy()

    indices_a = logic_layer.indices[0].cpu().numpy()
    indices_b = logic_layer.indices[1].cpu().numpy()

    classifier_weights = model.classifier.weight.data.cpu().numpy()

    rules = []
    for i in range(len(gate_types)):
        concept_a = CONCEPTS[indices_a[i]]
        concept_b = CONCEPTS[indices_b[i]]
        gate_type = int(gate_types[i])
        weight_benign = float(classifier_weights[0, i])
        weight_malignant = float(classifier_weights[1, i])

        rule_str = gate_to_str(concept_a, concept_b, gate_type)
        rules.append({
            'rule_idx': i,
            'rule': rule_str,
            'concept_a': concept_a,
            'concept_b': concept_b,
            'gate_type': gate_type,
            'weight_benign': weight_benign,
            'weight_malignant': weight_malignant
        })

    return rules


def extract_rules(model, top_k=10):
    
    log_print("\n" + "=" * 60)
    log_print("Extracted Logic Rules")
    log_print("=" * 60)

    rules = get_all_rules(model)

    rules_sorted = sorted(rules, key=lambda x: x['weight_malignant'], reverse=True)

    log_print(f"\nTop {top_k} Rules with highest malignant weight:")
    log_print("-" * 80)
    log_print(f"{'Rule':<55} {'w_malig':>10} {'w_benign':>10}")
    log_print("-" * 80)
    for rule in rules_sorted[:top_k]:
        log_print(f"{rule['rule']:<55} {rule['weight_malignant']:>+10.4f} {rule['weight_benign']:>+10.4f}")

    log_print(f"\nTop {top_k} Rules with lowest malignant weight (predict benign):")
    log_print("-" * 80)
    log_print(f"{'Rule':<55} {'w_malig':>10} {'w_benign':>10}")
    log_print("-" * 80)
    for rule in rules_sorted[-top_k:]:
        log_print(f"{rule['rule']:<55} {rule['weight_malignant']:>+10.4f} {rule['weight_benign']:>+10.4f}")

    return rules_sorted



def explain_cases(model, dataloader, device):
    model.eval()
    rules = get_all_rules(model)

    explanations = []

    with torch.no_grad():
        for batch_idx, (concepts, labels, image_paths) in enumerate(dataloader):
            concepts = concepts.to(device)

            logic_outputs = model.get_logic_outputs(concepts)  # (batch, n_neurons)

            logits = model.classifier(logic_outputs)  # (batch, 2)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            logic_outputs_np = logic_outputs.cpu().numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(concepts)):
                case_explanation = {
                    'image': image_paths[i],
                    'true_label': int(labels[i].item()),
                    'pred_label': int(preds[i].item()),
                    'pred_prob_benign': float(probs_np[i, 0]),
                    'pred_prob_malignant': float(probs_np[i, 1]),
                    'activated_rules': []
                }

                for rule_idx, rule in enumerate(rules):
                    output = float(logic_outputs_np[i, rule_idx])

                    if output > 0.5: 
                        contribution_benign = output * rule['weight_benign']
                        contribution_malignant = output * rule['weight_malignant']
                        case_explanation['activated_rules'].append({
                            'rule_idx': rule_idx,
                            'rule': rule['rule'],
                            'output': round(output, 4),
                            'weight_benign': round(rule['weight_benign'], 4),
                            'weight_malignant': round(rule['weight_malignant'], 4),
                            'contribution_benign': round(contribution_benign, 4),
                            'contribution_malignant': round(contribution_malignant, 4)
                        })

                # Sort activated rules by absolute contribution to malignant
                case_explanation['activated_rules'].sort(
                    key=lambda x: abs(x['contribution_malignant']), reverse=True
                )

                explanations.append(case_explanation)

    return explanations


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    setup_logging(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_print(f"Using device: {device}")

    log_print(f"Output directory: {args.output_dir}")

    log_print(f"\nLoading dataset from {args.data_path}...")
    train_dataset, val_dataset, test_dataset = load_dataset(args.data_path)

    log_print(f"  Train: {len(train_dataset)} samples")
    log_print(f"  Val: {len(val_dataset)} samples")
    log_print(f"  Test: {len(test_dataset)} samples")

    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    log_print(f"  Train class distribution: malignant={int(n_pos)}, benign={int(n_neg)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    log_print(f"\nGenerating {args.n_logic_neurons} concept pairs (excluding self-pairs)...")
    concept_pairs = generate_concept_pairs(args.n_logic_neurons, device=device)

    log_print(f"\nCreating model...")
    log_print(f"  n_concepts: {N_CONCEPTS}")
    log_print(f"  n_logic_neurons: {args.n_logic_neurons}")
    log_print(f"  n_logic_layers: {args.n_logic_layers}")
    log_print(f"  fixed_gates: {args.fixed_gates}")

    model = LogicClassifier(
        n_concepts=N_CONCEPTS,
        n_logic_neurons=args.n_logic_neurons,
        n_logic_layers=args.n_logic_layers,
        concept_pairs=concept_pairs,
        device=device,
        fixed_gates=args.fixed_gates
    ).to(device)

    if n_pos > 0:
        class_weights = torch.tensor([1.0, n_neg / n_pos], device=device)
    else:
        class_weights = torch.tensor([1.0, 1.0], device=device)
    log_print(f"  class_weights: [1.0, {class_weights[1].item():.2f}]")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    log_print(f"\nStarting training for {args.epochs} epochs...")
    best_valid_loss = float('inf')
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(val_metrics['loss'])

        if epoch % args.print_every == 0:
            log_print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
                      f"AUC: {train_metrics['auc']:.4f}, BalAcc: {train_metrics['balanced_acc']:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}, BalAcc: {val_metrics['balanced_acc']:.4f} | "
                      f"LR: {current_lr:.6f}")

        if val_metrics['loss'] < best_valid_loss:
            best_valid_loss = val_metrics['loss']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'concept_pairs': concept_pairs
            }, best_model_path)

    log_print(f"\nBest validation loss: {best_valid_loss:.4f} at epoch {best_epoch}")

    log_print("\nLoading best model for evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    log_print("\nEvaluating on test set...")
    test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    log_print(f"\nTest Results:")
    log_print(f"  Loss: {test_metrics['loss']:.4f}")
    log_print(f"  Accuracy: {test_metrics['acc']:.4f}")
    log_print(f"  Balanced Accuracy: {test_metrics['balanced_acc']:.4f}")
    log_print(f"  AUC: {test_metrics['auc']:.4f}")
    log_print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    log_print(f"  Specificity: {test_metrics['specificity']:.4f}")
    log_print(f"  Precision: {test_metrics['precision']:.4f}")
    log_print(f"  F1 Score: {test_metrics['f1']:.4f}")

    log_print("\nClassification Report:")
    report = classification_report(test_labels, test_preds, target_names=['benign', 'malignant'])
    log_print(report)

    test_metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    test_metrics_save = {k: round(v, 6) for k, v in test_metrics.items()}
    test_metrics_save['best_epoch'] = best_epoch
    test_metrics_save['best_val_loss'] = round(best_valid_loss, 6)
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics_save, f, indent=2)
    log_print(f"\nTest metrics saved to {test_metrics_path}")

    rules = extract_rules(model, top_k=args.top_k_rules)

    rules_path = os.path.join(args.output_dir, 'learned_rules.json')
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)
    log_print(f"All learned rules saved to {rules_path}")

    log_print("\nGenerating case-level explanations...")

    train_explanations = explain_cases(model, train_loader, device)
    log_print(f"  Train: {len(train_explanations)} cases")

    val_explanations = explain_cases(model, val_loader, device)
    log_print(f"  Val: {len(val_explanations)} cases")

    test_explanations = explain_cases(model, test_loader, device)
    log_print(f"  Test: {len(test_explanations)} cases")

    case_explanations = {
        'train': train_explanations,
        'val': val_explanations,
        'test': test_explanations
    }

    explanations_path = os.path.join(args.output_dir, 'case_explanations.json')
    with open(explanations_path, 'w') as f:
        json.dump(case_explanations, f, indent=2)
    log_print(f"Case explanations saved to {explanations_path}")

    log_print("\n" + "=" * 60)
    log_print("Training complete!")
    log_print("=" * 60)
    log_print(f"Output directory: {args.output_dir}")
    log_print(f"  - best_model.pth: Best model checkpoint")
    log_print(f"  - training_log.txt: All training outputs")
    log_print(f"  - test_metrics.json: Final test set metrics")
    log_print(f"  - learned_rules.json: All learned logic rules with 2-class weights")
    log_print(f"  - case_explanations.json: Case-level rule activations with 2-class contributions")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment xxx')

    # Data
    parser.add_argument('--data_path', type=str,
                        default=None,
                        help='Path to dataset JSON file')
    parser.add_argument('--output_dir', type=str,
                        default=None,
                        help='Directory for all outputs (created if not exists)')

    parser.add_argument('--n_logic_neurons', type=int, default=64,
                        help='Number of logic neurons')
    parser.add_argument('--n_logic_layers', type=int, default=1,
                        help='Number of logic layers')
    parser.add_argument('--fixed_gates', action='store_true',
                        help='Use fixed random gates instead of learned gates')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    parser.add_argument('--print_every', type=int, default=1,
                        help='Print and log every N epochs')
    parser.add_argument('--top_k_rules', type=int, default=10,
                        help='Number of top rules to display')

    args = parser.parse_args()
    main(args)
