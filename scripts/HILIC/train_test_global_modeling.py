# -*- coding: utf-8 -*-
"""
Train a single unified model on merged datasets without task embeddings
and evaluate with per-task metrics and high-quality scatter plots.

Author: 纪宏超
"""

import os
import math
import random
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ---- PyTorch Geometric ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
from torch_geometric.nn import GCNConv, GATConv

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import rdchem

# --- data files ---
data_files = ['data/0027/0027_rtdata_canonical_success.tsv',
              'data/0183/0183_rtdata_canonical_success.tsv',
              'data/0184/0184_rtdata_canonical_success.tsv',
              'data/0185/0185_rtdata_canonical_success.tsv',
              'data/0231/0231_rtdata_canonical_success.tsv',
              'data/0282/0282_rtdata_canonical_success.tsv',
              'data/0283/0283_rtdata_canonical_success.tsv',
              'data/0372/0372_rtdata_canonical_success.tsv',
              'data/0373/0373_rtdata_canonical_success.tsv',
              'data/0374/0374_rtdata_canonical_success.tsv',
              'data/0375/0375_rtdata_canonical_success.tsv',
              'data/0376/0376_rtdata_canonical_success.tsv',
              'data/0377/0377_rtdata_canonical_success.tsv',
              'data/0378/0378_rtdata_canonical_success.tsv']

# ------------------------------
# Utils
# ------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Featurization
# ------------------------------
ATOMIC_NUM_SET = list(range(1, 40))
HYBRIDIZATION_SET = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

BOND_TYPE_SET = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

def one_hot(x, allowed):
    return [int(x == s) for s in allowed] + [int(x not in allowed)]

def atom_features(atom: rdchem.Atom) -> List[int]:
    return (
        one_hot(atom.GetAtomicNum(), ATOMIC_NUM_SET)
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_SET)
        + [
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED),
        ]
    )

def bond_features(bond: rdchem.Bond) -> List[int]:
    if bond is None:
        return [0] * (len(BOND_TYPE_SET) + 3)
    bt = bond.GetBondType()
    return (
        one_hot(bt, BOND_TYPE_SET)
        + [int(bond.GetIsConjugated()), int(bond.IsInRing()), int(bond.GetStereo() != rdchem.BondStereo.STEREONONE)]
    )

def smiles_to_data(smiles: str, y: float | None = None, task_id: Optional[int] = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bf, bf])

    if len(edge_index) == 0:
        for idx in range(mol.GetNumAtoms()):
            edge_index.append([idx, idx])
            edge_attr.append(bond_features(None))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    if task_id is not None:
        data.task = torch.tensor([task_id], dtype=torch.long)
    return data

# ------------------------------
# Dataset loader
# ------------------------------
def load_dataset(csv_path: str, task_id: int) -> Tuple[List[Data], dict]:
    df = pd.read_csv(csv_path, sep='\t')
    if not {"smiles.std", "rt"}.issubset(df.columns):
        raise ValueError("CSV must have columns: smiles.std, rt")

    rt_values = df["rt"].astype(float) * 60
    mean_val = rt_values.mean()
    std_val = rt_values.std()
    if std_val < 1e-6:
        std_val = 1.0

    data_list = []
    for smi, y in zip(df["smiles.std"].astype(str), rt_values):
        try:
            y_norm = (y - mean_val) / std_val
            data_list.append(smiles_to_data(smi, y_norm, task_id=task_id))
        except Exception as e:
            print(f"[WARN] Skipping SMILES: {smi} -> {e}")

    return data_list, {task_id: (mean_val, std_val)}

# ------------------------------
# GIN Regressor (No Task Embedding)
# ------------------------------
class GINUnified(nn.Module):
    def __init__(self, node_dim, hidden=64, layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            nn_node = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden)
            )
            conv = GINConv(nn=nn_node)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g).view(-1)


class GCNUnified(nn.Module):
    def __init__(self, node_dim, hidden=64, layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            conv = GCNConv(hidden, hidden)
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g).view(-1)
    
    
class GATUnified(nn.Module):
    def __init__(self, node_dim, hidden=64, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层 GATConv
        self.convs.append(GATConv(hidden, hidden // heads, heads=heads))
        self.bns.append(BatchNorm(hidden))
        
        # 其余层
        for _ in range(layers - 1):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads))
            self.bns.append(BatchNorm(hidden))
            
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)  # GAT常用ELU
        g = global_add_pool(h, batch)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g).view(-1)


# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data)
        loss = loss_fn(pred, data.y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def task_evaluate(model, loader, device, task_stats, num_tasks):
    """Evaluate per task, return overall and per-task metrics."""
    model.eval()
    ys_all, ps_all = [], []
    ys_by_task, ps_by_task = defaultdict(list), defaultdict(list)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            task_ids = data.task.view(-1).cpu().numpy()

            for y_hat, y_true, t_id in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy(), task_ids):
                mean, std = task_stats[int(t_id)]
                y_orig = y_true * std + mean
                p_orig = y_hat * std + mean

                ys_all.append(y_orig)
                ps_all.append(p_orig)
                ys_by_task[int(t_id)].append(y_orig)
                ps_by_task[int(t_id)].append(p_orig)

    y = np.array(ys_all)
    p = np.array(ps_all)

    non_zero_mask = y != 0
    mre = np.mean(np.abs((y[non_zero_mask] - p[non_zero_mask]) / y[non_zero_mask])) if np.sum(non_zero_mask) > 0 else np.nan

    # overall metrics
    overall = {
        "MAE": mean_absolute_error(y, p),
        "MedAE": median_absolute_error(y, p),
        "MRE": mre,
        "R2": r2_score(y, p),
    }

    # per-task metrics
    per_task = {}
    for t in range(num_tasks):
        if len(ys_by_task[t]) == 0:
            continue
        y_t = np.array(ys_by_task[t])
        p_t = np.array(ps_by_task[t])

        non_zero_mask_t = y_t != 0
        mre_t = np.mean(np.abs((y_t[non_zero_mask_t] - p_t[non_zero_mask_t]) / y_t[non_zero_mask_t])) if np.sum(non_zero_mask_t) > 0 else np.nan

        per_task[t] = {
            "MAE": mean_absolute_error(y_t, p_t),
            "MedAE": median_absolute_error(y_t, p_t),
            "MRE": mre_t,
            "R2": r2_score(y_t, p_t),
        }

    return overall, per_task, y, p

# ------------------------------
# Scatter plot with density and trendline
# ------------------------------
def plot_scatter(y_true, y_pred, save_path=None):
    """Generate high-quality scatter plot with density coloring."""
    # Calculate density
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    scatter = ax.scatter(y_true, y_pred, c=z, s=15, cmap='viridis', alpha=0.8)

    ax.plot(y_true, y_true, color='black', linestyle='--', linewidth=2,
            label=f'Trendline (R²={r2_score(y_true, y_pred):.3f})')

    # aesthetics
    ax.set_xlabel("True RT (sec)", fontsize=12)
    ax.set_ylabel("Predicted RT (sec)", fontsize=12)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label='Density')

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# ------------------------------
# Main
# ------------------------------
def main():
    batch_size = 32
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load all datasets and merge
    all_data = []
    task_stats = {}
    for task_id, csv in enumerate(data_files):
        print(f"Loading dataset {task_id}: {csv}")
        ds, stats = load_dataset(csv, task_id)
        all_data.extend(ds)
        task_stats.update(stats)

    print(f"Total graphs: {len(all_data)}")

    node_dim = all_data[0].x.size(-1)

    # Stratified split by y
    y_vals = np.array([d.y.item() for d in all_data])
    quantiles = pd.qcut(pd.Series(y_vals), q=min(10, len(y_vals)), labels=False, duplicates='drop')
    idx = np.arange(len(all_data))
    idx_train, idx_temp, _, y_temp = train_test_split(idx, quantiles, test_size=0.2, stratify=quantiles)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_temp)

    train_set = [all_data[i] for i in idx_train]
    val_set = [all_data[i] for i in idx_val]
    test_set = [all_data[i] for i in idx_test]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model
    # model = GINUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    # model = GCNUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    model = GATUnified(node_dim=node_dim, hidden=64, layers=4, heads=4, dropout=0.02).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None

    # Training
    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        overall_val, _, y_val_true, y_val_pred = task_evaluate(model, val_loader, device, task_stats, len(data_files))
        scheduler.step(overall_val["MAE"])

        if overall_val["MAE"] < best_val:
            best_val = overall_val["MAE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val MAE: {overall_val['MAE']:.4f} | R2: {overall_val['R2']:.4f}")

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    print("\nEvaluating on test set...")
    overall_test, per_task_test, y_test, p_test = task_evaluate(model, test_loader, device, task_stats, len(data_files))
    print("Test Metrics:", per_task_test)

    # Scatter plot
    plot_scatter(y_test, p_test)


if __name__ == "__main__":
    main()
