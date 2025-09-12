# -*- coding: utf-8 -*-
"""
Train separate models on each dataset individually and compute performance metrics
for comparison with multi-task learning.

Author: 纪宏超
"""

import os
import math
import random
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---- PyTorch Geometric ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
from torch_geometric.nn import GCNConv, GATConv

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import rdchem

# --- data files ---
data_files = [
    'data/0019/0019_rtdata_canonical_success.tsv',
    'data/0052/0052_rtdata_canonical_success.tsv',
    'data/0179/0179_rtdata_canonical_success.tsv',
    'data/0180/0180_rtdata_canonical_success.tsv',
    'data/0234/0234_rtdata_canonical_success.tsv',
    'data/0235/0235_rtdata_canonical_success.tsv',
    'data/0260/0260_rtdata_canonical_success.tsv',
    'data/0261/0261_rtdata_canonical_success.tsv',
    'data/0264/0264_rtdata_canonical_success.tsv',
    'data/0317/0317_rtdata_canonical_success.tsv',
    'data/0319/0319_rtdata_canonical_success.tsv',
    'data/0321/0321_rtdata_canonical_success.tsv',
    'data/0323/0323_rtdata_canonical_success.tsv',
    'data/0331/0331_rtdata_canonical_success.tsv'
]

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

def smiles_to_data(smiles: str, y: float | None = None) -> Data:
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
    return data

# ------------------------------
# Dataset loader
# ------------------------------
def load_dataset(csv_path: str) -> Tuple[List[Data], dict]:
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
            data_list.append(smiles_to_data(smi, y_norm))
        except Exception as e:
            print(f"[WARN] Skipping SMILES: {smi} -> {e}")

    return data_list, (mean_val, std_val)

# ------------------------------
# Single-task GIN Model
# ------------------------------
class SingleTaskGIN(nn.Module):
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

    def forward(self, data):
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


class SingleTaskGCN(nn.Module):
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
    
    
class SingleTaskGAT(nn.Module):
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

def evaluate(model, loader, device, mean_val, std_val):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            y_true = data.y.view(-1).cpu().numpy() * std_val + mean_val
            y_pred = pred.cpu().numpy() * std_val + mean_val
            ys.append(y_true)
            ps.append(y_pred)

    y = np.concatenate(ys)
    p = np.concatenate(ps)

    non_zero_mask = y != 0
    mre = np.mean(np.abs((y[non_zero_mask] - p[non_zero_mask]) / y[non_zero_mask])) if np.sum(non_zero_mask) > 0 else np.nan

    return {
        "MAE": mean_absolute_error(y, p),
        "MedAE": median_absolute_error(y, p),
        "MRE": mre,
        "R2": r2_score(y, p)
    }

# ------------------------------
# Main Loop
# ------------------------------
if __name__ == "__main__":
    
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}

    for task_id, csv_path in enumerate(data_files):
        print(f"\n=== Training single-task model on dataset {task_id} ===")
        dataset, (mean_val, std_val) = load_dataset(csv_path)
        if len(dataset) < 10:
            print(f"[WARN] Dataset {csv_path} too small, skipped.")
            continue

        # Split train/val/test (stratify by rt quantiles)
        y_vals = np.array([d.y.item() for d in dataset])
        quantiles = pd.qcut(pd.Series(y_vals), q=min(10, len(y_vals)), labels=False, duplicates='drop')
        idx = np.arange(len(dataset))
        idx_train, idx_temp, _, y_temp = train_test_split(idx, quantiles, test_size=0.2, stratify=quantiles)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_temp)

        train_set = [dataset[i] for i in idx_train]
        val_set = [dataset[i] for i in idx_val]
        test_set = [dataset[i] for i in idx_test]

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        node_dim = dataset[0].x.size(-1)
        model = SingleTaskGIN(node_dim=node_dim, hidden=64, layers=4, dropout=0.1).to(device)
        # model = SingleTaskGCN(node_dim=node_dim, hidden=64, layers=4, dropout=0.1).to(device)
        # model = SingleTaskGAT(node_dim=node_dim, hidden=64, layers=4, heads=4, dropout=0.02).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(1, 201):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            val_metrics = evaluate(model, val_loader, device, mean_val, std_val)
            scheduler.step(val_metrics["MAE"])

            if val_metrics["MAE"] < best_val_loss:
                best_val_loss = val_metrics["MAE"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                      f"Val MAE: {val_metrics['MAE']:.4f} | R2: {val_metrics['R2']:.4f}")

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Test set evaluation
        test_metrics = evaluate(model, test_loader, device, mean_val, std_val)
        results[task_id] = test_metrics
        print(f"Test Results for dataset {task_id}: {test_metrics}")

    # Summarize overall results
    print("\n=== Summary of single-task models ===")
    df_results = pd.DataFrame(results).T
    print(df_results)
    print("Average Metrics:")
    print(df_results.mean())
