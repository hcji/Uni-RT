# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 17:03:01 2025

@author: hongchao ji
"""

import os
import datetime
import math
import random
from tqdm import tqdm
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
from scipy.stats import boxcox, gaussian_kde
from scipy.special import inv_boxcox
import matplotlib.pyplot as plt

# ---- PyTorch Geometric ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import rdchem
from models import SingleTaskGINKAN,SingleTaskGIN,SingleTaskGAT,SingleTaskGCN

path = 'D:/Uni-RT/'

DATA_MODE = 'RPLC-GINKAN' 

current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(path, 'specific_pernormol_results', DATA_MODE, current_time_str)


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory created: {OUTPUT_DIR}")
else:
    print(f"Directory already exists: {OUTPUT_DIR}")

print(f"--- Results will be saved to: {OUTPUT_DIR} ---")

# --- data files ---
data_files = [
     path+'data/0027/0027_rtdata_canonical_success.tsv',
     path+'data/0183/0183_rtdata_canonical_success.tsv',
     path+'data/0184/0184_rtdata_canonical_success.tsv',
     path+'data/0185/0185_rtdata_canonical_success.tsv',
     path+'data/0231/0231_rtdata_canonical_success.tsv',
     path+'data/0282/0282_rtdata_canonical_success.tsv',
     path+'data/0283/0283_rtdata_canonical_success.tsv',
     path+'data/0372/0372_rtdata_canonical_success.tsv',
     path+'data/0373/0373_rtdata_canonical_success.tsv',
     path+'data/0374/0374_rtdata_canonical_success.tsv',
     path+'data/0375/0375_rtdata_canonical_success.tsv',
     path+'data/0376/0376_rtdata_canonical_success.tsv',
     path+'data/0377/0377_rtdata_canonical_success.tsv',
     path+'data/0378/0378_rtdata_canonical_success.tsv'
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

def smiles_to_data(smiles, y) :
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
def load_dataset(csv_path: str) -> List[Data]:
    df = pd.read_csv(csv_path, sep='\t')
    if not {"smiles.std", "rt"}.issubset(df.columns):
        raise ValueError("CSV must have columns: smiles.std, rt")

    # 原始 RT (秒)
    rt_values = df["rt"].astype(float) * 60
   
    data_list = []
    for smi, y in zip(df["smiles.std"].astype(str), rt_values):
        try:
            data_list.append(smiles_to_data(smi, y))
        except Exception as e:
            print(f"[WARN] Skipping SMILES: {smi} -> {e}")

    return data_list

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

    non_zero_mask = np.abs(y) > 1e-6 
    mre = np.mean(np.abs((y[non_zero_mask] - p[non_zero_mask]) / y[non_zero_mask])) if np.sum(non_zero_mask) > 0 else 0.0

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
    norm_params_list = [] 

    for task_id, csv_path in enumerate(data_files):
       
        file_name = os.path.basename(csv_path)
        data_id = file_name.split('_')[0]
        
        print(f"\n=== Training single-task model on dataset {data_id} ===")
        
        dataset = load_dataset(csv_path)
        if len(dataset) < 10:
            print(f"[WARN] Dataset {data_id} too small, skipped.")
            continue
        y_vals = np.array([d.y.item() for d in dataset])
        n_quantiles = min(10, len(y_vals) // 5)
        
        if len(y_vals) > 20 and n_quantiles > 1:
            try:
                quantiles = pd.qcut(pd.Series(y_vals), q=n_quantiles, labels=False, duplicates='drop')
            except:
                quantiles = None
        else:
            quantiles = None
            
        idx = np.arange(len(dataset))
        
        # Train/Temp Split
        try:
            if quantiles is not None:
                idx_train, idx_temp, _, y_temp = train_test_split(idx, quantiles, test_size=0.2, stratify=quantiles, random_state=42)
            else:
                raise ValueError
        except ValueError:
            idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42)
            y_temp = None

        # Val/Test Split
        try:
            if y_temp is not None:
                idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_temp, random_state=42)
            else:
                raise ValueError
        except ValueError:
            idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

        train_set = [dataset[i] for i in idx_train]
        val_set = [dataset[i] for i in idx_val]
        test_set = [dataset[i] for i in idx_test]

        train_y_tensor = torch.tensor([d.y.item() for d in train_set])
        mean_val = train_y_tensor.mean().item()
        std_val = train_y_tensor.std().item()
        if std_val == 0: std_val = 1.0

        norm_params_list.append({
            "Data ID": data_id, 
            "mean": mean_val,
            "std": std_val
        })

        def normalize_dataset(data_list, mean, std):
            for d in data_list:
                d.y = (d.y - mean) / std
        
        normalize_dataset(train_set, mean_val, std_val)
        normalize_dataset(val_set, mean_val, std_val)
        normalize_dataset(test_set, mean_val, std_val)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        node_dim = dataset[0].x.size(-1)
        model = SingleTaskGINKAN(node_dim=node_dim, hidden=64, layers=4).to(device)
        #model = SingleTaskGIN(node_dim=node_dim, hidden=64, layers=4, dropout=0.1).to(device)
        #model = SingleTaskGAT(node_dim=node_dim, hidden=64, layers=4, dropout=0.1).to(device)
        #model = SingleTaskGCN(node_dim=node_dim, hidden=64, layers=4, dropout=0.1).to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
        loss_fn = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_mae = float('inf')
        best_state = None

        for epoch in range(1, 501):
            _ = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            val_metrics = evaluate(model, val_loader, device, mean_val, std_val)
            scheduler.step(val_metrics["MAE"])

            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics = evaluate(model, test_loader, device, mean_val, std_val)
        
        results[data_id] = test_metrics 
        print(f"Dataset {data_id} Results: {test_metrics}")

    print("\n=== Summary of single-task models ===")
    
    df_results = pd.DataFrame.from_dict(results, orient='index')
    
    cols_order = ['MAE', 'MedAE', 'MRE', 'R2']
    df_results = df_results[cols_order]
    
    avg_row = df_results.mean()
    std_row = df_results.std()
    df_results.loc['Avg'] = avg_row
    df_results.loc['Std'] = std_row
    
    df_results.index.name = 'Data ID'

    df_results = df_results.round(3)
    
    print(df_results)
    

    results_path = os.path.join(OUTPUT_DIR, 'test_metrics_summary.csv')
    df_results.to_csv(results_path)
    print(f"Metrics saved to: {results_path}")

    norm_df = pd.DataFrame(norm_params_list)
    norm_path = os.path.join(OUTPUT_DIR, 'normalization_params.csv')
    norm_df.to_csv(norm_path, index=False)
    
    