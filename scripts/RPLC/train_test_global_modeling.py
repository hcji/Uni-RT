
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 17:03:01 2025

@author: hongchao ji
"""
import os
import math
import random
import datetime
import json  
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ---- PyTorch Geometric ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import rdchem

# ---- Models (Assuming these exist in your 'models.py') ----
from models import GINKANUnified, GINUnified, GATUnified, GCNUnified
# ==========================================
# 1. Configuration & Directory Setup
# ==========================================
# Base path
path = 'D:/Uni-RT/'

# Experiment Mode
EXP_MODE = 'RPLC-GINKAN' 

# Current Time
current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save Directory
SAVE_DIR = os.path.join(path, 'Global_results', EXP_MODE, current_time_str)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"========================================================")
print(f"All results will be saved to: {SAVE_DIR}")
print(f"========================================================")

# --- data files ---
data_files = [
    path+'data/0019/0019_rtdata_canonical_success.tsv',
    path+'data/0052/0052_rtdata_canonical_success.tsv',
    path+'data/0179/0179_rtdata_canonical_success.tsv',
    path+'data/0180/0180_rtdata_canonical_success.tsv',
    path+'data/0234/0234_rtdata_canonical_success.tsv',
    path+'data/0235/0235_rtdata_canonical_success.tsv',
    path+'data/0260/0260_rtdata_canonical_success.tsv',
    path+'data/0261/0261_rtdata_canonical_success.tsv',
    path+'data/0264/0264_rtdata_canonical_success.tsv',
    path+'data/0317/0317_rtdata_canonical_success.tsv',
    path+'data/0319/0319_rtdata_canonical_success.tsv',
    path+'data/0321/0321_rtdata_canonical_success.tsv',
    path+'data/0323/0323_rtdata_canonical_success.tsv',
    path+'data/0331/0331_rtdata_canonical_success.tsv'
]

# ------------------------------
# Utils
# ------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------
# Featurization
# ------------------------------
ATOMIC_NUM_SET = list(range(1, 40))
HYBRIDIZATION_SET = [
    rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPE_SET = [
    rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC,
]

def one_hot(x, allowed):
    return [int(x == s) for s in allowed] + [int(x not in allowed)]

def atom_features(atom: rdchem.Atom) -> List[int]:
    return (
        one_hot(atom.GetAtomicNum(), ATOMIC_NUM_SET)
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_SET)
        + [
            atom.GetTotalDegree(), atom.GetTotalNumHs(),
            atom.GetFormalCharge(), int(atom.GetIsAromatic()),
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
        + [int(bond.GetIsConjugated()), int(bond.IsInRing()),
           int(bond.GetStereo() != rdchem.BondStereo.STEREONONE)]
    )

def smiles_to_data(smiles, y, task_id):
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
    if y is not None and not (isinstance(y, float) and math.isnan(y)):
        data.y = torch.tensor([y], dtype=torch.float)
    else:
        data.y = torch.tensor([float('nan')], dtype=torch.float)
    data.smiles = smiles
    if task_id is not None:
        data.task = torch.tensor([int(task_id)], dtype=torch.long)
    return data

# ------------------------------
# Dataset loader
# ------------------------------
def load_dataset_raw(csv_path: str, task_id: int) -> List[Tuple[str, float, int]]:
    """Loads raw data (smiles, rt, task_id) without any transformation."""
    df = pd.read_csv(csv_path, sep='\t')
    # Ensure necessary columns exist
    if 'smiles.std' not in df.columns or 'rt' not in df.columns:
        print(f"Warning: Columns missing in {csv_path}")
        return []
        
    df.dropna(subset=['smiles.std', 'rt'], inplace=True)
    df = df[pd.to_numeric(df['rt'], errors='coerce').notna()]
    
    data = []
    for _, row in df.iterrows():
        smi = str(row['smiles.std'])
        rt = float(row['rt']) * 60 # Convert minutes to seconds
        data.append((smi, rt, task_id))
    return data

# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn, main_task_id=0, main_task_weight=5.0):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data)

        valid_mask = ~torch.isnan(data.y)
        if valid_mask.sum() == 0:
            continue
        
        pred = pred[valid_mask]
        y_true = data.y[valid_mask]
        task_ids = data.task.view(-1)[valid_mask]

        base_loss = loss_fn(pred, y_true.view(-1))

        # Weighting strategy
        weights = torch.ones_like(task_ids, dtype=torch.float, device=device)
        weights[task_ids == main_task_id] = main_task_weight

        loss = (base_loss * weights).mean()

        if torch.isnan(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, task_stats):
    """
    Evaluates the model. 
    y_true and y_pred are denormalized using task_stats based on the task_id of each sample.
    """
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            
            task_ids = data.task.view(-1).cpu().numpy()
            y_raw_tensor = data.y.view(-1).cpu().numpy()
            pred_raw_tensor = pred.view(-1).cpu().numpy()

            for i in range(len(task_ids)):
                y_val = y_raw_tensor[i]
                if np.isnan(y_val): continue
                
                t_id = int(task_ids[i])
                p_val = pred_raw_tensor[i]
                
                # Retrieve stats for this task
                stats = task_stats.get(t_id, {'mean': 0.0, 'std': 1.0})
                mean_val = stats['mean']
                std_val = stats['std']
                
                # Denormalize: y_orig = y_norm * std + mean
                y_final = y_val * std_val + mean_val
                pred_final = p_val * std_val + mean_val
                
                ys.append(y_final)
                ps.append(pred_final)

    if not ys:
        return {"MAE": float('inf'), "RMSE": float('inf'), "R2": -float('inf')}
        
    y = np.array(ys)
    p = np.array(ps)

    valid_indices = np.isfinite(y) & np.isfinite(p)
    if np.sum(valid_indices) < 2:
        return {"MAE": float('inf'), "RMSE": float('inf'), "R2": -float('inf')}
    
    y_clean, p_clean = y[valid_indices], p[valid_indices]

    mae = mean_absolute_error(y_clean, p_clean)
    rmse = math.sqrt(((y_clean - p_clean) ** 2).mean())
    r2 = r2_score(y_clean, p_clean)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ------------------------------
# Scatter plot
# ------------------------------
def plot_scatter(y_true, y_pred, save_path=None):
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid_indices], y_pred[valid_indices]

    if len(y_true) < 2:
        print("[WARN] Not enough valid data points to generate a scatter plot.")
        return

    xy = np.vstack([y_true, y_pred])
    try:
        z = gaussian_kde(xy)(xy)
    except np.linalg.LinAlgError:
        z = np.ones_like(y_true)

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    scatter = ax.scatter(y_true, y_pred, c=z, s=15, cmap='viridis', alpha=0.8)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    r2_val = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'$R^2$ = {r2_val:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    ax.set_xlabel("True RT (seconds)", fontsize=12)
    ax.set_ylabel("Predicted RT (seconds)", fontsize=12)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label='Density')

    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Scatter plot saved to: {save_path}")

def task_evaluate(model, loader, device, task_stats, num_tasks=28, save_dir=None):
    model.eval()
    ys_all, ps_all = [], []
    ys_by_task, ps_by_task = defaultdict(list), defaultdict(list)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            task_ids = data.task.view(-1).cpu().numpy()
            y_raw = data.y.view(-1).cpu().numpy()
            p_raw = pred.view(-1).cpu().numpy()
            
            for i in range(len(task_ids)):
                if np.isnan(y_raw[i]): continue
                
                t_id = int(task_ids[i])
                stats = task_stats.get(t_id, {'mean': 0.0, 'std': 1.0})
                
                # Denormalize
                y_orig = y_raw[i] * stats['std'] + stats['mean']
                p_orig = p_raw[i] * stats['std'] + stats['mean']

                ys_all.append(y_orig)
                ps_all.append(p_orig)
                ys_by_task[t_id].append(y_orig)
                ps_by_task[t_id].append(p_orig)

    y = np.array(ys_all)
    p = np.array(ps_all)
    
    valid_indices = np.isfinite(y) & np.isfinite(p)
    y_clean, p_clean = y[valid_indices], p[valid_indices]

    if len(y_clean) == 0:
        return {}, {}, np.array([]), np.array([])
        
    non_zero_mask = y_clean != 0
    mre = np.nan
    if np.sum(non_zero_mask) > 0:
        mre = np.mean(np.abs((y_clean[non_zero_mask] - p_clean[non_zero_mask]) / y_clean[non_zero_mask]))

    overall = {
        "MAE": mean_absolute_error(y_clean, p_clean),
        "MedAE": median_absolute_error(y_clean, p_clean),
        "MRE": mre,
        "R2": r2_score(y_clean, p_clean),
    }

    per_task = {}
    for t in range(num_tasks):
        if len(ys_by_task[t]) == 0:
            continue
        
        y_t = np.array(ys_by_task[t])
        p_t = np.array(ps_by_task[t])

        valid_indices_t = np.isfinite(y_t) & np.isfinite(p_t)
        y_t_clean, p_t_clean = y_t[valid_indices_t], p_t[valid_indices_t]

        if len(y_t_clean) < 2: continue
        
        # Plot Scatter for main task (Task 0) or others if desired
        if t == 0:
            fname = f'task_{t}_scatter.png'
            sp = os.path.join(save_dir, fname) if save_dir else fname
            plot_scatter(y_t_clean, p_t_clean, save_path=sp)
        
        non_zero_mask_t = y_t_clean != 0
        mre_t = np.nan
        if np.sum(non_zero_mask_t) > 0:
            mre_t = np.mean(np.abs((y_t_clean[non_zero_mask_t] - p_t_clean[non_zero_mask_t]) / y_t_clean[non_zero_mask_t]))

        per_task[t] = {
            "MAE": mean_absolute_error(y_t_clean, p_t_clean),
            "MedAE": median_absolute_error(y_t_clean, p_t_clean),
            "MRE": mre_t,
            "R2": r2_score(y_t_clean, p_t_clean),
        }

    return overall, per_task, y, p


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    batch_size = 32
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==================================================================================
    # Step 1 & 2 (Modified): Per-Task Stratified Splitting
    # ==================================================================================
    print("Step 1 & 2: Loading data and performing Per-Task Stratified Splitting...")
    raw_train_set = []
    raw_val_set = []
    raw_test_set = []

    for task_id, csv_path in enumerate(data_files):
      
        task_data = load_dataset_raw(csv_path, task_id)
        if len(task_data) == 0:
            print(f"Warning: Task {task_id} is empty.")
            continue
 
        task_rts = np.array([item[1] for item in task_data])
        task_indices = np.arange(len(task_data))
        
        try:
      
            n_bins = min(10, len(task_data) // 5) 
            if n_bins < 2: raise ValueError("Not enough data for stratification")
            
            quantiles = pd.qcut(task_rts, q=n_bins, labels=False, duplicates='drop')
            idx_train_local, idx_temp_local = train_test_split(
                task_indices, test_size=0.2, stratify=quantiles, random_state=42
            )
            
            temp_rts = task_rts[idx_temp_local]
            n_bins_temp = min(5, len(temp_rts) // 2)
            
            if n_bins_temp < 2:
                idx_val_local, idx_test_local = train_test_split(
                    idx_temp_local, test_size=0.5, random_state=42
                )
            else:
                quantiles_temp = pd.qcut(temp_rts, q=n_bins_temp, labels=False, duplicates='drop')
                idx_val_local, idx_test_local = train_test_split(
                    idx_temp_local, test_size=0.5, stratify=quantiles_temp, random_state=42
                )
                
        except ValueError:
            idx_train_local, idx_temp_local = train_test_split(task_indices, test_size=0.2, random_state=42)
            idx_val_local, idx_test_local = train_test_split(idx_temp_local, test_size=0.5, random_state=42)

        for idx in idx_train_local:
            raw_train_set.append(task_data[idx])
        for idx in idx_val_local:
            raw_val_set.append(task_data[idx])
        for idx in idx_test_local:
            raw_test_set.append(task_data[idx])
            
    print(f"Data Split Summary:")
    print(f"  Train: {len(raw_train_set)}")
    print(f"  Val  : {len(raw_val_set)}")
    print(f"  Test : {len(raw_test_set)}")

    # ==================================================================================
    # Step 3: Calculate Mean/Std PER TASK using ONLY the Training Set
    # ==================================================================================
    print("Step 3: Calculating Per-Task Normalization Stats (Train Set Only)...")
    
    # Store raw training RTs for each task
    task_rt_map = defaultdict(list)
    for _, rt, t_id in raw_train_set:
        if rt > 0: 
            task_rt_map[t_id].append(rt)
            
    # Calculate stats
    task_stats = {}
    num_tasks = len(data_files)
    
    for t_id in range(num_tasks):
        rts = np.array(task_rt_map[t_id])
        if len(rts) > 1:
            mean_val = float(np.mean(rts))
            std_val = float(np.std(rts))
            if std_val == 0: std_val = 1.0 # Prevent division by zero
        else:
            # Fallback if specific task has no training data (unlikely but safe to handle)
            mean_val = 0.0
            std_val = 1.0
            
        task_stats[t_id] = {'mean': mean_val, 'std': std_val}
    
    print(f"Computed stats for {len(task_stats)} tasks.")

    # ==================================================================================
    # Step 4 & 5: Transform datasets and create DataLoaders
    # ==================================================================================
    print("Step 4 & 5: Creating PyG datasets and dataloaders...")
    
    def create_pyg_dataset(raw_list, stats_map):
        dataset = []
        for smi, rt, task_id in tqdm(raw_list, desc="Processing"):
            if rt > 0:
                stats = stats_map[task_id]
                # Z-Score Normalization
                y_norm = (rt - stats['mean']) / stats['std']
                data_obj = smiles_to_data(smi, y_norm, task_id)
            else:
                data_obj = smiles_to_data(smi, float('nan'), task_id)
            
            if not torch.isnan(data_obj.y):
                dataset.append(data_obj)
        return dataset

    print("Processing Training Set...")
    train_set = create_pyg_dataset(raw_train_set, task_stats)
    print("Processing Validation Set...")
    val_set = create_pyg_dataset(raw_val_set, task_stats)
    print("Processing Test Set...")
    test_set = create_pyg_dataset(raw_test_set, task_stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    node_dim = atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).__len__()
    
    model = GINKANUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    #model = GINKANUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    #model = GINUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    #model = GCNUnified(node_dim=node_dim, hidden=64, layers=4, dropout=0.02).to(device)
    #model = GATUnified(node_dim=node_dim, hidden=64, layers=4, heads=4, dropout=0.02).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = nn.L1Loss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None

    print("\n--- Starting Model Training ---")
    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        # Evaluate uses per-task stats to reverse normalization
        metrics = evaluate(model, val_loader, device, task_stats)
        
        if metrics["MAE"] is not None and not math.isinf(metrics["MAE"]):
            scheduler.step(metrics["MAE"])
            if metrics["MAE"] < best_val:
                best_val = metrics["MAE"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss (Norm): {train_loss:.4f} | "
                  f"Val (Orig Scale): MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        # Save best model
        model_save_path = os.path.join(SAVE_DIR, 'best_model.pth')
        torch.save(best_state, model_save_path)
        print(f"Best model weights saved to: {model_save_path}")

    print("\nEvaluating on test set (best val checkpoint)...")
    overall_test, per_task_test, y_test, p_test = task_evaluate(
        model, test_loader, device, task_stats, len(data_files), save_dir=SAVE_DIR
    )
    
    print("\n--- Overall Test Metrics ---")
    if overall_test:
        for metric, value in overall_test.items():
            print(f"{metric}: {value:.4f}")
    
    print("\n--- Per-Task Test Metrics (MAE, R2) ---")
    if per_task_test:
        for task_id, metrics in per_task_test.items():
            print(f"Task {task_id}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    print("\n--- Saving results to DataFrame ---")
    output_csv_filename = f'{current_time_str}_per_task_performance.csv'
    output_csv_path = os.path.join(SAVE_DIR, output_csv_filename)

    if per_task_test:
        results_df = pd.DataFrame.from_dict(per_task_test, orient='index')
        per_task_mean = results_df.mean()
        per_task_std = results_df.std()

        if overall_test:
            results_df.loc['Overall'] = overall_test
        
        results_df.loc['Mean_Per_Task'] = per_task_mean
        results_df.loc['Std_Per_Task'] = per_task_std
        
        cols_order = ['MAE', 'MedAE', 'MRE', 'R2']
        existing_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[existing_cols]
    
        results_df.to_csv(output_csv_path, index=True)
        print(f"\nResults summary saved to {output_csv_path}")
      
    # Save overall scatter plot
    output_png_filename = f'{current_time_str}_overall_scatter.png'
    output_png_path = os.path.join(SAVE_DIR, output_png_filename)
    plot_scatter(y_test, p_test, output_png_path)

    # -------------------------------------------------------------
    # Save Normalization Parameters (Mean/Std per task)
    # -------------------------------------------------------------
    print("\n--- Saving Normalization Parameters ---")
    stats_path = os.path.join(SAVE_DIR, 'task_norm_stats.json')
    with open(stats_path, 'w') as f:
        # json.dump requires keys to be strings usually, but ints work in some parsers.
        # We convert to string keys for standard JSON compatibility.
        json_ready_stats = {str(k): v for k, v in task_stats.items()}
        json.dump(json_ready_stats, f, indent=4)
    print(f"Per-task normalization stats saved to: {stats_path}")
    print(EXP_MODE)
    print(model)