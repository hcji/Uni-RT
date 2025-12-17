
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 12:55:42 2025

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
from models import GINKANmultiRegressor,GINmultiRegressor,GATmultiRegressor,GCNmultiRegressor
from models import GINKANUnified, GINUnified, GATUnified, GCNUnified

BASE_PATH = 'D:/Uni-RT/'

DATA_MODE = 'HILIC-GINKAN' 

current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_PATH, 'golobal1215_persample_boxcos_results', DATA_MODE, current_time_str)


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Directory created: {OUTPUT_DIR}")
else:
    print(f"Directory already exists: {OUTPUT_DIR}")

print(f"--- Running Mode: {DATA_MODE} ---")
print(f"--- Results will be saved to: {OUTPUT_DIR} ---")


data_files = [
     BASE_PATH+'data/0027/0027_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0183/0183_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0184/0184_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0185/0185_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0231/0231_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0282/0282_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0283/0283_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0372/0372_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0373/0373_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0374/0374_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0375/0375_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0376/0376_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0377/0377_rtdata_canonical_success.tsv',
     BASE_PATH+'data/0378/0378_rtdata_canonical_success.tsv'
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
    df.dropna(subset=['smiles.std', 'rt'], inplace=True)
    df = df[pd.to_numeric(df['rt'], errors='coerce').notna()]
    
    data = []
    for _, row in df.iterrows():
        smi = str(row['smiles.std'])
        rt = float(row['rt']) * 60
        data.append((smi, rt, task_id))
    return data

# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn, main_task_id=0, main_task_weight=5.0):
    """
    Trains the model for one epoch using Mean Absolute Error (MAE) loss.
    """
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

        # Calculate absolute error for each sample, as loss_fn is L1Loss(reduction='none')
        base_loss = loss_fn(pred, y_true.view(-1))

        # Apply different weights based on task ID
        weights = torch.ones_like(task_ids, dtype=torch.float, device=device)
        weights[task_ids == main_task_id] = main_task_weight

        # Calculate the weighted mean absolute error
        loss = (base_loss * weights).mean()

        if torch.isnan(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, transform_stats):
    model.eval()
    ys, ps = [], []
    mean, std, lmbda = transform_stats

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            
            pred_rescaled, y_rescaled = [], []

            for y_hat, y_true in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy()):
                if np.isnan(y_true): continue
                
                pred_denorm = y_hat * std + mean
                y_denorm = y_true * std + mean
                
                try:
                    pred_final = inv_boxcox(pred_denorm, lmbda)
                    y_final = inv_boxcox(y_denorm, lmbda)
                except ValueError:
                    continue

                pred_rescaled.append(pred_final)
                y_rescaled.append(y_final)
                
            ys.append(np.array(y_rescaled))
            ps.append(np.array(pred_rescaled))

    if not ys or not any(len(arr) > 0 for arr in ys):
        return {"MAE": float('inf'), "RMSE": float('inf'), "R2": -float('inf')}
        
    y = np.concatenate(ys)
    p = np.concatenate(ps)

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
    plt.show()

def task_evaluate(model, loader, device, transform_stats, num_tasks=28):
    model.eval()
    ys_all, ps_all = [], []
    ys_by_task, ps_by_task = defaultdict(list), defaultdict(list)
    mean, std, lmbda = transform_stats
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            task_ids = data.task.view(-1).cpu().numpy()
            
            for y_hat, y_true, t_id in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy(), task_ids):
                if np.isnan(y_true): continue
                
                y_denorm = y_true * std + mean
                p_denorm = y_hat * std + mean
                
                try:
                    y_orig = inv_boxcox(y_denorm, lmbda)
                    p_orig = inv_boxcox(p_denorm, lmbda)
                except ValueError:
                    continue

                ys_all.append(y_orig)
                ps_all.append(p_orig)
                ys_by_task[int(t_id)].append(y_orig)
                ps_by_task[int(t_id)].append(p_orig)

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
        
        if t == 0:
            print(f"Plotting scatter for Task {t}...")
            task0_path = os.path.join(OUTPUT_DIR, f'task_{t}_scatter.png')
            plot_scatter(y_t_clean, p_t_clean, save_path=task0_path)
        
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
    seed_everything(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Step 1: Loading all raw data into a DataFrame...")
    all_raw_data = []

    for task_id, csv in enumerate(data_files):
       
        all_raw_data.extend(load_dataset_raw(csv, task_id))

    df_all = pd.DataFrame(all_raw_data, columns=['smiles', 'rt', 'task_id'])
    df_all['original_index'] = df_all.index

    print("Step 2: Splitting data per task with stratification on RT values...")
    final_idx_train, final_idx_val, final_idx_test = [], [], []

    for task_id, group in df_all.groupby('task_id'):
        num_samples = len(group)
        print(f"  - Processing Task {task_id} with {num_samples} samples...")

        task_indices = group['original_index'].values
        task_rt_values = group['rt'].values

        if num_samples < 5: 
            print(f"    [WARN] Task {task_id} has too few samples. Adding all to training set.")
            final_idx_train.extend(task_indices)
            continue

        try:

            strata = pd.qcut(pd.Series(task_rt_values), q=10, labels=False, duplicates='drop')

            try:
                idx_train_task, idx_temp_task, _, y_temp_task = train_test_split(
                    task_indices, strata, test_size=0.2, stratify=strata, random_state=42
                )
            except ValueError: 
                idx_train_task, idx_temp_task, _, y_temp_task = train_test_split(
                    task_indices, strata, test_size=0.2, random_state=42
                )

            try:
                idx_val_task, idx_test_task = train_test_split(
                    idx_temp_task, test_size=0.5, stratify=y_temp_task, random_state=42
                )
            except ValueError:
                idx_val_task, idx_test_task = train_test_split(
                    idx_temp_task, test_size=0.5, random_state=42
                )

        except Exception as e:

            print(f"    [WARN] Stratification failed for Task {task_id} due to '{e}'. Falling back to random split.")
            idx_train_task, idx_temp_task = train_test_split(task_indices, test_size=0.2, random_state=42)
            idx_val_task, idx_test_task = train_test_split(idx_temp_task, test_size=0.5, random_state=42)

        final_idx_train.extend(idx_train_task)
        final_idx_val.extend(idx_val_task)
        final_idx_test.extend(idx_test_task)

    idx_train, idx_val, idx_test = final_idx_train, final_idx_val, final_idx_test
    print(f"\nTotal data split: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    print("\nStep 3: Learning Box-Cox parameters from the training set ONLY...")
    train_rt_values = np.array([all_raw_data[i][1] for i in idx_train if all_raw_data[i][1] > 0])
    if len(train_rt_values) < 3:
        raise ValueError("Not enough positive data in the training set for Box-Cox transformation.")
    
    rt_transformed_train, train_lambda = boxcox(train_rt_values)
    train_mean = rt_transformed_train.mean()
    train_std = rt_transformed_train.std()
    transform_stats = (train_mean, train_std, train_lambda)
    print(f"Stats from Training Set: Lambda={train_lambda:.4f}, Mean={train_mean:.4f}, Std={train_std:.4f}")

    print("\nStep 4: Transforming all datasets using learned parameters...")
    all_data_pyg = []

    for smi, rt, task_id in tqdm(all_raw_data, desc="Creating PyG data"):
        if rt > 0:
            rt_bc = boxcox(rt, train_lambda)
            y_norm = (rt_bc - train_mean) / train_std
            all_data_pyg.append(smiles_to_data(smi, y_norm, task_id))
        else:
            all_data_pyg.append(smiles_to_data(smi, float('nan'), task_id))
            
    node_dim = atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).__len__()

    print("\nStep 5: Creating final datasets and dataloaders...")

    train_set = [all_data_pyg[i] for i in idx_train]
    val_set = [all_data_pyg[i] for i in idx_val]
    test_set = [all_data_pyg[i] for i in idx_test]

    train_set = [d for d in train_set if not torch.isnan(d.y)]
    val_set = [d for d in val_set if not torch.isnan(d.y)]
    test_set = [d for d in test_set if not torch.isnan(d.y)]
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = GINKANmultiRegressor(node_dim=node_dim,num_tasks=len(data_files), task_emb_dim=len(data_files), 
            hidden=64,layers=4,dropout=0.02).to(device)
    #model = GINmultiRegressor(node_dim=node_dim,num_tasks=len(data_files),task_emb_dim=len(data_files),
        #hidden=64,layers=4, dropout=0.02).to(device)
    #model = GATmultiRegressor(node_dim=node_dim,num_tasks=len(data_files),task_emb_dim=len(data_files),
        #hidden=64,layers=4, dropout=0.02).to(device)
    #model = GCNmultiRegressor(node_dim=node_dim,num_tasks=len(data_files),task_emb_dim=len(data_files),
        #hidden=64,layers=4, dropout=0.02).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = nn.L1Loss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None

    print("\n--- Starting Model Training ---")
    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        metrics = evaluate(model, val_loader, device, transform_stats)
        
        if metrics["MAE"] is not None and not math.isinf(metrics["MAE"]):
            scheduler.step(metrics["MAE"])
            if metrics["MAE"] < best_val:
                best_val = metrics["MAE"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nEvaluating on test set (best val checkpoint)...")
    overall_test, per_task_test, y_test, p_test = task_evaluate(model, test_loader, device, transform_stats, len(data_files))
    
    print("\n--- Overall Test Metrics ---")
    if overall_test:
        for metric, value in overall_test.items():
            print(f"{metric}: {value:.4f}")
    
    print("\n--- Per-Task Test Metrics (MAE, R2) ---")
    if per_task_test:
        for task_id, metrics in per_task_test.items():
            print(f"Task {task_id}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    print("\n--- Saving results to DataFrame with Summary Statistics ---")

    base_filename = f'{current_time_str}_per_task_performance-box-train'

    csv_name = f'{base_filename}.csv'
    output_csv_path = os.path.join(OUTPUT_DIR, csv_name)

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
        
        print("\n--- Final Evaluation Summary ---")
        print(results_df.round(4))
        print(f"\nResults summary successfully saved to {output_csv_path}")
    else:
        print("No per-task results to save.")

    param_path = os.path.join(OUTPUT_DIR, f'{base_filename}_boxcox_params.txt')
    try:
        with open(param_path, 'w') as f:
            f.write(f"lambda: {train_lambda}\n")
            f.write(f"mean: {train_mean}\n")
            f.write(f"std: {train_std}\n")
        print(f"Box-Cox transformation parameters saved to {param_path}")
    except Exception as e:
        print(f"[ERROR] Could not save Box-Cox parameters: {e}")

    png_name = f'{base_filename}.png'
    output_png_path = os.path.join(OUTPUT_DIR, png_name)
    plot_scatter(y_test, p_test, output_png_path)