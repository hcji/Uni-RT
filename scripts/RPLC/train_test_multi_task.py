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
# ==========================================
# 1. Configuration & Directory Setup
# ==========================================

BASE_PATH = 'D:/Uni-RT/'

EXP_MODE = 'RPLC-GINKAN' 

current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

SAVE_DIR = os.path.join(BASE_PATH, 'Multi_results', EXP_MODE, current_time_str)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"========================================================")
print(f"Experiment Mode: {EXP_MODE}")
print(f"All results will be saved to: {SAVE_DIR}")
print(f"========================================================")

# --- data files ---
data_files = [
    BASE_PATH+'data/0019/0019_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0052/0052_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0179/0179_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0180/0180_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0234/0234_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0235/0235_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0260/0260_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0261/0261_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0264/0264_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0317/0317_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0319/0319_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0321/0321_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0323/0323_rtdata_canonical_success.tsv',
    BASE_PATH+'data/0331/0331_rtdata_canonical_success.tsv'
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

def task_evaluate(model, loader, device, transform_stats, num_tasks=28, save_dir=None):
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
        
        # 保存特定任务的散点图到指定目录
        if t == 0:
            print(f"Plotting scatter for Task {t}...")
            fname = f'task_{t}_scatter.png'
            if save_dir:
                sp = os.path.join(save_dir, fname)
            else:
                sp = fname
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
# Prediction Function for New Molecules
# ------------------------------
def predict_moleculesRT(model, smiles_list, target_task_id, transform_stats, device, ids=None):
    """
    Predict RT for a list of SMILES under a specific task condition.
    """
    model.eval()
    mean, std, lmbda = transform_stats
    results = []
    
    # Create PyG Data objects
    data_list = []
    valid_indices = []
    
    print(f"Preparing graphs for {len(smiles_list)} molecules...")
    for idx, smi in enumerate(tqdm(smiles_list)):
        try:
            # We don't know Y, so pass None. Task ID is the target condition.
            d = smiles_to_data(smi, None, target_task_id)
            if d is not None:
                data_list.append(d)
                valid_indices.append(idx)
            else:
                results.append({'ID': ids[idx] if ids else idx, 'SMILES': smi, 'Predicted_RT': None, 'Note': 'Invalid SMILES'})
        except Exception as e:
             results.append({'ID': ids[idx] if ids else idx, 'SMILES': smi, 'Predicted_RT': None, 'Note': f'Error: {str(e)}'})

    if not data_list:
        return pd.DataFrame(results)

    loader = DataLoader(data_list, batch_size=64, shuffle=False)
    
    print(f"Predicting for Task ID {target_task_id}...")
    preds_collected = []
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.to(device)
            # Force the task ID to be the target task (broadcast to batch size)
            # Although smiles_to_data sets it, we ensure it matches the batch structure here if needed
            # The model reads data.task, which is correctly set by smiles_to_data
            
            raw_pred = model(data)
            
            # Inverse Transform
            raw_pred_np = raw_pred.cpu().numpy()
            for val in raw_pred_np:
                # Denormalize
                val_denorm = val * std + mean
                # Inverse Box-Cox
                try:
                    val_final = inv_boxcox(val_denorm, lmbda)
                except:
                    val_final = np.nan
                preds_collected.append(val_final)
    
    # Merge valid results back
    ptr = 0
    final_output = []
    
    # We need to map back to original input order.
    # We already added failures to 'results'. Now we add successes.
    # A cleaner way: Re-iterate original list.
    
    success_map = {valid_indices[i]: preds_collected[i] for i in range(len(valid_indices))}
    
    for idx, smi in enumerate(smiles_list):
        row_id = ids[idx] if ids else idx
        if idx in success_map:
            final_output.append({'ID': row_id, 'SMILES': smi, 'Predicted_RT': success_map[idx], 'Note': 'Success'})
        else:
            # Already handled in first loop or skipped
            # Check if it was added to results (failures)
            found = False
            for r in results:
                if r['ID'] == row_id:
                    final_output.append(r)
                    found = True
                    break
            if not found:
                 final_output.append({'ID': row_id, 'SMILES': smi, 'Predicted_RT': None, 'Note': 'Unknown Error'})

    return pd.DataFrame(final_output)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    batch_size = 32
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    print("Step 1: Loading all raw data...")
    all_raw_data = []
    for task_id, csv in enumerate(data_files):
        all_raw_data.extend(load_dataset_raw(csv, task_id))

    print("Step 2: Splitting data indices before transformation...")
    all_rt_values_raw = np.array([rt for _, rt, _ in all_raw_data])
    idx = np.arange(len(all_raw_data))

    quantiles = pd.qcut(pd.Series(all_rt_values_raw), q=10, labels=False, duplicates='drop')
    
    try:
        idx_train, idx_temp, _, y_temp = train_test_split(idx, quantiles, test_size=0.2, stratify=quantiles, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_temp, random_state=42)
    except ValueError:
        print("[WARN] Stratified split failed. Using random split.")
        idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

    print("Step 3: Learning Box-Cox parameters from the training set ONLY...")
    train_rt_values = np.array([all_raw_data[i][1] for i in idx_train if all_raw_data[i][1] > 0])
    if len(train_rt_values) < 3:
        raise ValueError("Not enough positive data in the training set for Box-Cox transformation.")
    
    rt_transformed_train, train_lambda = boxcox(train_rt_values)
    train_mean = rt_transformed_train.mean()
    train_std = rt_transformed_train.std()
    transform_stats = (train_mean, train_std, train_lambda) # 这是我们唯一使用的统计数据
    print(f"Stats from Training Set: Lambda={train_lambda:.4f}, Mean={train_mean:.4f}, Std={train_std:.4f}")

    print("Step 4: Transforming all datasets using learned parameters...")
    all_data = []
    for smi, rt, task_id in tqdm(all_raw_data, desc="Creating PyG data"):
        if rt > 0:
            rt_bc = boxcox(rt, train_lambda)
            y_norm = (rt_bc - train_mean) / train_std
            all_data.append(smiles_to_data(smi, y_norm, task_id))
        else:
            all_data.append(smiles_to_data(smi, float('nan'), task_id))
            
    node_dim = atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).__len__()

    print("Step 5: Creating final datasets and dataloaders...")
    train_set = [all_data[i] for i in idx_train]
    val_set = [all_data[i] for i in idx_val]
    test_set = [all_data[i] for i in idx_test]

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
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | "
                  f"Val: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model_save_path = os.path.join(SAVE_DIR, 'best_model.pth')
        torch.save(best_state, model_save_path)
        print(f"Best model weights saved to: {model_save_path}")

    print("\nEvaluating on test set (best val checkpoint)...")
    # 传入 save_dir
    overall_test, per_task_test, y_test, p_test = task_evaluate(
        model, test_loader, device, transform_stats, len(data_files), save_dir=SAVE_DIR
    )
    
    print("\n--- Overall Test Metrics ---")
    if overall_test:
        for metric, value in overall_test.items():
            print(f"{metric}: {value:.4f}")
    
    print("\n--- Per-Task Test Metrics (MAE, R2) ---")
    if per_task_test:
        for task_id, metrics in per_task_test.items():
            print(f"Task {task_id}: MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    print("\n--- Saving results to DataFrame with Summary Statistics ---")
    
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
        
        print("\n--- Final Evaluation Summary ---")
        print(results_df.round(4))
        
        print(f"\nResults summary successfully saved to {output_csv_path}")
    else:
        print("No per-task results to save.")
    output_png_filename = f'{current_time_str}_overall_scatter.png'
    output_png_path = os.path.join(SAVE_DIR, output_png_filename)
    plot_scatter(y_test, p_test, output_png_path)

    print("\n--- Saving transformation statistics ---")
    stats_dict = {
        "boxcox_lambda": float(train_lambda),
        "mean": float(train_mean),
        "std": float(train_std)
    }
    stats_path = os.path.join(SAVE_DIR, 'transform_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=4)
    print(f"Transformation statistics saved to: {stats_path}")
    # =========================================================
    print("\n--- Running Auxiliary Identification Prediction ---")
    

    TARGET_PREDICT_TASK_ID = 14
    EXTERNAL_PREDICT_FILE=''
    
    TARGET_SMILES_COL = 'smiles.std'

    if EXTERNAL_PREDICT_FILE and os.path.exists(EXTERNAL_PREDICT_FILE):
        print(f"Reading external file: {EXTERNAL_PREDICT_FILE}")
        
        try:
          
            df_ext = pd.read_excel(EXTERNAL_PREDICT_FILE)
                   
            if TARGET_SMILES_COL in df_ext.columns:
                print(f"Found column '{TARGET_SMILES_COL}'. Preparing for prediction...")
     
                valid_mask = df_ext[TARGET_SMILES_COL].notna() & (df_ext[TARGET_SMILES_COL].astype(str).str.strip() != '')
                
                smiles_list = df_ext.loc[valid_mask, TARGET_SMILES_COL].astype(str).tolist()
                ids_list = df_ext.loc[valid_mask].index.tolist()
                
                if smiles_list:
                    print(f"Predicting for {len(smiles_list)} molecules on Task ID {TARGET_PREDICT_TASK_ID}...")

                    pred_res = predict_moleculesRT(model, smiles_list, TARGET_PREDICT_TASK_ID, transform_stats, device, ids=ids_list)

                    rt_mapping = dict(zip(pred_res['ID'], pred_res['Predicted_RT']))

                    new_col_name = f'Predicted_RT_Task{TARGET_PREDICT_TASK_ID}'

                    df_ext[new_col_name] = df_ext.index.map(rt_mapping)

                    try:
                        df_ext.to_excel(EXTERNAL_PREDICT_FILE, index=False)
                        print(f"Success! Predictions saved to column '{new_col_name}' in: {EXTERNAL_PREDICT_FILE}")
                    except PermissionError:
                        print(f"[Error] Permission denied: Could not write to {EXTERNAL_PREDICT_FILE}.")
                        print("Please close the Excel file if it is currently open and try again.")
                    except Exception as e:
                        print(f"[Error] Failed to save file: {e}")
                        
                else:
                    print(f"No valid SMILES found in column '{TARGET_SMILES_COL}'.")
            else:
                print(f"Error: Column '{TARGET_SMILES_COL}' not found in the Excel file.")
                print(f"Available columns: {list(df_ext.columns)}")
                
        except Exception as e:
            print(f"An error occurred while processing the file: {e}")
            
    else:
        print(f"Warning: External file path not found or set to None: {EXTERNAL_PREDICT_FILE}")

    print("\nAll tasks completed.")