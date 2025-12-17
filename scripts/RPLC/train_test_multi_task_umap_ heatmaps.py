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
import warnings

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

# ---- Visualization Imports ----
import seaborn as sns
try:
    import umap
except ImportError:
    import umap.umap_ as umap

# ---- PyTorch Geometric ----
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import rdchem

# ---- Models ----
from models import GINKANmultiRegressor, GINmultiRegressor, GATmultiRegressor, GCNmultiRegressor
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Directory Setup
# ==========================================
BASE_PATH = 'D:/Uni-RT/'
EXP_MODE = 'RPLC-GINKAN' 
current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_DIR = os.path.join(BASE_PATH, 'Global_results', EXP_MODE, current_time_str)

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
# Utils & Featurization
# ------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ATOMIC_NUM_SET = list(range(1, 40))
HYBRIDIZATION_SET = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2]
BOND_TYPE_SET = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]

def one_hot(x, allowed):
    return [int(x == s) for s in allowed] + [int(x not in allowed)]

def atom_features(atom: rdchem.Atom) -> List[int]:
    return (one_hot(atom.GetAtomicNum(), ATOMIC_NUM_SET) + one_hot(atom.GetHybridization(), HYBRIDIZATION_SET) + [atom.GetTotalDegree(), atom.GetTotalNumHs(), atom.GetFormalCharge(), int(atom.GetIsAromatic()), int(atom.IsInRing()), int(atom.GetChiralTag() != rdchem.ChiralType.CHI_UNSPECIFIED)])

def bond_features(bond: rdchem.Bond) -> List[int]:
    if bond is None: return [0] * (len(BOND_TYPE_SET) + 3)
    bt = bond.GetBondType()
    return (one_hot(bt, BOND_TYPE_SET) + [int(bond.GetIsConjugated()), int(bond.IsInRing()), int(bond.GetStereo() != rdchem.BondStereo.STEREONONE)])

def smiles_to_data(smiles, y, task_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError(f"Invalid SMILES: {smiles}")
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
    data.y = torch.tensor([y], dtype=torch.float) if y is not None and not (isinstance(y, float) and math.isnan(y)) else torch.tensor([float('nan')], dtype=torch.float)
    data.task = torch.tensor([int(task_id)], dtype=torch.long) if task_id is not None else None
    return data

def load_dataset_raw(csv_path: str, task_id: int) -> List[Tuple[str, float, int]]:
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
# Training / Evaluation / Plotting
# ------------------------------
def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(data)
        valid_mask = ~torch.isnan(data.y)
        if valid_mask.sum() == 0: continue
        loss = loss_fn(pred[valid_mask], data.y[valid_mask].view(-1)).mean()
        if torch.isnan(loss): continue
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
            for y_hat, y_true in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy()):
                if np.isnan(y_true): continue
                try:
                    pred_final = inv_boxcox(y_hat * std + mean, lmbda)
                    y_final = inv_boxcox(y_true * std + mean, lmbda)
                    ys.append(y_final)
                    ps.append(pred_final)
                except ValueError: continue
    if not ys: return {"MAE": float('inf'), "RMSE": float('inf'), "R2": -float('inf')}
    y, p = np.array(ys), np.array(ps)
    valid = np.isfinite(y) & np.isfinite(p)
    if np.sum(valid) < 2: return {"MAE": float('inf'), "RMSE": float('inf'), "R2": -float('inf')}
    y, p = y[valid], p[valid]
    return {"MAE": mean_absolute_error(y, p), "RMSE": math.sqrt(((y - p) ** 2).mean()), "R2": r2_score(y, p)}

def plot_scatter(y_true, y_pred, save_path=None, title=""):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if len(y_true) < 2: return
    xy = np.vstack([y_true, y_pred])
    try: z = gaussian_kde(xy)(xy)
    except: z = np.ones_like(y_true)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
    sc = ax.scatter(y_true, y_pred, c=z, s=15, cmap='viridis', alpha=0.8)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, 'k--', alpha=0.75)
    ax.text(0.05, 0.95, f'$R^2$={r2_score(y_true, y_pred):.3f}\nMAE={mean_absolute_error(y_true, y_pred):.3f}', transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
    ax.set_xlabel("True RT (s)"); ax.set_ylabel("Predicted RT (s)"); ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='Density')
    if save_path: plt.savefig(save_path, dpi=300); plt.close(fig)

def task_evaluate(model, loader, device, transform_stats, num_tasks, save_dir=None):
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
                try:
                    y_orig = inv_boxcox(y_true * std + mean, lmbda)
                    p_orig = inv_boxcox(y_hat * std + mean, lmbda)
                    ys_all.append(y_orig); ps_all.append(p_orig)
                    ys_by_task[int(t_id)].append(y_orig); ps_by_task[int(t_id)].append(p_orig)
                except: continue
    y, p = np.array(ys_all), np.array(ps_all)
    valid = np.isfinite(y) & np.isfinite(p)
    y_clean, p_clean = y[valid], p[valid]
    overall = {"MAE": mean_absolute_error(y_clean, p_clean), "MedAE": median_absolute_error(y_clean, p_clean), "R2": r2_score(y_clean, p_clean)} if len(y_clean)>0 else {}
    per_task = {}
    for t in range(num_tasks):
        yt, pt = np.array(ys_by_task[t]), np.array(ps_by_task[t])
        valid_t = np.isfinite(yt) & np.isfinite(pt)
        yt, pt = yt[valid_t], pt[valid_t]
        if len(yt) < 2: continue
        if t == 0: plot_scatter(yt, pt, save_path=os.path.join(save_dir, f'task_{t}_scatter.png') if save_dir else None, title=f"Task {t}")
        per_task[t] = {"MAE": mean_absolute_error(yt, pt), "R2": r2_score(yt, pt)}
    return overall, per_task, y, p

# ==========================================
# Feature Extraction (Updated for GINKAN)
# ==========================================
def _extract_pooled_features(model, loader, device):
    """Common extraction: Embedding -> Backbone -> Pool"""
    model.eval()
    features, y_norm, tasks = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # GINKAN usually follows the same forward pass logic for feature extraction
            h = model.embed(data.x)
            for conv, bn in zip(model.convs, model.bns):
                h = conv(h, data.edge_index)
                h = bn(h)
                h = F.relu(h)
            g = global_add_pool(h, data.batch)
            features.append(g.cpu().numpy())
            y_norm.append(data.y.view(-1).cpu().numpy())
            tasks.append(data.task.view(-1).cpu().numpy())
    return np.concatenate(features), np.concatenate(y_norm), np.concatenate(tasks)

def _denorm(y_norm, transform_stats):
    mean, std, lmbda = transform_stats
    rts = []
    for y in y_norm:
        if np.isnan(y): rts.append(np.nan)
        else:
            try: rts.append(inv_boxcox(y * std + mean, lmbda))
            except: rts.append(np.nan)
    return np.array(rts)

def extract_pre_adapter_features(model, loader, device, stats):
    """Features from GINKAN backbone before any adapter/film"""
    feats, y_norm, tasks = _extract_pooled_features(model, loader, device)
    rts = _denorm(y_norm, stats)
    mask = ~np.isnan(rts)
    return feats[mask], rts[mask], tasks[mask]

def extract_adapter_crossstitch_features(model, loader, device, stats):
    """Features after Adapter + CrossStitch"""
    model.eval()
    features, y_norm, tasks = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            h = model.embed(data.x)
            for conv, bn in zip(model.convs, model.bns):
                h = F.relu(bn(conv(h, data.edge_index)))
            g = global_add_pool(h, data.batch)
            graph_tasks = data.task.view(-1)
            
            if hasattr(model, 'adapters'):
                g_adj = torch.zeros_like(g)
                for t in range(model.num_tasks):
                    m = (graph_tasks == t)
                    if m.sum() > 0: g_adj[m] = model.adapters[t](g[m])
                g = g_adj
            if hasattr(model, 'use_cross_stitch') and model.use_cross_stitch:
                g = model.cross_stitch(g, graph_tasks)
            
            features.append(g.cpu().numpy())
            y_norm.append(data.y.view(-1).cpu().numpy())
            tasks.append(graph_tasks.cpu().numpy())
    
    feats = np.concatenate(features)
    rts = _denorm(np.concatenate(y_norm), stats)
    ts = np.concatenate(tasks)
    mask = ~np.isnan(rts)
    return feats[mask], rts[mask], ts[mask]

def extract_film_features(model, loader, device, stats):
    """Features after FiLM modulation"""
    model.eval()
    features, y_norm, tasks = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            h = model.embed(data.x)
            for conv, bn in zip(model.convs, model.bns):
                h = F.relu(bn(conv(h, data.edge_index)))
            g = global_add_pool(h, data.batch)
            graph_tasks = data.task.view(-1)
            
            if hasattr(model, 'adapters'):
                g_adj = torch.zeros_like(g)
                for t in range(model.num_tasks):
                    m = (graph_tasks == t)
                    if m.sum() > 0: g_adj[m] = model.adapters[t](g[m])
                g = g_adj
            if hasattr(model, 'use_cross_stitch') and model.use_cross_stitch:
                g = model.cross_stitch(g, graph_tasks)
            
            task_emb = model.task_embed(graph_tasks)
            gamma1, beta1 = model.head_film_gen_1(task_emb)
            gamma2, beta2 = model.head_film_gen_2(task_emb)
            # Assuming GINKAN head structure is similar to standard GIN for film modulation
            g_film = gamma2 * (F.relu(model.head.fc1(gamma1 * g + beta1))) + beta2
            
            features.append(g_film.cpu().numpy())
            y_norm.append(data.y.view(-1).cpu().numpy())
            tasks.append(graph_tasks.cpu().numpy())
            
    feats = np.concatenate(features)
    rts = _denorm(np.concatenate(y_norm), stats)
    ts = np.concatenate(tasks)
    mask = ~np.isnan(rts)
    return feats[mask], rts[mask], ts[mask]

def extract_film_params(model, device="cpu"):
    model.eval(); model = model.to(device)
    params = {}
    with torch.no_grad():
        for t in range(model.num_tasks):
            te = model.task_embed(torch.tensor([t], device=device))
            g1, b1 = model.head_film_gen_1(te)
            g2, b2 = model.head_film_gen_2(te)
            params[t] = {"gamma1": g1.cpu().numpy().flatten(), "beta1": b1.cpu().numpy().flatten(),
                         "gamma2": g2.cpu().numpy().flatten(), "beta2": b2.cpu().numpy().flatten()}
    return params

def extract_task_embeddings(model, device="cpu"):
    model.eval(); model = model.to(device)
    embs = {}
    with torch.no_grad():
        for t in range(model.num_tasks):
            embs[t] = model.task_embed(torch.tensor([t], device=device)).cpu().numpy().flatten()
    return embs

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    
    batch_size = 32
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Step 1: Loading raw data...")
    all_raw_data = []
    for task_id, csv in enumerate(data_files):
        all_raw_data.extend(load_dataset_raw(csv, task_id))

    # 2. Split
    print("Step 2: Splitting indices...")
    all_rt = np.array([d[1] for d in all_raw_data])
    idx = np.arange(len(all_raw_data))
    q_val = min(10, len(np.unique(all_rt[all_rt > 0])) - 1)
    if q_val < 2:
        idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    else:
        quantiles = pd.qcut(pd.Series(all_rt), q=q_val, labels=False, duplicates='drop')
        try:
            idx_train, idx_temp = train_test_split(idx, test_size=0.2, stratify=quantiles, random_state=42)
            idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=pd.qcut(pd.Series(all_rt[idx_temp]), q=q_val, labels=False, duplicates='drop'), random_state=42)
        except:
             idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=42)
             idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)

    # 3. Learn Stats
    print("Step 3: Learning Box-Cox from Training Set...")
    train_rt = np.array([all_raw_data[i][1] for i in idx_train if all_raw_data[i][1] > 0])
    rt_bc_train, train_lambda = boxcox(train_rt)
    train_mean, train_std = rt_bc_train.mean(), rt_bc_train.std()
    transform_stats = (train_mean, train_std, train_lambda)
    print(f"Stats: Lambda={train_lambda:.4f}, Mean={train_mean:.4f}, Std={train_std:.4f}")

    # 4. Transform
    print("Step 4: Transforming data...")
    all_data = []
    for smi, rt, tid in tqdm(all_raw_data):
        if rt > 0:
            y_norm = (boxcox(rt, train_lambda) - train_mean) / train_std
            all_data.append(smiles_to_data(smi, y_norm, tid))
        else: all_data.append(smiles_to_data(smi, float('nan'), tid))
    
    node_dim = atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).__len__()
    train_loader = DataLoader([all_data[i] for i in idx_train if not torch.isnan(all_data[i].y)], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader([all_data[i] for i in idx_val if not torch.isnan(all_data[i].y)], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader([all_data[i] for i in idx_test if not torch.isnan(all_data[i].y)], batch_size=batch_size, shuffle=False)

    # 5. Model
    # Explicitly using GINKANmultiRegressor
    model = GINKANmultiRegressor(node_dim=node_dim, num_tasks=len(data_files), task_emb_dim=len(data_files), hidden=64, layers=4, dropout=0.02).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    
    # *** FIX IS HERE ***
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    loss_fn = nn.L1Loss(reduction='none')

    # 6. Train
    print("\n--- Training ---")
    best_val, best_state = float('inf'), None
    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        metrics = evaluate(model, val_loader, device, transform_stats)
        if metrics["MAE"] < best_val:
            best_val = metrics["MAE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        scheduler.step(metrics["MAE"])
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val MAE: {metrics['MAE']:.4f}")

    if best_state: model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(SAVE_DIR, 'best_model.pth'))

    # 7. Evaluate
    print("\n--- Evaluation ---")
    overall, per_task, y_test, p_test = task_evaluate(model, test_loader, device, transform_stats, len(data_files), SAVE_DIR)
    
    res_df = pd.DataFrame.from_dict(per_task, orient='index')
    if overall: res_df.loc['Overall'] = overall
    res_df.to_csv(os.path.join(SAVE_DIR, f'{current_time_str}_results.csv'))
    plot_scatter(y_test, p_test, os.path.join(SAVE_DIR, 'overall_scatter.png'), "Overall Test")
    
    # ---------------------------------------------------------
    # 8. UMAP Visualization (Modified to show only 5 tasks)
    # ---------------------------------------------------------
    print("\n--- Visualizing (Filtering for specific tasks) ---")
    

    dnames = [os.path.basename(f).split('_')[0] for f in data_files]    

    f_gin, r_gin, t_gin = extract_pre_adapter_features(model, test_loader, device, transform_stats)
    f_ad, r_ad, t_ad = extract_adapter_crossstitch_features(model, test_loader, device, transform_stats)
    f_film, r_film, t_film = extract_film_features(model, test_loader, device, transform_stats)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']    
    data_umap = [(f_gin, r_gin, t_gin, "GINKAN"),
                 (f_ad, r_ad, t_ad, "GINKAN + Adapter CrossStitch"),
                 (f_film, r_film, t_film, "Full UniRT")]
    target_tasks_to_show = [9, 10, 11, 12,13]
    print(f"Plotting only tasks: {target_tasks_to_show}")
    for ax, (feats, rts, tids, title) in zip(axes, data_umap):
        if len(feats) < 15: 
            continue            
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        emb = reducer.fit_transform(feats)        

        unique_tasks = np.unique(tids)        
        for i, t in enumerate(unique_tasks):
        
            if int(t) not in target_tasks_to_show:
                continue           
            mask = tids == t
            if mask.sum() == 0: continue            
            label_name = dnames[int(t)] if int(t) < len(dnames) else str(t)           
            sc = ax.scatter(
                emb[mask, 0], 
                emb[mask, 1], 
                c=rts[mask], 
                cmap='viridis', 
                s=30,     
                alpha=0.8, 
                marker=markers[int(t) % len(markers)], 
                label=label_name, 
                edgecolor='k', 
                linewidth=0.1)
        ax.set_title(title)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Retention Time (s)')

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=5)    
    plt.tight_layout(rect=[0, 0.05, 1, 1])   
    save_path = os.path.join(SAVE_DIR, 'umap_ginkan_'+'_filtered.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Filtered UMAP plot saved to: {save_path}")
      
    # 9. Heatmaps
    tembs = extract_task_embeddings(model, device)
    film_p = extract_film_params(model, device)
    
    t_vecs = np.stack([tembs[t] for t in sorted(tembs)])
    g_vecs = np.stack([np.concatenate([film_p[t]['gamma1'], film_p[t]['gamma2']]) for t in sorted(film_p)])
    b_vecs = np.stack([np.concatenate([film_p[t]['beta1'], film_p[t]['beta2']]) for t in sorted(film_p)])

    fig, axs = plt.subplots(1, 3, figsize=(24, 7), dpi=300)
    for ax, mat, name, cmap in zip(axs, [t_vecs, g_vecs, b_vecs], ["Semantic Embedding", "Gamma Params", "Beta Params"], ['plasma', 'Blues', 'Greens']):
        if len(mat) > 1:
            sns.heatmap(np.corrcoef(mat), annot=True, fmt=".2f", ax=ax, cmap=cmap, square=True, xticklabels=dnames, yticklabels=dnames)
            ax.set_title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'heatmaps_ginkan.png'), dpi=300)
    
    plt.close()
    
    print(f"Done! Results in {SAVE_DIR}")