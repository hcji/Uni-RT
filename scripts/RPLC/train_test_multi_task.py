# -*- coding: utf-8 -*-
"""
Simplified GIN + FiLM + Adapter + optional Cross-Stitch
Mordred descriptor input removed, only graph features are used.

@author: 纪宏超
"""

import os
import math
import random
from typing import List, Tuple, Optional
from collections import defaultdict
import umap
import matplotlib.cm as cm
import seaborn as sns

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

data_name = [s.split('/')[1] for s in data_files]

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
    if y is not None and not (isinstance(y, float) and math.isnan(y)):
        data.y = torch.tensor([y], dtype=torch.float)
    data.smiles = smiles
    if task_id is not None:
        data.task = torch.tensor([int(task_id)], dtype=torch.long)
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
# FiLM, Adapter, CrossStitch
# ------------------------------
class FiLMParamGenerator(nn.Module):
    def __init__(self, task_emb_dim: int, hidden: int, scale_gamma: float = 0.5, scale_beta: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(task_emb_dim, task_emb_dim),
            nn.ReLU(),
            nn.Linear(task_emb_dim, 2 * hidden)
        )
        self.scale_gamma = nn.Parameter(torch.tensor(scale_gamma), requires_grad=False)
        self.scale_beta  = nn.Parameter(torch.tensor(scale_beta), requires_grad=False)

    def forward(self, task_emb: torch.Tensor) -> tuple:
        out = self.net(task_emb)
        gamma_raw, beta_raw = out.chunk(2, dim=-1)
        gamma = 1.0 + self.scale_gamma * torch.tanh(gamma_raw)
        beta  = self.scale_beta * torch.tanh(beta_raw)
        return gamma, beta

class Adapter(nn.Module):
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.down = nn.Linear(hidden_dim, hidden_dim // reduction)
        self.up = nn.Linear(hidden_dim // reduction, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, h):
        return self.up(self.relu(self.down(h))) + h

class CrossStitch(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = nn.Parameter(torch.eye(num_tasks))

    def forward(self, h_task, graph_tasks):
        h_new = torch.zeros_like(h_task)
        for i in range(self.num_tasks):
            mask_i = (graph_tasks == i)
            if mask_i.sum() == 0:
                continue
            h_i = h_task[mask_i]
            fused = 0.0
            for j in range(self.num_tasks):
                mask_j = (graph_tasks == j)
                if mask_j.sum() == 0:
                    continue
                h_j = h_task[mask_j].mean(dim=0, keepdim=True)
                fused += self.alpha[i, j] * h_j
            h_new[mask_i] = h_i + fused
        return h_new

# ------------------------------
# Simplified MLP with FiLM only
# ------------------------------
class MLPWithFiLM(nn.Module):
    def __init__(self, in_dim, hidden, out_dim,
                 film_generator_1: FiLMParamGenerator,
                 film_generator_2: FiLMParamGenerator,
                 dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.film_generator_1 = film_generator_1
        self.film_generator_2 = film_generator_2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x, task_emb):
        gamma1, beta1 = self.film_generator_1(task_emb)
        x = gamma1 * x + beta1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        gamma2, beta2 = self.film_generator_2(task_emb)
        x = gamma2 * x + beta2
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------------
# GINRegressor without Mordred
# ------------------------------
class GINRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_tasks: int = 28,
        task_emb_dim: int = 32,
        hidden: int = 64,
        layers: int = 4,
        dropout: float = 0.1,
        adapter_reduction: int = 4,
        use_cross_stitch: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch

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

        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)

        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)

        self.head = MLPWithFiLM(
            in_dim=hidden,
            hidden=hidden,
            out_dim=1,
            film_generator_1=self.head_film_gen_1,
            film_generator_2=self.head_film_gen_2,
            dropout=dropout
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)

        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)

        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = (graph_tasks == t_id)
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted

        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)

        task_emb = self.task_embed(graph_tasks)
        out = self.head(g, task_emb)
        return out.view(-1)


class GCNRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_tasks: int = 28,
        task_emb_dim: int = 32,
        hidden: int = 64,
        layers: int = 4,
        dropout: float = 0.1,
        adapter_reduction: int = 4,
        use_cross_stitch: bool = True,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch

        # node embedding
        self.embed = nn.Linear(node_dim, hidden)

        # GCN layers + BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))

        # task embedding + FiLM head + adapters / cross-stitch as before
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)

        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)

        self.head = MLPWithFiLM(
            in_dim=hidden,
            hidden=hidden,
            out_dim=1,
            film_generator_1=self.head_film_gen_1,
            film_generator_2=self.head_film_gen_2,
            dropout=dropout
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)

        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
        g = global_add_pool(h, batch)

        # adapters per task
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = (graph_tasks == t_id)
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted

        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)

        task_emb = self.task_embed(graph_tasks)
        out = self.head(g, task_emb)
        return out.view(-1)


class GATRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_tasks: int = 28,
        task_emb_dim: int = 32,
        hidden: int = 64,
        layers: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        adapter_reduction: int = 4,
        use_cross_stitch: bool = True,
    ):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads for this GAT design"
        self.hidden = hidden
        self.layers = layers
        self.num_tasks = num_tasks
        self.use_cross_stitch = use_cross_stitch
        self.heads = heads

        # node embedding
        self.embed = nn.Linear(node_dim, hidden)

        # GAT layers + BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # we set out_channels = hidden // heads and concat=True so output dim = out_channels * heads = hidden
        for _ in range(layers):
            self.convs.append(GATConv(hidden, hidden // heads, heads=heads, concat=True))
            self.bns.append(BatchNorm(hidden))

        # task embedding + FiLM head + adapters / cross-stitch as before
        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)
        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)

        self.adapters = nn.ModuleList([Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)])
        if use_cross_stitch:
            self.cross_stitch = CrossStitch(num_tasks)

        self.head = MLPWithFiLM(
            in_dim=hidden,
            hidden=hidden,
            out_dim=1,
            film_generator_1=self.head_film_gen_1,
            film_generator_2=self.head_film_gen_2,
            dropout=dropout
        )

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_tasks = data.task.view(-1).to(batch.device)

        h = self.embed(x)
        for conv, bn in zip(self.convs, self.bns):
            # GATConv returns shape [N, hidden] because we set concat=True and hidden//heads * heads == hidden
            h = conv(h, edge_index)
            h = bn(h)
            h = F.elu(h)  # GAT typically uses elu
        g = global_add_pool(h, batch)

        # adapters per task
        g_adjusted = torch.zeros_like(g)
        for t_id in range(self.num_tasks):
            mask = (graph_tasks == t_id)
            if mask.sum() > 0:
                g_adjusted[mask] = self.adapters[t_id](g[mask])
        g = g_adjusted

        if self.use_cross_stitch:
            g = self.cross_stitch(g, graph_tasks)

        task_emb = self.task_embed(graph_tasks)
        out = self.head(g, task_emb)
        return out.view(-1)



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

def evaluate(model, loader, device, task_stats):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            task_ids = data.task.view(-1).cpu().numpy()
            pred_rescaled, y_rescaled = [], []
            for y_hat, y_true, t_id in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy(), task_ids):
                mean, std = task_stats[int(t_id)]
                pred_rescaled.append(y_hat * std + mean)
                y_rescaled.append(y_true * std + mean)
            ys.append(np.array(y_rescaled))
            ps.append(np.array(pred_rescaled))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    mae = mean_absolute_error(y, p)
    rmse = math.sqrt(((y - p) ** 2).mean())
    r2 = r2_score(y, p)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def task_evaluate(model, loader, device, task_stats, num_tasks=28):
    model.eval()
    ys_all, ps_all = [], []
    ys_by_task, ps_by_task = defaultdict(list), defaultdict(list)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            task_ids = data.task.view(-1).cpu().numpy()
            
            for y_hat, y_true, t_id in zip(pred.cpu().numpy(), data.y.view(-1).cpu().numpy(), task_ids):
                # 反归一化
                mean, std = task_stats[int(t_id)]
                y_orig = y_true * std + mean
                p_orig = y_hat * std + mean

                ys_all.append(y_orig)
                ps_all.append(p_orig)
                ys_by_task[int(t_id)].append(y_orig)
                ps_by_task[int(t_id)].append(p_orig)

    # 转为 numpy 数组
    y = np.array(ys_all)
    p = np.array(ps_all)
    
    # 避免除零误差，MRE中y为0时会被忽略
    non_zero_mask = y != 0
    if np.sum(non_zero_mask) > 0:
        mre = np.mean(np.abs((y[non_zero_mask] - p[non_zero_mask]) / y[non_zero_mask]))
    else:
        mre = np.nan

    # Overall metrics
    overall = {
        "MAE": mean_absolute_error(y, p),
        "MedAE": median_absolute_error(y, p),
        "MRE": mre,
        "R2": r2_score(y, p),
    }

    # Per-task metrics
    per_task = {}
    for t in range(num_tasks):
        if len(ys_by_task[t]) == 0:
            continue
        
        y_t = np.array(ys_by_task[t])
        p_t = np.array(ps_by_task[t])
        
        non_zero_mask_t = y_t != 0
        if np.sum(non_zero_mask_t) > 0:
            mre_t = np.mean(np.abs((y_t[non_zero_mask_t] - p_t[non_zero_mask_t]) / y_t[non_zero_mask_t]))
        else:
            mre_t = np.nan

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


def extract_film_features(model, loader, device, task_stats):
    model.eval()
    features, rts, task_ids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_tasks = data.task.view(-1).to(batch.device)

            # 前向传播至 pooling 前
            h = model.embed(x)
            for conv, bn in zip(model.convs, model.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
            g = global_add_pool(h, batch)

            # Adapter + Cross-Stitch
            g_adjusted = torch.zeros_like(g)
            for t_id in range(model.num_tasks):
                mask = (graph_tasks == t_id)
                if mask.sum() > 0:
                    g_adjusted[mask] = model.adapters[t_id](g[mask])
            g = g_adjusted
            if model.use_cross_stitch:
                g = model.cross_stitch(g, graph_tasks)

            # Task embedding 与 FiLM 调制
            task_emb = model.task_embed(graph_tasks)
            gamma1, beta1 = model.head_film_gen_1(task_emb)
            gamma2, beta2 = model.head_film_gen_2(task_emb)
            
            # FiLM-modulated feature
            g_film = gamma2 * (F.relu(model.head.fc1(gamma1 * g + beta1))) + beta2
            features.append(g_film.cpu().numpy())

            # 反归一化 RT
            y_orig = []
            for y_norm, t_id in zip(data.y.view(-1).cpu().numpy(), graph_tasks.cpu().numpy()):
                mean, std = task_stats[int(t_id)]
                y_orig.append(y_norm * std + mean)
            rts.append(np.array(y_orig))

            task_ids.append(graph_tasks.cpu().numpy())

    features = np.concatenate(features, axis=0)
    rts = np.concatenate(rts, axis=0)
    task_ids = np.concatenate(task_ids, axis=0)
    return features, rts, task_ids


def extract_adapter_crossstitch_features(model, loader, device, task_stats):
    model.eval()
    features, rts, task_ids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_tasks = data.task.view(-1).to(batch.device)

            # 前向传播至 pooling 前
            h = model.embed(x)
            for conv, bn in zip(model.convs, model.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
            g = global_add_pool(h, batch)

            # Adapter + Cross-Stitch
            g_adjusted = torch.zeros_like(g)
            for t_id in range(model.num_tasks):
                mask = (graph_tasks == t_id)
                if mask.sum() > 0:
                    g_adjusted[mask] = model.adapters[t_id](g[mask])
            g = g_adjusted
            if model.use_cross_stitch:
                g = model.cross_stitch(g, graph_tasks)
            features.append(g.cpu().numpy())

            y_orig = []
            for y_norm, t_id in zip(data.y.view(-1).cpu().numpy(), graph_tasks.cpu().numpy()):
                mean, std = task_stats[int(t_id)]
                y_orig.append(y_norm * std + mean)
            rts.append(np.array(y_orig))

            task_ids.append(graph_tasks.cpu().numpy())

    features = np.concatenate(features, axis=0)
    rts = np.concatenate(rts, axis=0)
    task_ids = np.concatenate(task_ids, axis=0)
    return features, rts, task_ids


def extract_pre_adapter_features(model, loader, device, task_stats):
    model.eval()
    features, rts, task_ids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            graph_tasks = data.task.view(-1).to(batch.device)

            # 前向传播至 pooling 前
            h = model.embed(x)
            for conv, bn in zip(model.convs, model.bns):
                h = conv(h, edge_index)
                h = bn(h)
                h = F.relu(h)
            g = global_add_pool(h, batch)

            features.append(g.cpu().numpy())
            y_orig = []
            for y_norm, t_id in zip(data.y.view(-1).cpu().numpy(), graph_tasks.cpu().numpy()):
                mean, std = task_stats[int(t_id)]
                y_orig.append(y_norm * std + mean)
            rts.append(np.array(y_orig))
            task_ids.append(graph_tasks.cpu().numpy())

    features = np.concatenate(features, axis=0)
    rts = np.concatenate(rts, axis=0)
    task_ids = np.concatenate(task_ids, axis=0)
    return features, rts, task_ids


def extract_film_params(model, device="cpu"):
    model.eval()
    model = model.to(device)

    gamma_beta_dict = {}

    with torch.no_grad():
        for t_id in range(model.num_tasks):
            # 获取任务嵌入
            task_id_tensor = torch.tensor([t_id], dtype=torch.long, device=device)
            task_emb = model.task_embed(task_id_tensor)

            # 生成 gamma / beta
            gamma1, beta1 = model.head_film_gen_1(task_emb)
            gamma2, beta2 = model.head_film_gen_2(task_emb)

            # 保存为 numpy 数组
            gamma_beta_dict[t_id] = {
                "gamma1": gamma1.cpu().numpy().flatten(),
                "beta1":  beta1.cpu().numpy().flatten(),
                "gamma2": gamma2.cpu().numpy().flatten(),
                "beta2":  beta2.cpu().numpy().flatten(),
            }

    return gamma_beta_dict


def extract_cross_stitch_alpha(model):
    alpha_matrices = {}

    if isinstance(model.cross_stitch, (nn.ModuleList, list)):
        for i, cs in enumerate(model.cross_stitch):
            if hasattr(cs, "alpha"):
                alpha_matrices[f"cross_stitch_{i}"] = cs.alpha.detach().cpu().numpy()
    else:
        # 单层 Cross-Stitch
        if hasattr(model.cross_stitch, "alpha"):
            alpha_matrices["cross_stitch"] = model.cross_stitch.alpha.detach().cpu().numpy()

    return alpha_matrices


def extract_adapter_weights(model):
    adapter_weights = {}
    for i, adapter in enumerate(model.adapters):
        params = []
        for name, p in adapter.named_parameters():
            if p.requires_grad:
                params.append(p.detach().cpu().flatten().numpy())
        adapter_weights[i] = np.concatenate(params)
    return adapter_weights


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    batch_size = 32
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_data = []
    task_stats = {}
    for task_id, csv in enumerate(data_files):
        print(f"Loading dataset {task_id}: {csv}")
        ds, stats = load_dataset(csv, task_id)
        if len(ds) == 0:
            print(f"[WARN] dataset {csv} produced 0 graphs.")
        all_data.extend(ds)
        task_stats.update(stats)

    assert len(all_data) > 0, "Empty dataset after parsing."

    node_dim = all_data[0].x.size(-1)

    y_vals = np.array([d.y.item() for d in all_data])
    quantiles = pd.qcut(pd.Series(y_vals), q=min(10, len(y_vals)), labels=False, duplicates='drop')
    idx = np.arange(len(all_data))
    idx_train, idx_temp, _, y_temp = train_test_split(idx, quantiles, test_size=0.2, stratify=quantiles)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, stratify=y_temp)

    train_set = [all_data[i] for i in idx_train]
    val_set = [all_data[i] for i in idx_val]
    test_set = [all_data[i] for i in idx_test]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


    model = GINRegressor(
        node_dim=node_dim,
        num_tasks=len(data_files),
        task_emb_dim=len(data_files),
        hidden=64,
        layers=4,
        dropout=0.02
    ).to(device)


    # model = GCNRegressor(
    #     node_dim=node_dim,
    #     num_tasks=len(data_files),
    #     task_emb_dim=len(data_files),
    #     hidden=64,
    #     layers=4,
    #     dropout=0.02
    # ).to(device)


    # model = GATRegressor(
    #     node_dim=node_dim,
    #     num_tasks=len(data_files),
    #     task_emb_dim=len(data_files),
    #     hidden=64,
    #     layers=4,
    #     dropout=0.02
    # ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val = float('inf')
    best_state = None

    for epoch in range(1, 201):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
        metrics = evaluate(model, val_loader, device, task_stats)
        scheduler.step(metrics["RMSE"])
        if metrics["RMSE"] < best_val:
            best_val = metrics["RMSE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | "
                  f"Val: MAE={metrics['MAE']:.4f} RMSE={metrics['RMSE']:.4f} R2={metrics['R2']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Evaluating on test set (best val checkpoint)...")
    overall_test, per_task_test, y_test, p_test = task_evaluate(model, test_loader, device, task_stats, len(data_files))
    print("Test Metrics:", per_task_test)

    # Scatter plot
    plot_scatter(y_test, p_test)


    # Feature plot
    features, rts, task_ids = extract_film_features(model, test_loader, device, task_stats)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(features)
    palette = sns.color_palette("plasma", as_cmap=True)
    
    plt.figure(figsize=(6, 4), dpi = 300)
    unique_tasks = np.arange(4)
    markers = ['o', 's', 'D', '^']

    for i, t in enumerate(np.unique(unique_tasks)):
        mask = task_ids == t
        sc = plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=rts[mask],
            cmap='plasma',
            s=40,
            alpha=0.8,
            marker=markers[i % len(markers)],
            edgecolor='black',
            linewidth=0.3,
            label=f'Task {t}'
        )
    
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    cbar = plt.colorbar(sc)
    cbar.set_label('Retention Time (s)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


    # FiLM correlation plot
    film_params = extract_film_params(model, device)
    task_vectors = []
    for t_id, params in film_params.items():
        vec = np.concatenate([params["gamma1"], params["gamma2"]])
        # vec = np.concatenate([params["beta1"], params["beta2"]])
        task_vectors.append(vec)
    task_vectors = np.stack(task_vectors)  # shape: (num_tasks, total_dim)

    corr_matrix = np.corrcoef(task_vectors)
    corr_matrix = np.round(corr_matrix, 2)

    plt.figure(figsize=(7, 7), dpi = 300)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='Blues',
        square=True,
        cbar=False,
        xticklabels=data_name,
        yticklabels=data_name,
    )
    plt.tight_layout()
    plt.show()
    
    