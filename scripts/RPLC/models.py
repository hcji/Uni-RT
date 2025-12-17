"""
Created on Fri Dec 12 12:55:42 2025

@author: hongchao ji 
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

# ------------------------------
# FiLM, Adapter, CrossStitch, KAN
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
    def __init__(self, num_tasks, main_task_id=0, alpha_init=1.0, beta_init=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.main_task_id = main_task_id

        alpha = torch.eye(num_tasks) * alpha_init
        alpha[main_task_id, :] = beta_init
        alpha[main_task_id, main_task_id] = alpha_init
        self.alpha = nn.Parameter(alpha)

    def forward(self, h_task, graph_tasks):
        h_new = torch.zeros_like(h_task)
        for i in range(self.num_tasks):
            mask_i = (graph_tasks == i)
            if mask_i.sum() == 0:
                continue

            h_i = h_task[mask_i]
            fused = 0.0

            if i == self.main_task_id:
                for j in range(self.num_tasks):
                    mask_j = (graph_tasks == j)
                    if mask_j.sum() == 0:
                        continue
                    h_j = h_task[mask_j].mean(dim=0, keepdim=True)
                    fused += self.alpha[i, j] * h_j
                h_new[mask_i] = h_i + fused
            else:
                h_new[mask_i] = h_i
        return h_new


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )
        B = y.transpose(0, 1).to(A.device)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class MultiHeadKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            heads=1,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(MultiHeadKAN, self).__init__()
        self.heads = heads
        self.kans = torch.nn.ModuleList([
            KAN(layers_hidden, grid_size, spline_order, scale_noise, scale_base, scale_spline, base_activation,
                grid_eps, grid_range) for _ in range(heads)])

    def forward(self, x: torch.Tensor, update_grid=False):
        output = torch.stack([kan(x[:, i, :], update_grid) for i, kan in enumerate(self.kans)], dim=1)
        return output

    def eforward(self, x: torch.Tensor, update_grid=False):
        outputs = [kan(x, update_grid) for kan in self.kans]
        return torch.stack(outputs, dim=1)

    def reset_parameters(self):
        for kan in self.kans:
            kan.reset_parameters()


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

class GINKANConv(GINConv):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 grid_size=5, spline_order=3, eps=0.0, train_eps=False):
        nn_kan = KAN(layers_hidden=[input_dim, hidden_dim, output_dim],
                     grid_size=grid_size,
                     spline_order=spline_order)

        super(GINKANConv, self).__init__(nn=nn_kan, eps=eps, train_eps=train_eps)

    def forward(self, x, edge_index):
        return super().forward(x, edge_index)

class GINKANmultiRegressor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        num_tasks: int = 28,
        task_emb_dim: int = 32,
        hidden: int = 64,
        layers: int = 4,
        grid_size: int = 5,
        spline_order: int = 3,
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
            conv = GINKANConv(
                input_dim=hidden,
                hidden_dim=hidden,
                output_dim=hidden,
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))

        self.task_embed = nn.Embedding(num_tasks, task_emb_dim)

        self.head_film_gen_1 = FiLMParamGenerator(task_emb_dim, hidden)
        self.head_film_gen_2 = FiLMParamGenerator(task_emb_dim, hidden)

        self.adapters = nn.ModuleList([
            Adapter(hidden, reduction=adapter_reduction) for _ in range(num_tasks)
        ])

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
# ------------------------------
# GINRegressor without Mordred
# ------------------------------
class GINmultiRegressor(nn.Module):
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


class GCNmultiRegressor(nn.Module):
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


class GATmultiRegressor(nn.Module):
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
# GIN-KAN Unified Model
# ------------------------------
class GINKANUnified(nn.Module):
    def __init__(
        self, 
        node_dim, 
        hidden=64, 
        layers=4, 
        grid_size=5,       
        spline_order=3,    
        dropout=0.1        
    ):
        super().__init__()

        self.embed = nn.Linear(node_dim, hidden)


        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(layers):

            conv = GINKANConv(
                input_dim=hidden,
                hidden_dim=hidden,
                output_dim=hidden,
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))

        self.head = KAN(
            layers_hidden=[hidden, hidden, 1],
            grid_size=grid_size,
            spline_order=spline_order
        )

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Atom Embedding
        h = self.embed(x)
        
        # GIN-KAN Message Passing
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            
        # Global Pooling
        g = global_add_pool(h, batch)
        
        # Dropout
        g = self.dropout_layer(g)
        
        # KAN Head Prediction
        out = self.head(g)
        
        return out.view(-1)
    
    
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

        self.convs.append(GATConv(hidden, hidden // heads, heads=heads))
        self.bns.append(BatchNorm(hidden))

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
            h = F.elu(h) 
        g = global_add_pool(h, batch)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g).view(-1)

# ------------------------------
# Single-task GIN-KAN Model
# ------------------------------
class SingleTaskGINKAN(nn.Module):
    def __init__(
        self, 
        node_dim: int, 
        hidden: int = 64, 
        layers: int = 4, 
        grid_size: int = 5, 
        spline_order: int = 3
    ):
        super().__init__()

        self.embed = nn.Linear(node_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(layers):

            conv = GINKANConv(
                input_dim=hidden,
                hidden_dim=hidden,
                output_dim=hidden,
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))

        self.head = KAN(
            layers_hidden=[hidden, hidden, 1],
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Atom Embedding
        h = self.embed(x)
        
        # Message Passing
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            
        # Global Pooling
        g = global_add_pool(h, batch)
        
        # Prediction via KAN Head
        out = self.head(g)
        
        return out.view(-1)
    
    
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

        self.convs.append(GATConv(hidden, hidden // heads, heads=heads))
        self.bns.append(BatchNorm(hidden))

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
            h = F.elu(h) 
        g = global_add_pool(h, batch)
        g = F.relu(self.fc1(g))
        g = self.dropout(g)
        return self.fc2(g).view(-1)

