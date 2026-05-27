"""
Phase A — CIFAR-10 EAHM-FL revision experiments.

Module entry point: main(results_dir, auto_pusher=None, smoke=False)

Addresses reviewer concerns from TAI-2026-Mar-A-00599:
  R1.1-1.3 (val-set assumption)   : contribution = local loss reduction per client
  R1.10, R2.1 (credit assignment) : per-client contribution, not round-level
  R1.8, R2.4, R3.3c (baselines)   : SCAFFOLD + FedDyn added
  R1.9 (ablation breadth)         : ablation runs across all four alphas
  R1.11 (seeds)                   : 5 seeds headline, 3 secondary
  R1.6 (temporal dynamics)        : selection + contribution histories logged
  R1.19 (fairness)                : per-class accuracy logged every round
  R3.3d (comm/compute cost)       : per-round bytes + wall-clock + peak GPU logged

RUN ORDER: FedAvg -> EAHM-FL -> FedAvg -> EAHM-FL ... at alpha=0.1 first.
After the first ~4 runs (~30 min) we have early validation of both
infrastructure (pushes working) and method (EAHM-FL converging).
"""

import os
import sys
import json
import time
import random
import math
import gc
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# CONFIG
# =============================================================================

NUM_CLIENTS = 10
CLIENTS_PER_ROUND = 3
NUM_ROUNDS = 50
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.01
LR_DECAY = 0.998
LR_MIN = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
NUM_CLASSES = 10

HEADLINE_ALPHAS = [0.1, 0.3]
HEADLINE_SEEDS = [42, 43, 44, 45, 46]
SECONDARY_ALPHAS = [0.5, 1.0]
SECONDARY_SEEDS = [42, 43, 44]

ALGORITHMS = [
    "FedAvg",
    "FedProx",
    "FedNova",
    "SCAFFOLD",
    "FedDyn",
    "EAHM-FL",
    "EAHM-FL-no_selection",
    "EAHM-FL-no_weighting",
]

FEDPROX_MU = 0.01
FEDDYN_ALPHA = 0.01
SCAFFOLD_CV_CLIP = 5.0

EAHM_WARMUP = 5
EAHM_UCB_C = 0.5
EAHM_EMA = 0.2
EAHM_MIN_W = 0.3
EAHM_MAX_W = 2.0
EAHM_MIN_PARTICIPATION = 3
EAHM_PERCENTILE = 95.0

# =============================================================================
# MODEL — small CNN matching the original 62.99% / 54.65% results
# =============================================================================

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model():
    return SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)

def count_params(m): return sum(p.numel() for p in m.parameters())
def model_bytes(m): return count_params(m) * 4

# =============================================================================
# DATA
# =============================================================================

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dirichlet_partition(labels, num_clients, alpha, num_classes, rng):
    ci = {i: [] for i in range(num_clients)}
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        rng.shuffle(idx_k)
        prop = rng.dirichlet(np.repeat(alpha, num_clients))
        prop = np.maximum(prop, 1e-5); prop = prop / prop.sum()
        prop = (prop * len(idx_k)).astype(int)
        prop[np.argmax(prop)] += len(idx_k) - prop.sum()
        start = 0
        for cid in range(num_clients):
            end = start + prop[cid]
            ci[cid].extend(idx_k[start:end].tolist())
            start = end
    return ci

def load_data(seed, alpha, data_root):
    rng = np.random.RandomState(seed)
    tr_tx = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    te_tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=tr_tx)
    test_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=te_tx)
    eval_train_ds = torchvision.datasets.CIFAR10(root=str(data_root), train=True, download=False, transform=te_tx)

    labels = np.array(train_ds.targets)
    ci = dirichlet_partition(labels, NUM_CLIENTS, alpha, NUM_CLASSES, rng)

    cl, cel = {}, {}
    for cid, idx in ci.items():
        cl[cid] = DataLoader(Subset(train_ds, idx), batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=2, pin_memory=True)
        cel[cid] = DataLoader(Subset(eval_train_ds, idx), batch_size=128,
                              shuffle=False, num_workers=2, pin_memory=True)
    tl = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    sizes = {cid: len(idx) for cid, idx in ci.items()}
    return cl, cel, tl, sizes

# =============================================================================
# EVAL + TRAIN
# =============================================================================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction='sum')
    correct, total, total_loss = 0, 0, 0.0
    pcc = np.zeros(NUM_CLASSES); pct = np.zeros(NUM_CLASSES)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x); total_loss += crit(out, y).item()
        pred = out.argmax(1); correct += pred.eq(y).sum().item(); total += y.size(0)
        for c in range(NUM_CLASSES):
            m = y == c
            pct[c] += m.sum().item()
            pcc[c] += (pred[m] == c).sum().item()
    pca = np.divide(pcc, pct, out=np.zeros_like(pcc), where=pct > 0)
    return correct / total, total_loss / total, pca.tolist()

@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction='sum')
    total_loss, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        total_loss += crit(model(x), y).item()
        total += y.size(0)
    return total_loss / max(total, 1)

def get_lr(rnd):
    return max(LR * (LR_DECAY ** rnd), LR_MIN)

def train_fedavg(model, loader, rnd):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=get_lr(rnd),
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
    return model.state_dict()

def train_fedprox(model, loader, rnd, global_state):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=get_lr(rnd),
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    gp = {k: v.to(DEVICE) for k, v in global_state.items()}
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y)
            prox = 0.0
            for name, p in model.named_parameters():
                if name in gp: prox = prox + ((p - gp[name]) ** 2).sum()
            loss = loss + (FEDPROX_MU / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
    return model.state_dict()

def train_scaffold(model, loader, rnd, global_state, c_local, c_global):
    model.train()
    lr = get_lr(rnd)
    opt = torch.optim.SGD(model.parameters(), lr=lr,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    initial = {k: v.clone().detach() for k, v in model.state_dict().items()}
    cl_ = {k: v.to(DEVICE) for k, v in c_local.items()}
    cg = {k: v.to(DEVICE) for k, v in c_global.items()}
    step = 0
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None and name in cg:
                    p.grad.data.add_(cg[name] - cl_[name])
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step(); step += 1
    final = model.state_dict()
    new_cl = {}
    for k in c_local:
        if k in initial and k in final:
            drift = (initial[k].to(DEVICE) - final[k].to(DEVICE)).float()
            option = torch.clamp(drift / max(step * lr, 1e-8), -1.0, 1.0)
            new_cl[k] = torch.clamp(cl_[k] - cg[k] + option,
                                    -SCAFFOLD_CV_CLIP, SCAFFOLD_CV_CLIP).cpu()
        else:
            new_cl[k] = c_local[k]
    return final, new_cl

def train_feddyn(model, loader, rnd, global_state, h_local):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=get_lr(rnd),
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    gp = {k: v.to(DEVICE) for k, v in global_state.items()}
    hd = {k: v.to(DEVICE) for k, v in h_local.items()}
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = crit(model(x), y)
            lin, quad = 0.0, 0.0
            for name, p in model.named_parameters():
                if name in gp and name in hd:
                    lin = lin + (hd[name] * p).sum()
                    quad = quad + ((p - gp[name]) ** 2).sum()
            loss = loss + lin + (FEDDYN_ALPHA / 2.0) * quad
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            opt.step()
    final = model.state_dict()
    new_h = {}
    for k, v in h_local.items():
        if k in final and k in global_state:
            delta = (final[k].to(DEVICE) - gp[k]).float()
            new_h[k] = (hd[k] - FEDDYN_ALPHA * delta).cpu()
        else:
            new_h[k] = v
    return final, new_h

def aggregate_weighted(updates, weights):
    total = sum(weights.values())
    norm = {cid: w / total for cid, w in weights.items()}
    first_cid = next(iter(updates))
    keys = list(updates[first_cid].keys())
    agg = {}
    for k in keys:
        first = updates[first_cid][k]
        if first.dtype in (torch.int64, torch.int32, torch.long):
            agg[k] = first.clone()
        else:
            agg[k] = sum(norm[cid] * updates[cid][k].float().to(DEVICE) for cid in updates)
    return agg

def aggregate_fednova(updates, weights, tau_eff):
    norm_w = {}; total = 0.0
    for cid in updates:
        w = weights[cid] / max(tau_eff.get(cid, 1), 1)
        norm_w[cid] = w; total += w
    norm_w = {cid: w / total for cid, w in norm_w.items()}
    first_cid = next(iter(updates))
    keys = list(updates[first_cid].keys())
    agg = {}
    for k in keys:
        first = updates[first_cid][k]
        if first.dtype in (torch.int64, torch.int32, torch.long):
            agg[k] = first.clone()
        else:
            agg[k] = sum(norm_w[cid] * updates[cid][k].float().to(DEVICE) for cid in updates)
    return agg

# =============================================================================
# EAHM TRACKER
# =============================================================================

class EAHMTracker:
    def __init__(self, num_clients):
        self.mu = np.zeros(num_clients)
        self.var = np.ones(num_clients) * 0.01
        self.n = np.zeros(num_clients, dtype=int)
        self.history = defaultdict(list)
        self.selection_history = []

    def update(self, cid, contribution):
        self.n[cid] += 1
        delta = contribution - self.mu[cid]
        self.mu[cid] = (1 - EAHM_EMA) * self.mu[cid] + EAHM_EMA * contribution
        self.var[cid] = (1 - EAHM_EMA) * self.var[cid] + EAHM_EMA * delta * delta
        self.history[int(cid)].append(float(contribution))

    def ucb_scores(self, t):
        scores = np.zeros(len(self.mu))
        for i in range(len(self.mu)):
            if self.n[i] == 0:
                scores[i] = float('inf')
            else:
                sigma = math.sqrt(max(self.var[i], 0.0))
                scores[i] = self.mu[i] + EAHM_UCB_C * sigma * \
                    math.sqrt(2 * math.log(max(t, 2)) / self.n[i])
        return scores

    def select(self, t, k, do_ucb=True):
        if (not do_ucb) or t < EAHM_WARMUP:
            sel = [int(c) for c in np.random.choice(len(self.mu), k, replace=False)]
        else:
            scores = self.ucb_scores(t)
            sel = [int(c) for c in np.argsort(scores)[-k:]]
        self.selection_history.append(list(sel))
        return sel

    def aggregation_weights(self, selected, data_sizes, do_weighting=True):
        weights = {}
        if not do_weighting:
            for cid in selected: weights[cid] = float(data_sizes[cid])
            return weights
        eligible = [self.mu[c] for c in selected if self.n[c] >= EAHM_MIN_PARTICIPATION]
        tau = float(np.percentile(eligible, EAHM_PERCENTILE)) if len(eligible) >= 2 else 1.0
        for cid in selected:
            base = float(data_sizes[cid])
            if self.n[cid] < EAHM_MIN_PARTICIPATION or abs(tau) <= 1e-8:
                weights[cid] = base
            else:
                scale = max(EAHM_MIN_W, min(EAHM_MAX_W, self.mu[cid] / tau))
                weights[cid] = base * scale
        return weights

    def to_dict(self):
        return {
            'mu': self.mu.tolist(),
            'var': self.var.tolist(),
            'n': [int(x) for x in self.n],
            'history': {str(k): v for k, v in self.history.items()},
            'selection_history': self.selection_history,
        }

# =============================================================================
# RUN ONE
# =============================================================================

def needs_eahm(a):     return a.startswith("EAHM-FL")
def needs_scaffold(a): return a == "SCAFFOLD"
def needs_feddyn(a):   return a == "FedDyn"

def run_one(algorithm, alpha, seed, paths, data_root):
    """Run one (algo, alpha, seed) experiment. Returns 'SKIP' or 'DONE'."""
    runs_dir, client_dir, ckpt_dir = paths['runs'], paths['client'], paths['ckpt']
    run_id = f"{algorithm}_alpha{alpha}_seed{seed}"
    csv_path    = runs_dir / f"{run_id}.csv"
    done_path   = runs_dir / f"{run_id}.done"
    client_path = client_dir / f"{run_id}.json"
    model_ckpt  = ckpt_dir / f"{run_id}.pt"
    state_ckpt  = ckpt_dir / f"{run_id}_state.pt"

    if done_path.exists():
        return "SKIP"

    print(f"\n>>> {run_id}")
    t_start = time.time()
    set_all_seeds(seed)
    cl, cel, tl, sizes = load_data(seed, alpha, data_root)

    model = build_model()
    bytes_per_xfer = model_bytes(model)

    tracker = EAHMTracker(NUM_CLIENTS) if needs_eahm(algorithm) else None
    c_global, c_locals, h_locals = None, None, None

    if needs_scaffold(algorithm):
        c_global = {k: torch.zeros_like(v, dtype=torch.float32).cpu()
                    for k, v in model.state_dict().items()}
        c_locals = {cid: {k: torch.zeros_like(v, dtype=torch.float32).cpu()
                          for k, v in model.state_dict().items()}
                    for cid in range(NUM_CLIENTS)}
    if needs_feddyn(algorithm):
        h_locals = {cid: {n: torch.zeros_like(p, dtype=torch.float32).cpu()
                          for n, p in model.named_parameters()}
                    for cid in range(NUM_CLIENTS)}

    # ---- Resume ----
    start_round = 1
    cum_up, cum_down = 0.0, 0.0

    can_resume = csv_path.exists() and model_ckpt.exists()
    if needs_eahm(algorithm):
        can_resume = can_resume and client_path.exists()
    if needs_scaffold(algorithm) or needs_feddyn(algorithm):
        can_resume = can_resume and state_ckpt.exists()

    if can_resume:
        try:
            existing = pd.read_csv(csv_path)
            if len(existing) > 0:
                start_round = int(existing['round'].max()) + 1
                cum_up = float(existing['cum_up_bytes'].iloc[-1])
                cum_down = float(existing['cum_down_bytes'].iloc[-1])
                model.load_state_dict(torch.load(model_ckpt, map_location=DEVICE))
                if needs_eahm(algorithm):
                    with open(client_path) as f:
                        d = json.load(f)
                    tracker.mu = np.array(d['mu'])
                    tracker.var = np.array(d['var'])
                    tracker.n = np.array(d['n'], dtype=int)
                    tracker.history = defaultdict(list, {int(k): v for k, v in d['history'].items()})
                    tracker.selection_history = d['selection_history']
                if needs_scaffold(algorithm):
                    sb = torch.load(state_ckpt, map_location='cpu')
                    c_global = sb['c_global']; c_locals = sb['c_locals']
                if needs_feddyn(algorithm):
                    sb = torch.load(state_ckpt, map_location='cpu')
                    h_locals = sb['h_locals']
                print(f"    RESUME from round {start_round}")
        except Exception as e:
            print(f"    RESUME failed ({e}); restarting from round 1")
            start_round = 1; cum_up, cum_down = 0.0, 0.0
            if csv_path.exists(): csv_path.unlink()
            if client_path.exists(): client_path.unlink()
    else:
        if csv_path.exists():
            print(f"    Stale CSV without checkpoint; restarting run")
            csv_path.unlink()
            if client_path.exists(): client_path.unlink()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for rnd in range(start_round, NUM_ROUNDS + 1):
        t_round = time.time()
        t_sel = time.time()
        if needs_eahm(algorithm):
            do_ucb = (algorithm != "EAHM-FL-no_selection")
            selected = tracker.select(rnd, CLIENTS_PER_ROUND, do_ucb=do_ucb)
        else:
            selected = [int(c) for c in np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)]
        sel_time = time.time() - t_sel

        global_state = {k: v.clone() for k, v in model.state_dict().items()}
        cum_down += bytes_per_xfer * len(selected)
        if algorithm == "SCAFFOLD":
            cum_down += bytes_per_xfer * len(selected)

        updates = {}; train_times = {}; contributions = {}
        new_c_locals = {} if needs_scaffold(algorithm) else None
        new_h_locals = {} if needs_feddyn(algorithm) else None
        tau_eff = {}

        for cid in selected:
            local = build_model(); local.load_state_dict(global_state)
            loss_before = evaluate_loss(local, cel[cid])

            t_cli = time.time()
            if algorithm == "FedAvg":
                new_state = train_fedavg(local, cl[cid], rnd)
            elif algorithm == "FedProx":
                new_state = train_fedprox(local, cl[cid], rnd, global_state)
            elif algorithm == "FedNova":
                new_state = train_fedavg(local, cl[cid], rnd)
                tau_eff[cid] = LOCAL_EPOCHS * len(cl[cid])
            elif algorithm == "SCAFFOLD":
                new_state, new_cl_ = train_scaffold(
                    local, cl[cid], rnd, global_state, c_locals[cid], c_global)
                new_c_locals[cid] = new_cl_
            elif algorithm == "FedDyn":
                new_state, new_hl_ = train_feddyn(
                    local, cl[cid], rnd, global_state, h_locals[cid])
                new_h_locals[cid] = new_hl_
            else:  # EAHM variants
                new_state = train_fedavg(local, cl[cid], rnd)
            train_times[cid] = time.time() - t_cli

            local.load_state_dict(new_state)
            loss_after = evaluate_loss(local, cel[cid])

            updates[cid] = {k: v.detach().cpu() for k, v in new_state.items()}
            contributions[cid] = float(loss_before - loss_after)
            del local

        cum_up += bytes_per_xfer * len(selected)
        if algorithm == "SCAFFOLD":
            cum_up += bytes_per_xfer * len(selected)

        t_agg = time.time()
        if needs_eahm(algorithm):
            for cid in selected: tracker.update(cid, contributions[cid])
            do_weighting = (algorithm != "EAHM-FL-no_weighting")
            weights = tracker.aggregation_weights(selected, sizes, do_weighting=do_weighting)
            agg_state = aggregate_weighted(updates, weights)
        elif algorithm == "FedNova":
            weights = {cid: float(sizes[cid]) for cid in selected}
            agg_state = aggregate_fednova(updates, weights, tau_eff)
        elif algorithm == "SCAFFOLD":
            weights = {cid: float(sizes[cid]) for cid in selected}
            agg_state = aggregate_weighted(updates, weights)
            for k in c_global:
                delta_sum = sum((new_c_locals[cid][k] - c_locals[cid][k]) for cid in selected)
                c_global[k] = c_global[k] + delta_sum / NUM_CLIENTS
                c_global[k] = torch.clamp(c_global[k], -SCAFFOLD_CV_CLIP, SCAFFOLD_CV_CLIP)
            for cid in selected: c_locals[cid] = new_c_locals[cid]
        elif algorithm == "FedDyn":
            weights = {cid: float(sizes[cid]) for cid in selected}
            agg_state = aggregate_weighted(updates, weights)
            for cid in selected: h_locals[cid] = new_h_locals[cid]
        else:
            weights = {cid: float(sizes[cid]) for cid in selected}
            agg_state = aggregate_weighted(updates, weights)
        model.load_state_dict(agg_state)
        agg_time = time.time() - t_agg

        test_acc, test_loss, per_class_acc = evaluate(model, tl)
        round_time = time.time() - t_round
        peak_mem = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0

        row = {
            'round': rnd, 'algorithm': algorithm, 'alpha': alpha, 'seed': seed,
            'test_acc': test_acc, 'test_loss': test_loss,
            'selected': ','.join(str(c) for c in selected),
            'mean_contribution': float(np.mean(list(contributions.values()))),
            'round_time_s': round_time, 'sel_time_s': sel_time, 'agg_time_s': agg_time,
            'mean_client_train_s': float(np.mean(list(train_times.values()))),
            'cum_up_bytes': cum_up, 'cum_down_bytes': cum_down,
            'peak_gpu_gb': peak_mem,
        }
        for c in range(NUM_CLASSES):
            row[f'class{c}_acc'] = per_class_acc[c]

        pd.DataFrame([row]).to_csv(csv_path, mode='a',
                                   header=not csv_path.exists(), index=False)
        torch.save(model.state_dict(), model_ckpt)
        if needs_eahm(algorithm):
            with open(client_path, 'w') as f:
                json.dump(tracker.to_dict(), f, default=str)
        if needs_scaffold(algorithm):
            torch.save({'c_global': c_global, 'c_locals': c_locals}, state_ckpt)
        if needs_feddyn(algorithm):
            torch.save({'h_locals': h_locals}, state_ckpt)

        if rnd % 10 == 0 or rnd == 1 or rnd == NUM_ROUNDS:
            print(f"    R{rnd:3d} acc={test_acc:.4f} loss={test_loss:.4f} "
                  f"t={round_time:.1f}s up={cum_up/1e9:.2f}GB peak={peak_mem:.2f}GB")

    elapsed = time.time() - t_start
    done_path.touch()
    print(f"    DONE in {elapsed/60:.1f} min")

    if model_ckpt.exists(): model_ckpt.unlink()
    if state_ckpt.exists(): state_ckpt.unlink()

    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return "DONE"

# =============================================================================
# RUN ORDER — EAHM-FL appears EARLY so we get fast validation
# =============================================================================

def build_run_list():
    """
    Order:
      Phase 1 (fast validation, ~30 min):
        FedAvg@0.1@42, EAHM-FL@0.1@42, FedAvg@0.1@43, EAHM-FL@0.1@43
        -> after these 4 we know: (a) infrastructure pushes, (b) EAHM-FL works
      Phase 2 (rest of headline alpha=0.1):
        FedProx, FedNova, SCAFFOLD, FedDyn, EAHM-FL ablations @ 0.1
      Phase 3 (alpha=0.3, all algos, headline seeds)
      Phase 4 (alpha=0.5, alpha=1.0, secondary seeds)
    """
    runs = []
    # --- Phase 1: fast validation ---
    runs.append(("FedAvg", 0.1, 42))
    runs.append(("EAHM-FL", 0.1, 42))
    runs.append(("FedAvg", 0.1, 43))
    runs.append(("EAHM-FL", 0.1, 43))

    seen = set(runs)

    # --- Phase 2: remaining alpha=0.1 headline runs ---
    for algo in ALGORITHMS:
        for s in HEADLINE_SEEDS:
            k = (algo, 0.1, s)
            if k not in seen:
                runs.append(k); seen.add(k)

    # --- Phase 3: alpha=0.3 ---
    for algo in ALGORITHMS:
        for s in HEADLINE_SEEDS:
            k = (algo, 0.3, s)
            if k not in seen:
                runs.append(k); seen.add(k)

    # --- Phase 4: secondary alphas ---
    for a in SECONDARY_ALPHAS:
        for algo in ALGORITHMS:
            for s in SECONDARY_SEEDS:
                k = (algo, a, s)
                if k not in seen:
                    runs.append(k); seen.add(k)
    return runs

# =============================================================================
# AGGREGATION + PLOTS + REPORT
# =============================================================================

def load_all_runs(runs_dir):
    rows = []
    for f in runs_dir.glob("*.csv"):
        try: rows.append(pd.read_csv(f))
        except Exception as e: print(f"  skip {f.name}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def aggregate_and_plot(paths):
    runs_dir = paths['runs']
    client_dir = paths['client']
    plots_dir = paths['plots']
    results_root = paths['results_root']

    df = load_all_runs(runs_dir)
    if df.empty:
        print("No runs found to aggregate.")
        return

    df.to_csv(results_root / "all_rounds.csv", index=False)

    finals = df.groupby(['algorithm', 'alpha', 'seed']).apply(
        lambda g: g.loc[g['round'].idxmax()]).reset_index(drop=True)
    finals.to_csv(results_root / "all_finals.csv", index=False)

    agg = finals.groupby(['algorithm', 'alpha']).agg(
        final_acc_mean=('test_acc', 'mean'),
        final_acc_std=('test_acc', 'std'),
        final_acc_n=('test_acc', 'count'),
        cum_up_gb=('cum_up_bytes', lambda x: x.mean() / 1e9),
        cum_down_gb=('cum_down_bytes', lambda x: x.mean() / 1e9),
        round_time_s=('round_time_s', 'mean'),
        peak_gpu_gb=('peak_gpu_gb', 'mean'),
    ).reset_index()
    agg.to_csv(results_root / "summary.csv", index=False)

    cost = finals.groupby('algorithm').agg(
        avg_cum_up_gb=('cum_up_bytes', lambda x: x.mean() / 1e9),
        avg_cum_down_gb=('cum_down_bytes', lambda x: x.mean() / 1e9),
        avg_round_time_s=('round_time_s', 'mean'),
        avg_peak_gpu_gb=('peak_gpu_gb', 'mean'),
    ).reset_index()
    cost.to_csv(results_root / "cost_table.csv", index=False)

    colors = {
        'FedAvg': '#1f77b4', 'FedProx': '#2ca02c', 'FedNova': '#9467bd',
        'SCAFFOLD': '#ff7f0e', 'FedDyn': '#8c564b', 'EAHM-FL': '#d62728',
        'EAHM-FL-no_selection': '#ff9896', 'EAHM-FL-no_weighting': '#e377c2',
    }
    alphas_sorted = sorted(agg['alpha'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGORITHMS:
        sub = agg[agg['algorithm'] == algo].sort_values('alpha')
        if sub.empty: continue
        ax.errorbar(sub['alpha'], sub['final_acc_mean'] * 100,
                    yerr=sub['final_acc_std'].fillna(0) * 100,
                    label=algo, marker='o', linewidth=2,
                    color=colors.get(algo, 'gray'), capsize=4)
    ax.set_xlabel('Dirichlet α'); ax.set_ylabel('Final acc (%)'); ax.set_xscale('log')
    ax.set_title('Final accuracy vs data heterogeneity'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(plots_dir / "01_accuracy_vs_alpha.png", dpi=140); plt.close()

    n_plots = len(alphas_sorted)
    nrow = (n_plots + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 5 * nrow))
    axes = axes.flat if n_plots > 1 else [axes]
    for ax, a in zip(axes, alphas_sorted):
        for algo in ALGORITHMS:
            sub = df[(df['algorithm'] == algo) & (df['alpha'] == a)]
            if sub.empty: continue
            grp = sub.groupby('round')['test_acc'].agg(['mean', 'std']).reset_index()
            ax.plot(grp['round'], grp['mean'] * 100, label=algo,
                    color=colors.get(algo, 'gray'), linewidth=1.5)
            ax.fill_between(grp['round'],
                            (grp['mean'] - grp['std'].fillna(0)) * 100,
                            (grp['mean'] + grp['std'].fillna(0)) * 100,
                            color=colors.get(algo, 'gray'), alpha=0.12)
        ax.set_xlabel('Round'); ax.set_ylabel('Acc (%)')
        ax.set_title(f'α = {a}'); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    plt.suptitle('Convergence across heterogeneity levels', fontsize=14)
    plt.tight_layout(); plt.savefig(plots_dir / "02_convergence.png", dpi=140); plt.close()

    ablation = ['EAHM-FL', 'EAHM-FL-no_selection', 'EAHM-FL-no_weighting']
    fig, axes = plt.subplots(1, len(alphas_sorted), figsize=(4 * len(alphas_sorted), 5), sharey=True)
    if len(alphas_sorted) == 1: axes = [axes]
    for ax, a in zip(axes, alphas_sorted):
        sub = agg[(agg['alpha'] == a) & (agg['algorithm'].isin(ablation))]
        if sub.empty: continue
        sub = sub.set_index('algorithm').reindex(ablation).reset_index()
        ax.bar(range(len(sub)), sub['final_acc_mean'] * 100,
               yerr=sub['final_acc_std'].fillna(0) * 100,
               color=[colors.get(x, 'gray') for x in sub['algorithm']], capsize=5)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([s.replace('EAHM-FL', '').lstrip('-') or 'full'
                            for s in sub['algorithm']], rotation=15)
        ax.set_title(f'α = {a}'); ax.set_ylabel('Final acc (%)'); ax.grid(alpha=0.3, axis='y')
    plt.suptitle('Ablation: selection vs weighting', fontsize=14)
    plt.tight_layout(); plt.savefig(plots_dir / "03_ablation.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    cs = cost.sort_values('avg_cum_up_gb')
    x = np.arange(len(cs)); w = 0.35
    ax.bar(x - w/2, cs['avg_cum_up_gb'], w, label='Upload',
           color=[colors.get(a, 'gray') for a in cs['algorithm']])
    ax.bar(x + w/2, cs['avg_cum_down_gb'], w, label='Download',
           color=[colors.get(a, 'gray') for a in cs['algorithm']], alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(cs['algorithm'], rotation=30, ha='right')
    ax.set_ylabel('Cumulative GB / client / 50 rounds')
    ax.set_title('Communication cost'); ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(plots_dir / "04_communication.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ts = cost.sort_values('avg_round_time_s')
    ax.bar(range(len(ts)), ts['avg_round_time_s'],
           color=[colors.get(a, 'gray') for a in ts['algorithm']])
    ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts['algorithm'], rotation=30, ha='right')
    ax.set_ylabel('Avg round time (s)'); ax.set_title('Computational cost'); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(plots_dir / "05_compute_time.png", dpi=140); plt.close()

    class_cols = [f'class{c}_acc' for c in range(NUM_CLASSES)]
    fd = finals[finals['alpha'] == 0.1]
    if not fd.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pa = fd.groupby('algorithm')[class_cols].mean()
        pa = pa.reindex([x for x in ALGORITHMS if x in pa.index])
        im = ax.imshow(pa.values * 100, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        ax.set_yticks(range(len(pa))); ax.set_yticklabels(pa.index)
        ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels([f'C{c}' for c in range(NUM_CLASSES)])
        ax.set_title('Per-class test accuracy (%) at α=0.1')
        for i in range(len(pa)):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f'{pa.values[i, j]*100:.0f}', ha='center', va='center',
                        fontsize=8, color='black' if pa.values[i, j] > 0.3 else 'white')
        plt.colorbar(im, ax=ax, label='Acc (%)')
        plt.tight_layout(); plt.savefig(plots_dir / "06_fairness_per_class.png", dpi=140); plt.close()

    eahm_files = list(client_dir.glob("EAHM-FL_alpha0.1_seed*.json"))
    if eahm_files:
        all_f = []
        for f in eahm_files:
            with open(f) as fh: d = json.load(fh)
            sh_list = d['selection_history']
            mat = np.zeros((NUM_CLIENTS, len(sh_list)))
            for ri, sel in enumerate(sh_list):
                for c in sel: mat[c, ri] = 1
            all_f.append(mat.cumsum(axis=1))
        af = np.mean(all_f, axis=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        for c in range(NUM_CLIENTS):
            ax.plot(af[c], label=f'Client {c}', linewidth=1.5)
        ax.set_xlabel('Round'); ax.set_ylabel('Cumulative selections')
        ax.set_title('EAHM-FL selection dynamics (α=0.1, mean over seeds)')
        ax.legend(ncol=2, fontsize=9, loc='upper left'); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(plots_dir / "07_selection_dynamics.png", dpi=140); plt.close()

    # Report
    lines = ["="*72, "PHASE A REPORT", "="*72,
             f"Generated: {datetime.now().isoformat()}",
             f"Total runs: {len(finals)}", ""]
    for a in alphas_sorted:
        lines.append(f"\nα = {a}\n" + "-"*60)
        sub = agg[agg['alpha'] == a].sort_values('final_acc_mean', ascending=False)
        for _, r in sub.iterrows():
            std = r['final_acc_std'] if not pd.isna(r['final_acc_std']) else 0.0
            lines.append(f"  {r['algorithm']:<25s} {r['final_acc_mean']*100:6.2f} ± {std*100:5.2f}%  (n={int(r['final_acc_n'])})")

    lines.append("\n" + "="*72)
    lines.append("PAIRWISE T-TESTS vs EAHM-FL (Welch)")
    lines.append("="*72)
    for a in HEADLINE_ALPHAS:
        lines.append(f"\nα = {a}")
        eahm = finals[(finals['alpha'] == a) & (finals['algorithm'] == 'EAHM-FL')]['test_acc'].values
        if len(eahm) < 2: continue
        for algo in ALGORITHMS:
            if algo == 'EAHM-FL': continue
            other = finals[(finals['alpha'] == a) & (finals['algorithm'] == algo)]['test_acc'].values
            if len(other) < 2: continue
            t, p = scipy_stats.ttest_ind(eahm, other, equal_var=False)
            diff = (eahm.mean() - other.mean()) * 100
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            lines.append(f"  EAHM-FL vs {algo:<25s} Δ={diff:+6.2f}pp  p={p:.4f} {sig}")

    lines.append("\n" + "="*72)
    lines.append("ABLATION CONTRIBUTIONS")
    lines.append("="*72)
    for a in alphas_sorted:
        full = agg[(agg['alpha'] == a) & (agg['algorithm'] == 'EAHM-FL')]
        ns = agg[(agg['alpha'] == a) & (agg['algorithm'] == 'EAHM-FL-no_selection')]
        nw = agg[(agg['alpha'] == a) & (agg['algorithm'] == 'EAHM-FL-no_weighting')]
        if full.empty or ns.empty or nw.empty: continue
        fa = full['final_acc_mean'].iloc[0] * 100
        nsa = ns['final_acc_mean'].iloc[0] * 100
        nwa = nw['final_acc_mean'].iloc[0] * 100
        sc = fa - nsa; wc = fa - nwa
        ratio = sc / wc if abs(wc) > 1e-6 else float('inf')
        lines.append(f"  α={a}: full={fa:.2f} no_sel={nsa:.2f} no_w={nwa:.2f}  "
                     f"sel_contrib={sc:+.2f} w_contrib={wc:+.2f} ratio={ratio:.2f}x")

    lines.append("\n" + "="*72)
    lines.append("COMM + COMPUTE COST")
    lines.append("="*72)
    lines.append(f"  {'Algorithm':<25s} {'UpGB':>8s} {'DownGB':>8s} {'Round_s':>9s} {'PeakGB':>8s}")
    for _, r in cost.sort_values('algorithm').iterrows():
        lines.append(f"  {r['algorithm']:<25s} {r['avg_cum_up_gb']:8.2f} "
                     f"{r['avg_cum_down_gb']:8.2f} {r['avg_round_time_s']:9.2f} {r['avg_peak_gpu_gb']:8.2f}")

    report = "\n".join(lines)
    print("\n" + report)
    (results_root / "PHASE_A_REPORT.txt").write_text(report)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main(results_dir, auto_pusher=None, data_root="/kaggle/working/data",
         abort_if_first_push_fails=True):
    """
    results_dir : Path-like for /kaggle/working/phase_a
    auto_pusher : AutoPusher instance (or None for local-only)
    data_root   : CIFAR-10 download root
    abort_if_first_push_fails : if True, raise after first failed push
                                so we know within ~5 min there's a problem
    """
    results_dir = Path(results_dir)
    paths = {
        'results_root': results_dir,
        'runs':   results_dir / "runs",
        'client': results_dir / "client_stats",
        'ckpt':   results_dir / "ckpts",
        'plots':  results_dir / "plots",
    }
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_list = build_run_list()
    print(f"\nTotal runs: {len(run_list)}")
    completed = sum(1 for (a, al, s) in run_list
                    if (paths['runs'] / f"{a}_alpha{al}_seed{s}.done").exists())
    print(f"Already complete: {completed}")

    # If auto_pusher provided, restore from repo first
    if auto_pusher is not None:
        n_restored = auto_pusher.restore_from_repo()
        print(f"Restored {n_restored} files from repo")
        # Re-check completed count after restore
        completed = sum(1 for (a, al, s) in run_list
                        if (paths['runs'] / f"{a}_alpha{al}_seed{s}.done").exists())
        print(f"After restore: {completed}/{len(run_list)} complete")

    first_push_attempted = False

    for i, (algo, a, s) in enumerate(run_list):
        print(f"\n[{i+1}/{len(run_list)}] {algo}_alpha{a}_seed{s}")
        try:
            result = run_one(algo, a, s, paths, data_root)
        except Exception as e:
            print(f"!! Run failed: {e}")
            import traceback; traceback.print_exc()
            continue

        if auto_pusher is not None and result == "DONE":
            auto_pusher.run_completed()
            pushed, ok = auto_pusher.maybe_push()
            if pushed:
                if not first_push_attempted:
                    first_push_attempted = True
                    if not ok and abort_if_first_push_fails:
                        print("\n" + "="*72)
                        print("FIRST PUSH FAILED — aborting so you can fix it")
                        print(f"Last error: {auto_pusher.last_push_error}")
                        print("="*72)
                        raise RuntimeError("first push failed; see above")

    # ---- Aggregate and final push ----
    print("\n" + "="*72)
    print("AGGREGATING + PLOTTING")
    print("="*72)
    aggregate_and_plot(paths)

    if auto_pusher is not None:
        auto_pusher.final_push()
        print(f"\nFinal pusher summary: {auto_pusher.summary()}")
    print(f"\nDone. Outputs at: {results_dir}")
