"""
run_eval.py
============
Standalone evaluation script for RS1 (BPR-MF) and RS2 (LightGCN).

Usage:
    python run_eval.py

What this script does:
  1. Mounts and loads preprocessed data from Google Drive
  2. Trains RS1 (BPR-MF) or loads saved checkpoint
  3. Trains RS2 (LightGCN) or loads saved checkpoint
  4. Evaluates both models using NDCG@10 and Novelty@10
  5. Prints side-by-side comparison summary
  6. Saves results to outputs/results_summary.csv
"""

import os
import sys
import json
import pickle
import time
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── PyTorch Geometric for LightGCN 
try:
    from torch_geometric.nn import LGConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("  torch_geometric not found.")
    print("    Install with: pip install torch-geometric")
    print("    RS2 will use BPR-MF as fallback.\n")

# CONFIGURATION — Update BASE path if your folder name is different


# Update this path to match your Google Drive folder name exactly
BASE      = '/content/drive/MyDrive/Recommendation System'
PROC      = f'{BASE}/data/processed'
MDIR      = f'{BASE}/models'
ODIR      = f'{BASE}/outputs'

os.makedirs(MDIR, exist_ok=True)
os.makedirs(ODIR, exist_ok=True)

# Hyperparameters — must match what was used in notebooks
RANDOM_SEED   = 42
EMBEDDING_DIM = 64
EPOCHS_RS1    = 30
EPOCHS_RS2    = 80
BATCH_SIZE    = 2048
LR            = 0.001
REG_LAMBDA    = 1e-4
K             = 10
NUM_LAYERS    = 3
RETRAIN       = False  # Set True to force retrain even if checkpoints exist

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# STEP 1 — MOUNT GOOGLE DRIVE (only needed when running in Colab)


def mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print(" Google Drive mounted.")
    except ImportError:
        print("ℹ️  Not running in Colab — skipping Drive mount.")
        print(f"   Make sure data exists at: {PROC}\n")


# STEP 2 — LOAD DATA


def load_data():
    print("\n" + "="*60)
    print("  STEP 1/5 — Loading Preprocessed Data")
    print("="*60)

    # Check files exist
    required_files = [
        f'{PROC}/train.csv',
        f'{PROC}/test.csv',
        f'{PROC}/dataset_info.json',
        f'{PROC}/train_user_items.pkl',
        f'{PROC}/item_popularity.pkl',
        f'{PROC}/movies.csv',
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f" Missing file: {f}")
            print("   Please run 00_Data_Preparation.ipynb first.")
            sys.exit(1)

    train_df  = pd.read_csv(f'{PROC}/train.csv')
    test_df   = pd.read_csv(f'{PROC}/test.csv')
    movies_df = pd.read_csv(f'{PROC}/movies.csv')

    with open(f'{PROC}/dataset_info.json') as f:
        info = json.load(f)
    with open(f'{PROC}/train_user_items.pkl', 'rb') as f:
        train_user_items = pickle.load(f)
    with open(f'{PROC}/item_popularity.pkl', 'rb') as f:
        item_popularity = pickle.load(f)

    #  Use max index + 1 to avoid embedding out-of-range errors
    NUM_USERS = int(train_df['user_idx'].max()) + 1
    NUM_ITEMS = int(train_df['item_idx'].max()) + 1

    idx_to_title = dict(zip(movies_df['item_idx'], movies_df['title']))

    print(f"  Users        : {NUM_USERS:,}")
    print(f"  Items        : {NUM_ITEMS:,}")
    print(f"  Train        : {len(train_df):,}")
    print(f"  Test         : {len(test_df):,}  (1 per user)")
    print(f"  Device       : {DEVICE}")
    print(f"  PyG available: {PYG_AVAILABLE}")

    return (train_df, test_df, movies_df, train_user_items,
            item_popularity, idx_to_title, NUM_USERS, NUM_ITEMS)


# BPR DATASET — shared by RS1 and RS2


class BPRDataset(Dataset):
    def __init__(self, train_df, train_user_items, num_items):
        self.data             = list(zip(train_df['user_idx'], train_df['item_idx']))
        self.train_user_items = train_user_items
        self.num_items        = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, pos = self.data[idx]
        neg = np.random.randint(0, self.num_items)
        while neg in self.train_user_items.get(user, set()):
            neg = np.random.randint(0, self.num_items)
        return (
            torch.tensor(int(user), dtype=torch.long),
            torch.tensor(int(pos),  dtype=torch.long),
            torch.tensor(int(neg),  dtype=torch.long),
        )


# RS1 — BPR MATRIX FACTORIZATION

class BPRMF(nn.Module):
    """
    Conventional Matrix Factorization with BPR ranking loss.
    Learns user and item embeddings independently via dot product scoring.
    """
    def __init__(self, num_users, num_items, dim=64, reg=1e-4):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, dim)
        self.item_emb = nn.Embedding(num_items, dim)
        self.reg = reg
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, u, pi, ni):
        ue = self.user_emb(u)
        pe = self.item_emb(pi)
        ne = self.item_emb(ni)
        bpr = -torch.log(
            torch.sigmoid((ue * pe).sum(1) - (ue * ne).sum(1)) + 1e-10
        ).mean()
        reg = self.reg * (
            ue.norm(2).pow(2) + pe.norm(2).pow(2) + ne.norm(2).pow(2)
        ) / u.shape[0]
        return bpr + reg

    def get_scores(self, user_idx):
        u = self.user_emb.weight[user_idx]
        return torch.matmul(self.item_emb.weight, u).detach().cpu().numpy()


def train_rs1(train_df, train_user_items, NUM_USERS, NUM_ITEMS):
    print("\n" + "="*60)
    print("  STEP 2/5 — RS1 (BPR-MF)")
    print("="*60)

    ckpt = f'{MDIR}/rs1_bprmf.pth'
    model = BPRMF(NUM_USERS, NUM_ITEMS, EMBEDDING_DIM, REG_LAMBDA).to(DEVICE)

    if os.path.exists(ckpt) and not RETRAIN:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print("   Loaded RS1 from saved checkpoint.")
        return model

    loader = DataLoader(
        BPRDataset(train_df, train_user_items, NUM_ITEMS),
        batch_size=BATCH_SIZE, shuffle=True
    )
    opt = optim.Adam(model.parameters(), lr=LR)
    t0  = time.time()

    print(f"  Training for {EPOCHS_RS1} epochs...")
    for epoch in range(1, EPOCHS_RS1 + 1):
        model.train()
        total_loss = 0
        for u, p, n in loader:
            u, p, n = u.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            opt.zero_grad()
            loss = model(u, p, n)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS_RS1} | Loss: {avg:.5f} | {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), ckpt)
    print(f"  RS1 trained in {(time.time()-t0)/60:.1f} min and saved.")
    return model


def recommend_rs1(model, user_idx, train_user_items, k=10):
    model.eval()
    with torch.no_grad():
        scores = model.get_scores(user_idx)
    for item in train_user_items.get(user_idx, set()):
        scores[item] = -np.inf
    return np.argsort(scores)[::-1][:k].tolist()


# RS2 — LightGCN


if PYG_AVAILABLE:
    class LightGCN(nn.Module):
        """
        Advanced recommender using graph convolutional network.
        Extends BPR-MF by propagating embeddings through the
        user-item interaction graph across multiple layers,
        capturing higher-order connectivity signals.
        """
        def __init__(self, num_users, num_items, dim=64, layers=3, reg=1e-4):
            super().__init__()
            self.nu    = num_users
            self.ni    = num_items
            self.reg   = reg
            self.emb   = nn.Embedding(num_users + num_items, dim)
            self.convs = nn.ModuleList([LGConv() for _ in range(layers)])
            nn.init.xavier_uniform_(self.emb.weight)

        def propagate(self, edge_index):
            x   = self.emb.weight
            out = [x]
            for conv in self.convs:
                x = conv(x, edge_index)
                out.append(x)
            return torch.stack(out).mean(0)  # mean pooling across all layers

        def forward(self, edge_index, u, pi, ni):
            e   = self.propagate(edge_index)
            ue  = e[:self.nu][u]
            pe  = e[self.nu:][pi]
            ne  = e[self.nu:][ni]
            bpr = -torch.log(
                torch.sigmoid((ue * pe).sum(1) - (ue * ne).sum(1)) + 1e-10
            ).mean()
            reg = self.reg * (
                self.emb.weight[u].norm(2).pow(2) +
                self.emb.weight[self.nu + pi].norm(2).pow(2) +
                self.emb.weight[self.nu + ni].norm(2).pow(2)
            ) / u.shape[0]
            return bpr + reg

        def get_scores(self, edge_index, user_idx):
            e = self.propagate(edge_index)
            return torch.matmul(e[self.nu:], e[user_idx]).detach().cpu().numpy()

else:
    # Fallback if PyG not installed
    LightGCN = BPRMF


def build_graph(train_df, NUM_USERS):
    u_arr = train_df['user_idx'].values
    i_arr = train_df['item_idx'].values + NUM_USERS
    src   = np.concatenate([u_arr, i_arr])
    dst   = np.concatenate([i_arr, u_arr])
    return torch.tensor([src, dst], dtype=torch.long).to(DEVICE)


def train_rs2(train_df, train_user_items, NUM_USERS, NUM_ITEMS, edge_index):
    print("\n" + "="*60)
    print("  STEP 3/5 — RS2 (LightGCN)")
    print("="*60)

    ckpt = f'{MDIR}/rs2_lightgcn.pth'

    if PYG_AVAILABLE:
        model = LightGCN(NUM_USERS, NUM_ITEMS, EMBEDDING_DIM, NUM_LAYERS, REG_LAMBDA).to(DEVICE)
    else:
        print("  Using BPR-MF fallback for RS2 (PyG not available).")
        model = BPRMF(NUM_USERS, NUM_ITEMS, EMBEDDING_DIM, REG_LAMBDA).to(DEVICE)

    if os.path.exists(ckpt) and not RETRAIN:
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print("  Loaded RS2 from saved checkpoint.")
        return model

    loader = DataLoader(
        BPRDataset(train_df, train_user_items, NUM_ITEMS),
        batch_size=BATCH_SIZE, shuffle=True
    )
    opt = optim.Adam(model.parameters(), lr=LR)
    t0  = time.time()

    print(f"  Training for {EPOCHS_RS2} epochs (graph layers: {NUM_LAYERS})...")
    for epoch in range(1, EPOCHS_RS2 + 1):
        model.train()
        total_loss = 0
        for u, p, n in loader:
            u, p, n = u.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            opt.zero_grad()
            if PYG_AVAILABLE:
                loss = model(edge_index, u, p, n)
            else:
                loss = model(u, p, n)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS_RS2} | Loss: {avg:.5f} | {time.time()-t0:.0f}s")

    torch.save(model.state_dict(), ckpt)
    print(f" RS2 trained in {(time.time()-t0)/60:.1f} min and saved.")
    return model


def recommend_rs2(model, edge_index, user_idx, train_user_items, k=10):
    model.eval()
    with torch.no_grad():
        if PYG_AVAILABLE:
            scores = model.get_scores(edge_index, user_idx)
        else:
            scores = model.get_scores(user_idx)
    for item in train_user_items.get(user_idx, set()):
        scores[item] = -np.inf
    return np.argsort(scores)[::-1][:k].tolist()


# STEP 4 — EVALUATION METRICS


def ndcg_at_k(recs, test_item, k=10):
    """
    NDCG@K — Normalised Discounted Cumulative Gain.
    Measures ranking accuracy. A hit at rank 1 scores higher than rank K.
    Formula: 1 / log2(rank + 1) if test_item in top-K, else 0.
    """
    if test_item in recs[:k]:
        rank = recs[:k].index(test_item) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def novelty_at_k(recs, item_popularity, k=10):
    """
    Novelty@K — measures how surprising/non-obvious recommendations are.
    Popular items score low; rare items score high.
    Formula: mean(-log2(popularity(i))) for i in top-K.
    """
    return np.mean([
        -np.log2(item_popularity.get(i, 1e-10) + 1e-10)
        for i in recs[:k]
    ])


def evaluate_model(model, recommender_fn, edge_index,
                   test_df, train_user_items, item_popularity,
                   label, k=10):
    print(f"\n  Evaluating {label} on {len(test_df):,} users...")
    model.eval()
    ndcg_list, nov_list = [], []
    t0 = time.time()

    for i, (_, row) in enumerate(test_df.iterrows()):
        user_idx  = int(row['user_idx'])
        test_item = int(row['item_idx'])
        recs      = recommender_fn(model, edge_index, user_idx, train_user_items, k)
        ndcg_list.append(ndcg_at_k(recs, test_item, k))
        nov_list.append(novelty_at_k(recs, item_popularity, k))

        if (i + 1) % 1000 == 0:
            print(f"    {i+1:,}/{len(test_df):,} users evaluated...")

    ndcg    = np.mean(ndcg_list)
    novelty = np.mean(nov_list)
    print(f"  Done in {time.time()-t0:.1f}s")
    return ndcg, novelty

# STEP 5 — DEMO RECOMMENDATIONS FOR 6 USERS

def show_demo_recommendations(rs1_model, rs2_model, edge_index,
                               train_user_items, idx_to_title, k=10):
    print("\n" + "="*60)
    print("  DEMO — Top-10 Recommendations for 6 Users")
    print("="*60)

    demo_users = list(train_user_items.keys())[:6]

    for user_idx in demo_users:
        recs_rs1 = recommend_rs1(rs1_model, user_idx, train_user_items, k)
        recs_rs2 = recommend_rs2(rs2_model, edge_index, user_idx, train_user_items, k)

        print(f"\n  User {user_idx}")
        print(f"  {'Rank':<6} {'RS1 (BPR-MF)':<40} {'RS2 (LightGCN)':<40}")
        print(f"  {'-'*86}")
        for rank, (i1, i2) in enumerate(zip(recs_rs1, recs_rs2), 1):
            t1 = idx_to_title.get(i1, f'Movie_{i1}')[:38]
            t2 = idx_to_title.get(i2, f'Movie_{i2}')[:38]
            print(f"  {rank:<6} {t1:<40} {t2:<40}")

# MAIN

def main():
    total_start = time.time()

    print("\n" + "="*60)
    print("  Recommender System Evaluation — RS1 vs RS2")
    print("  BPR-MF  vs  LightGCN  |  MovieLens 150K")
    print("="*60)

    # Mount Drive
    mount_drive()

    # Load data
    (train_df, test_df, movies_df, train_user_items,
     item_popularity, idx_to_title, NUM_USERS, NUM_ITEMS) = load_data()

    # Build graph (used by RS2)
    edge_index = build_graph(train_df, NUM_USERS)

    # Train / load RS1
    rs1_model = train_rs1(train_df, train_user_items, NUM_USERS, NUM_ITEMS)

    # Train / load RS2
    rs2_model = train_rs2(train_df, train_user_items, NUM_USERS, NUM_ITEMS, edge_index)

    # Evaluate RS1
    print("\n" + "="*60)
    print("  STEP 4/5 — Evaluation")
    print("="*60)

    rs1_ndcg, rs1_novelty = evaluate_model(
        rs1_model,
        lambda m, ei, u, tui, k: recommend_rs1(m, u, tui, k),
        None, test_df, train_user_items, item_popularity,
        "RS1 (BPR-MF)", K
    )

    # Evaluate RS2
    rs2_ndcg, rs2_novelty = evaluate_model(
        rs2_model,
        lambda m, ei, u, tui, k: recommend_rs2(m, ei, u, tui, k),
        edge_index, test_df, train_user_items, item_popularity,
        "RS2 (LightGCN)", K
    )

    # Demo recommendations
    show_demo_recommendations(
        rs1_model, rs2_model, edge_index,
        train_user_items, idx_to_title, K
    )

    # Compute improvements
    ndcg_change    = (rs2_ndcg    - rs1_ndcg)    / rs1_ndcg    * 100
    novelty_change = (rs2_novelty - rs1_novelty) / rs1_novelty * 100
    ndcg_sign      = "+" if ndcg_change    >= 0 else ""
    novelty_sign   = "+" if novelty_change >= 0 else ""

    # Print final summary
    print("\n" + "="*60)
    print("  STEP 5/5 — Final Results Summary")
    print("="*60)
    print(f"\n  {'Metric':<20} {'RS1 (BPR-MF)':>15} {'RS2 (LightGCN)':>15}")
    print(f"  {'-'*52}")
    print(f"  {'NDCG@'+str(K):<20} {rs1_ndcg:>15.5f} {rs2_ndcg:>15.5f}")
    print(f"  {'Novelty@'+str(K):<20} {rs1_novelty:>15.4f} {rs2_novelty:>15.4f}")
    print(f"  {'-'*52}")
    print(f"  {'NDCG Change':<20} {'':>15} {ndcg_sign}{ndcg_change:>14.2f}%")
    print(f"  {'Novelty Change':<20} {'':>15} {novelty_sign}{novelty_change:>14.2f}%")
    print("="*60)

    # Save results CSV
    summary_path = f'{ODIR}/results_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', f'ndcg@{K}', f'novelty@{K}'])
        writer.writerow(['RS1_BPR_MF',   round(rs1_ndcg, 6), round(rs1_novelty, 6)])
        writer.writerow(['RS2_LightGCN', round(rs2_ndcg, 6), round(rs2_novelty, 6)])

    # Save JSON results
    with open(f'{ODIR}/rs1_results.json', 'w') as f:
        json.dump({'model': 'RS1_BPR_MF',
                   f'ndcg@{K}': round(rs1_ndcg, 6),
                   f'novelty@{K}': round(rs1_novelty, 6)}, f, indent=2)
    with open(f'{ODIR}/rs2_results.json', 'w') as f:
        json.dump({'model': 'RS2_LightGCN',
                   f'ndcg@{K}': round(rs2_ndcg, 6),
                   f'novelty@{K}': round(rs2_novelty, 6)}, f, indent=2)

    total_time = (time.time() - total_start) / 60
    print(f"\n  Results saved to: {summary_path}")
    print(f" Total runtime: {total_time:.1f} minutes")
    print("\n Evaluation complete!\n")


if __name__ == '__main__':
    main()
