# 🎬 Recommender System: BPR-MF vs LightGCN

> End-to-end implementation and evaluation of two personalised ranking systems on the MovieLens 1M dataset — a conventional Matrix Factorization baseline (RS1) vs a Graph Neural Network approach (RS2).

---

## 📌 Project Overview

This project implements and compares two recommender systems:

| | RS1 | RS2 |
|---|---|---|
| **Model** | BPR-MF (Bayesian Personalised Ranking – Matrix Factorization) | LightGCN (Light Graph Convolutional Network) |
| **Type** | Conventional collaborative filtering | Graph-based collaborative filtering |
| **Embeddings** | Learned independently via dot product | Learned via multi-layer graph propagation |
| **Loss** | Pairwise BPR ranking loss | Pairwise BPR ranking loss |
| **NDCG@10** | 0.0149 | 0.01533 (+2.9%) |
| **Novelty@10** | 4.42 | 4.3832 |

Both systems are trained on the same sampled dataset, use the same train/test split, and are evaluated under identical conditions for a fair comparison.

---

## 📂 Project Structure

```
├── RS1_BPR_MF.ipynb          # Notebook: BPR-MF implementation & evaluation
├── RS2_LightGCN.ipynb        # Notebook: LightGCN implementation & evaluation
├── run_eval.py               # Runs evaluation for both RS1 and RS2, prints summary
├── requirements.txt          # Python dependencies with versions
├── README.md                 # You are here

```

---

## 📊 Dataset

- **Dataset:** [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) by GroupLens
- **Total ratings:** ~1,000,000
- **Sampled interactions:** 150,000 (randomly sampled, fixed seed for reproducibility)
- **Rating scale:** 1–5 (explicit feedback, treated as implicit for ranking)
- **Task:** Top-10 personalised item ranking

### Download the Dataset

```bash
# Download and extract MovieLens 1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/
```

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.9+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1 — Run Full Evaluation (Both Models)

```bash
python run_eval.py
```

This will:
- Load and sample the dataset
- Apply the train/test split
- Train RS1 (BPR-MF) and RS2 (LightGCN)
- Evaluate both on NDCG@10 and Novelty@10
- Print a side-by-side summary of results

### Option 2 — Run Notebooks Interactively

Open and run the notebooks in order:

```bash
jupyter notebook RS1_BPR_MF.ipynb
jupyter notebook RS2_LightGCN.ipynb
```

Each notebook is self-contained with step-by-step commentary.

---

## 🧪 Experimental Design

### Train/Test Split

- **Strategy:** Leave-one-out per user
- For each user, one interaction is held out as the test item
- Remaining interactions are used for training
- Identical split applied to both RS1 and RS2

### Negative Sampling

- For each test user, 99 random negative items (unseen) are sampled
- The model ranks the 1 positive item among 100 candidates (1 positive + 99 negatives)

### Evaluation Metrics

**Metric 1 — NDCG@10 (Normalised Discounted Cumulative Gain)**

$$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}$$

Measures ranking quality — rewards placing the relevant item higher in the Top-10 list.

**Metric 2 — Novelty@10**

$$Novelty@K = -\frac{1}{K} \sum_{i=1}^{K} \log_2 p(i)$$

where $p(i)$ is the popularity of item $i$ (fraction of users who interacted with it). Higher novelty means the system recommends less popular, more surprising items.

---

## 📈 Results

```
==========================================
        Evaluation Summary
==========================================
  Metric          RS1 (BPR-MF)   RS2 (LightGCN)
------------------------------------------
  NDCG@10         0.01490        0.01533
  Novelty@10      4.4200         4.3832
==========================================
  Users evaluated: 5,934
==========================================
```

### Interpretation

- **RS2 improves ranking accuracy by +2.9%** over RS1, demonstrating that graph-based propagation captures higher order user item connectivity that simple matrix factorization misses.
- **RS1 has slightly higher novelty**, suggesting BPR-MF occasionally recommends more niche items, while LightGCN's graph structure slightly favours well-connected (more popular) items.

---

## 🏗️ Model Architecture

### RS1 — BPR-MF

```
User ID ──► User Embedding (d-dim)
                                    ──► Dot Product ──► Score ──► BPR Loss
Item ID ──► Item Embedding (d-dim)
```

- Learns independent user and item embedding matrices
- Pairwise BPR loss: pushes score of positive items above negative items
- Optimised with Adam

### RS2 — LightGCN

```
User-Item Bipartite Graph
        │
        ▼
Layer 0: Initial Embeddings (E⁰)
        │
        ▼
Layer 1: E¹ = Ã · E⁰        (graph propagation)
        │
        ▼
Layer 2: E² = Ã · E¹
        │
        ▼
Final:  E = (E⁰ + E¹ + E²) / 3   (layer aggregation)
        │
        ▼
Score = Eᵤ · Eᵢᵀ  ──► BPR Loss
```

- Constructs a symmetric normalised adjacency matrix from the user-item graph
- Propagates embeddings across multiple layers to capture multi-hop relationships
- Aggregates all layer embeddings for final representations

---

## 👤 Sample Recommendations

Both models generate Top-10 personalised recommendations. The notebooks demonstrate outputs for 6 sample users side by side.

```
User 1 — RS1 (BPR-MF) Top-5:
  1. Schindler's List (1993)
  2. The Shawshank Redemption (1994)
  3. Pulp Fiction (1994)
  4. Fargo (1996)
  5. Goodfellas (1990)

User 1 — RS2 (LightGCN) Top-5:
  1. The Silence of the Lambs (1991)
  2. Schindler's List (1993)
  3. Se7en (1995)
  4. The Usual Suspects (1995)
  5. Pulp Fiction (1994)
```

---

## 📦 Requirements

See `requirements.txt` for full dependency list. Key libraries:

```
torch>=1.13.0
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.1.0
tqdm>=4.64.0
jupyter>=1.0.0
```

---

## 🔁 Reproducibility

All experiments use a fixed random seed (`seed=42`) for:
- Dataset sampling
- Train/test splitting
- Negative sampling during evaluation
- Model weight initialisation

---

## 📬 Contact

Built by **Zohaib** — ML Engineer & Data Scientist

- 🔗 [Fiverr Profile] https://www.fiverr.com/s/gDVagNE
- 📧 Open to ML projects — feel free to reach out!

---

## 📄 License

This project is for educational and portfolio purposes. The MovieLens dataset is provided by GroupLens Research and subject to their [terms of use](https://grouplens.org/datasets/movielens/).
