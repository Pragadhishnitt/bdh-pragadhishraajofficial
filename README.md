# BDH for Financial Reasoning: Bridging Neuroscience and Market Dynamics

**Baby Dragon Hatchling (BDH)** is a biologically inspired neural network architecture that combines sparse, interpretable activations with Hebbian working memory.

> **This repository contains a specialized implementation of BDH adapted for the Financial Domain (Track 2, Path A).**
> We explore how BDH's "neurons that fire together, wire together" dynamics can capture the evolving regimes of financial markets using S&P 500 earnings transcripts.

---

## 🚀 Key Features

*   **Two-Stage Protocol**:
    *   **Stage A (Pre-training)**: Learns a modular, sparse representation of financial language using **L1 + L2,1 regularization**.
    *   **Stage B (Hebbian Inference)**: Freezes weights and enables **dynamic synaptic plasticity** to adapt to new market regimes (2019-2024) without retraining.
*   **Frontier Metrics**: Implements **TMI** (Topological Modularity), **SPS** (Synaptic Persistence), and **SEC** (Sparsity-Entropy Correlation) to validate brain-like behavior.
*   **Domain-Specific Design**: Tailored for financial text with **Strict Causal Masking** and **Rotary Position Embeddings (RoPE)** for long-range context.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Pragadhishnitt/bdh-pragadhishraajofficial.git
cd bdh-pragadhishraajofficial

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃 Usage

We provide a unified `pipeline.py` to handle training, evaluation, and visualization.

### Command Structure
```bash
python pipeline.py --mode <MODE> --sector <SECTOR> --reg <REGULARIZATION>
```

### Available Options

#### 1. Modes (`--mode`)
| Mode | Description |
|------|-------------|
| `full` | **Complete Pipeline**: Train (Stage A) + Eval (Stage B) + Viz. |
| `train` | **Stage A Only**: Structural pre-training on historical data. |
| `eval` | **Stage B Only**: Hebbian inference evaluation (requires checkpoint). |
| `viz` | **Visualization**: Generate topology plots and metrics (requires checkpoint). |
| `quick` | **Debug**: Fast 100-iteration test run. |
| `medium` | **Balanced**: 3000 iterations (good for Kaggle/Colab). |
| `a100` | **High-Compute**: Optimized for A100 GPUs (larger batch/model). |
| `tech` | **Tech Sector**: Specialized preset for Technology sector (12k iters). |

#### 2. Sectors (`--sector`)
Filter the dataset to specific industries:
*   `all` (Default)
*   `technology`
*   `healthcare`
*   `financials`
*   `energy`

#### 3. Regularization (`--reg`)
Control the structural constraints during Stage A:
*   `both` (Default): L1 (Sparsity) + L2,1 (Modularity). **Recommended.**
*   `l1`: Activation sparsity only.
*   `l21`: Weight clustering only.

### Examples

**Run full experiment on Technology sector:**
```bash
python pipeline.py --mode full --sector technology --reg both
```

**Train base model on all sectors:**
```bash
python pipeline.py --mode train --sector all
```

**Evaluate existing checkpoint:**
```bash
python pipeline.py --mode eval --checkpoint outputs/checkpoints/model_final.pt
```

---

## 📊 Key Results

Our experiments on S&P 500 transcripts demonstrate:

| Metric | Result | Interpretation |
|--------|--------|----------------|
| **TMI (Modularity)** | **Q ≈ 0.7** | Neurons form distinct functional communities (e.g., "Risk", "Growth"). |
| **SPS (Memory)** | **H > 0.6** | Hebbian memory persists over long contexts (Hurst exponent > 0.5). |
| **SEC (Reasoning)** | **r > 0.2** | Higher surprise (perplexity) triggers denser brain activity. |

> *See `ULTIMATE_BDH_WALKTHROUGH.md` for a deep dive into these metrics.*

---

## 📂 Repository Structure

*   `bdh.py`: Core model architecture (RoPE, Sparse Activations).
*   `pipeline.py`: Unified entry point for experiments.
*   `new_dataset/`: Financial data loaders and config.
    *   `train_finance.py`: Stage A training loop.
    *   `evaluate_hebbian.py`: Stage B Hebbian inference.
    *   `dragon_metrics.py`: Implementation of TMI, SPS, SEC.
*   `ULTIMATE_BDH_WALKTHROUGH.md`: **Comprehensive technical guide.**

---

## 📚 Documentation

*   **[ULTIMATE_BDH_WALKTHROUGH.md](ULTIMATE_BDH_WALKTHROUGH.md)**: The definitive guide to our architecture, training strategy, and results.
*   **[Original Paper](https://doi.org/10.48550/arXiv.2509.26507)**: *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain*.

---

## 🖼️ Visualizations

*(Placeholders for your images - add them to the `figs/` directory)*

<img src="figs/architecture.png" width="600" alt="BDH Architecture"/>

---

## Acknowledgements

Based on the original BDH implementation by [Pathway](https://pathway.com). Adapted for the **Synaptix Frontier AI Challenge (Track 2)**.
