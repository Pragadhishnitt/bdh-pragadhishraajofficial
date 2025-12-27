"""
Visualization Utilities for BDH Competition

Implements visualization requirements 
- Topology plots (SVG of learned graph structure)
- Sparsity heatmaps (activation density over text)
- Activation atlases (neurons colored by semantic role)
- Graph export for Gephi/NetworkX
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import networkx as nx
    from community import community_louvain
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from dragon_metrics import compute_tmi, TMIResult


def plot_topology(
    model: nn.Module,
    output_path: str = "output/topology.svg",
    top_k_percent: float = 5.0,
    figsize: Tuple[int, int] = (12, 12),
    node_size_scale: float = 50.0,
    max_nodes: int = 500,  # LIMIT nodes to prevent slow computation
) -> str:
    """
    Generate SVG visualization of learned graph structure.
    
    OPTIMIZED VERSION: Limits to top max_nodes nodes by degree
    to prevent extremely slow spring_layout computation.
    """
    if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
        raise ImportError("matplotlib and networkx required for topology plotting")
    
    print("  Computing TMI...")
    # Compute TMI to get graph structure
    tmi_result = compute_tmi(model, top_k_percent)
    adj = tmi_result.adjacency_matrix
    
    print(f"  Adjacency matrix shape: {adj.shape}")
    
    # Find non-zero edges and limit nodes
    rows, cols = np.where(adj > 0)
    print(f"  Total edges: {len(rows)}")
    
    if len(rows) == 0:
        print("Warning: No edges in graph")
        # Create empty plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No edges found in graph", ha='center', va='center', fontsize=14)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        return output_path
    
    # OPTIMIZATION: Select top max_nodes by degree
    unique_nodes = np.unique(np.concatenate([rows, cols]))
    print(f"  Unique nodes: {len(unique_nodes)}")
    
    if len(unique_nodes) > max_nodes:
        print(f"  Limiting to top {max_nodes} nodes by degree...")
        node_degrees = np.bincount(np.concatenate([rows, cols]), minlength=adj.shape[0])
        top_nodes = set(np.argsort(node_degrees)[-max_nodes:])
        mask = np.array([r in top_nodes and c in top_nodes for r, c in zip(rows, cols)])
        rows, cols = rows[mask], cols[mask]
        print(f"  Filtered edges: {len(rows)}")
    
    # Build sparse graph
    print("  Building graph...")
    G = nx.Graph()
    edges = [(int(r), int(c), {"weight": float(adj[r, c])}) 
             for r, c in zip(rows, cols) if r < c]
    G.add_edges_from(edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    print(f"  Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
    
    if len(G.nodes()) == 0:
        print("Warning: No non-isolated nodes in graph")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No connected nodes", ha='center', va='center', fontsize=14)
        ax.axis("off")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close()
        return output_path
    
    # Community detection
    print("  Running community detection...")
    partition = community_louvain.best_partition(G, weight="weight")
    
    # Node colors by community
    communities = set(partition.values())
    cmap = plt.cm.get_cmap("tab20", len(communities))
    node_colors = [cmap(partition[node]) for node in G.nodes()]
    
    # Node sizes by degree
    degrees = dict(G.degree(weight="weight"))
    max_degree = max(degrees.values()) if degrees else 1
    node_sizes = [node_size_scale * (degrees[n] / max_degree + 0.1) for n in G.nodes()]
    
    # OPTIMIZED LAYOUT: Use kamada_kawai for <500 nodes, random for larger
    print("  Computing layout...")
    n_nodes = len(G.nodes())
    if n_nodes <= 200:
        pos = nx.kamada_kawai_layout(G)
    elif n_nodes <= 500:
        # Faster spring with fewer iterations
        pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)
    else:
        # Even faster: random layout with some structure
        pos = nx.random_layout(G, seed=42)
    
    # Create figure
    print("  Drawing plot...")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges (simplified - no per-edge alpha for speed)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.3, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
    )
    
    # Title and legend
    ax.set_title(
        f"BDH Neural Topology\n"
        f"Modularity Q = {tmi_result.modularity_q:.4f} | "
        f"Communities = {tmi_result.num_communities} | "
        f"Nodes = {n_nodes}",
        fontsize=14,
    )
    
    # Create legend for top communities
    top_communities = sorted(
        [(c, sum(1 for n, pc in partition.items() if pc == c)) 
         for c in communities],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    
    legend_patches = [
        Patch(color=cmap(c), label=f"Community {c} (n={size})")
        for c, size in top_communities
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)
    
    ax.axis("off")
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"  Saved topology plot: {output_path}")
    return output_path


def plot_sparsity_heatmap(
    activations: np.ndarray,
    text_tokens: List[str],
    output_path: str = "output/sparsity_heatmap.png",
    figsize: Tuple[int, int] = (16, 8),
) -> str:
    """
    Generate sparsity heatmap overlaid on transcript text.
    
    From PDF Appendix:
    "Dynamic visualization of activation density over time
    overlaid on transcript text."
    
    Args:
        activations: (T, N) array of neuron activations per token
        text_tokens: List of token strings
        output_path: Path to save image
        figsize: Figure size
        
    Returns:
        Path to saved visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for heatmap plotting")
    
    # Compute sparsity (fraction non-zero) per token
    sparsity = (activations > 0).mean(axis=1)  # (T,)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 4])
    
    # Top panel: sparsity over time
    ax1.plot(sparsity, 'b-', linewidth=1)
    ax1.fill_between(range(len(sparsity)), sparsity, alpha=0.3)
    ax1.set_ylabel("Activation Density")
    ax1.set_xlim(0, len(sparsity))
    ax1.set_title("Sparsity Dynamics During Inference")
    
    # Bottom panel: heatmap
    if SEABORN_AVAILABLE:
        # Show top 100 most active neurons
        mean_activation = activations.mean(axis=0)
        top_neurons = np.argsort(mean_activation)[-100:]
        
        sns.heatmap(
            activations[:, top_neurons].T,
            ax=ax2,
            cmap="viridis",
            cbar_kws={"label": "Activation"},
        )
        ax2.set_xlabel("Token Position")
        ax2.set_ylabel("Neuron (top 100)")
    else:
        ax2.imshow(activations.T, aspect='auto', cmap='viridis')
        ax2.set_xlabel("Token Position")
        ax2.set_ylabel("Neuron")
    
    # Add token labels for key positions
    n_labels = min(20, len(text_tokens))
    label_positions = np.linspace(0, len(text_tokens) - 1, n_labels).astype(int)
    
    for pos in label_positions:
        if pos < len(text_tokens):
            ax2.axvline(x=pos, color='white', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved sparsity heatmap: {output_path}")
    return output_path


def plot_activation_atlas(
    concept_activations: Dict[str, np.ndarray],
    output_path: str = "output/activation_atlas.svg",
    figsize: Tuple[int, int] = (14, 10),
) -> str:
    """
    Generate activation atlas with neurons colored by semantic role.
    
    From 
    "We will generate 'Activation Atlases'—visual maps where neurons
    are colored by their semantic role."
    
    Args:
        concept_activations: {concept_name: mean_activation_per_neuron}
        output_path: Path to save SVG
        figsize: Figure size
        
    Returns:
        Path to saved visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for atlas plotting")
    
    concepts = list(concept_activations.keys())
    n_concepts = len(concepts)
    
    if n_concepts == 0:
        print("No concept activations provided")
        return output_path
    
    # Stack activations: (n_concepts, n_neurons)
    activations = np.stack([concept_activations[c] for c in concepts])
    n_neurons = activations.shape[1]
    
    # Find dominant concept per neuron
    dominant_concept = np.argmax(activations, axis=0)
    
    # Selectivity score (how specialized each neuron is)
    selectivity = activations.max(axis=0) / (activations.sum(axis=0) + 1e-8)
    
    # Create 2D layout using t-SNE-like random placement
    np.random.seed(42)
    positions = np.random.randn(n_neurons, 2) * 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color map for concepts
    cmap = plt.cm.get_cmap("Set1", n_concepts)
    
    # Plot neurons
    for i, concept in enumerate(concepts):
        mask = dominant_concept == i
        if mask.sum() == 0:
            continue
        
        sizes = selectivity[mask] * 100 + 10
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            c=[cmap(i)],
            s=sizes,
            alpha=0.7,
            label=f"{concept} ({mask.sum()} neurons)",
        )
    
    ax.set_title("Activation Atlas: Neuron Semantic Specialization", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"Saved activation atlas: {output_path}")
    return output_path


def plot_sps_curve(
    snapshots: List[np.ndarray],
    hurst_exponent: float,
    output_path: str = "output/sps_curve.png",
    figsize: Tuple[int, int] = (12, 6),
) -> str:
    """
    Plot Synaptic Persistence Score autocorrelation curve.
    
    Shows how synaptic state memory decays over time.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for SPS plotting")
    
    if len(snapshots) < 2:
        print("Warning: Need at least 2 snapshots for SPS curve")
        return output_path
    
    # Compute autocorrelation
    max_lag = min(len(snapshots) - 1, 50)
    lags = np.arange(1, max_lag + 1)
    autocorr = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        corr_sum = 0
        count = 0
        for t in range(len(snapshots) - lag):
            # Cosine similarity
            s1 = snapshots[t]
            s2 = snapshots[t + lag]
            norm1 = np.linalg.norm(s1)
            norm2 = np.linalg.norm(s2)
            if norm1 > 0 and norm2 > 0:
                corr_sum += np.dot(s1, s2) / (norm1 * norm2)
                count += 1
        autocorr[i] = corr_sum / count if count > 0 else 0
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Autocorrelation decay
    ax1.plot(lags, autocorr, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel("Lag (snapshots)", fontsize=12)
    ax1.set_ylabel("Autocorrelation", fontsize=12)
    ax1.set_title(f"Synaptic Persistence (H={hurst_exponent:.3f})", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Right: Log-log plot for Hurst estimation
    ax2.loglog(lags, autocorr, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel("Lag (log scale)", fontsize=12)
    ax2.set_ylabel("Autocorrelation (log scale)", fontsize=12)
    ax2.set_title("Long-Range Dependence", fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add interpretation text
    interpretation = "H > 0.5: Long memory" if hurst_exponent > 0.5 else "H < 0.5: Short memory"
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved SPS curve: {output_path}")
    return output_path


def plot_sec_scatter(
    sparsity_values: List[float],
    perplexity_values: List[float],
    correlation: float,
    p_value: float,
    output_path: str = "output/sec_scatter.png",
    figsize: Tuple[int, int] = (10, 8),
) -> str:
    """
    Plot Sparsity-Entropy Correlation scatter plot.
    
    Shows relationship between activation density and perplexity.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for SEC plotting")
    
    if len(sparsity_values) != len(perplexity_values):
        print("Warning: Sparsity and perplexity arrays have different lengths")
        return output_path
    
    if len(sparsity_values) == 0:
        print("Warning: No data for SEC scatter plot")
        return output_path
    
    sparsity = np.array(sparsity_values)
    perplexity = np.array(perplexity_values)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with color gradient by time
    scatter = ax.scatter(
        sparsity, 
        perplexity,
        c=np.arange(len(sparsity)),
        cmap='viridis',
        alpha=0.6,
        s=30,
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time Step', fontsize=11)
    
    # Trend line
    if len(sparsity) > 1:
        z = np.polyfit(sparsity, perplexity, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sparsity.min(), sparsity.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    ax.set_xlabel("Activation Density (fraction non-zero)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title(
        f"Sparsity-Entropy Correlation\n"
        f"r={correlation:.3f}, p={p_value:.4f}",
        fontsize=14
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add interpretation
    if p_value < 0.05:
        sig_text = "Significant" if correlation > 0 else "Significant (negative)"
        color = 'green' if correlation > 0 else 'orange'
    else:
        sig_text = "Not significant"
        color = 'gray'
    
    ax.text(
        0.02, 0.98, sig_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.3)
    )
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved SEC scatter: {output_path}")
    return output_path



def plot_metrics_dashboard(
    metrics_history: Dict[str, List[float]],
    output_path: str = "output/metrics_dashboard.png",
    figsize: Tuple[int, int] = (16, 12),
) -> str:
    """
    Generate comprehensive metrics dashboard.
    
    Combines all frontier metrics into a single visualization
    for competition submission.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for dashboard")
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # TMI over training
    if "tmi" in metrics_history:
        ax = axes[0, 0]
        ax.plot(metrics_history["tmi"], 'b-', linewidth=2)
        ax.set_title("Topological Modularity Index (TMI)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Modularity Q")
        ax.grid(True, alpha=0.3)
    
    # Sparsity over training
    if "sparsity" in metrics_history:
        ax = axes[0, 1]
        ax.plot(metrics_history["sparsity"], 'g-', linewidth=2)
        ax.set_title("Activation Sparsity")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Density (fraction non-zero)")
        ax.grid(True, alpha=0.3)
    
    # SEC correlation
    if "sec_correlation" in metrics_history:
        ax = axes[1, 0]
        ax.plot(metrics_history["sec_correlation"], 'r-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title("Sparsity-Entropy Correlation (SEC)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Pearson r")
        ax.grid(True, alpha=0.3)
    
    # Loss over training
    if "loss" in metrics_history:
        ax = axes[1, 1]
        ax.plot(metrics_history["loss"], 'k-', linewidth=2)
        ax.set_title("Training Loss")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("BDH Training Metrics Dashboard", fontsize=16)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved metrics dashboard: {output_path}")
    return output_path


def export_topology_gexf(
    model: nn.Module,
    output_path: str,
    top_k_percent: float = 5.0,
    max_nodes: int = 1000,  # Limit for Kaggle memory
) -> str:
    """
    Export effective adjacency matrix as .gexf file.
    
    For analysis in Gephi or NetworkX visualization.
    Memory-optimized: only stores non-zero edges.
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required for GEXF export")
    
    tmi_result = compute_tmi(model, top_k_percent)
    adj = tmi_result.adjacency_matrix
    
    # Create SPARSE graph - only add non-zero edges
    G = nx.Graph()
    
    # Find non-zero edges efficiently
    rows, cols = np.where(adj > 0)
    
    # Limit to max_nodes to prevent memory issues
    unique_nodes = np.unique(np.concatenate([rows, cols]))
    if len(unique_nodes) > max_nodes:
        node_degrees = np.bincount(np.concatenate([rows, cols]), minlength=adj.shape[0])
        top_nodes = set(np.argsort(node_degrees)[-max_nodes:])
        mask = np.array([r in top_nodes and c in top_nodes for r, c in zip(rows, cols)])
        rows, cols = rows[mask], cols[mask]
    
    # Add edges (only upper triangle to avoid duplicates)
    edges = [(int(r), int(c), {"weight": float(adj[r, c])}) 
             for r, c in zip(rows, cols) if r < c]
    G.add_edges_from(edges)
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nx.write_gexf(G, output_path)
    
    print(f"Exported topology ({len(G.nodes())} nodes, {len(G.edges())} edges) to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Quick self-test with synthetic data
    print("Testing visualization module...")
    
    # Test sparsity heatmap
    activations = np.random.rand(100, 256) * (np.random.rand(100, 256) > 0.95)
    tokens = [f"token_{i}" for i in range(100)]
    
    try:
        plot_sparsity_heatmap(activations, tokens, "output/test_heatmap.png")
    except ImportError as e:
        print(f"Skipped heatmap test: {e}")
    
    # Test metrics dashboard
    metrics = {
        "tmi": np.cumsum(np.random.randn(100) * 0.01 + 0.01).tolist(),
        "sparsity": (0.05 + np.cumsum(np.random.randn(100) * 0.001)).tolist(),
        "sec_correlation": np.cumsum(np.random.randn(100) * 0.02).tolist(),
        "loss": (5.0 * np.exp(-np.arange(100) * 0.02) + np.random.rand(100) * 0.1).tolist(),
    }
    
    try:
        plot_metrics_dashboard(metrics, "output/test_dashboard.png")
    except ImportError as e:
        print(f"Skipped dashboard test: {e}")
    
    print("Visualization module ready!")
