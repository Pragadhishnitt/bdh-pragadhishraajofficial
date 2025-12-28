"""
Visualization Utilities for BDH Competition

Implements visualization requirements 
- Topology plots (SVG of learned graph structure)
- Sparsity heatmaps (activation density over text)
- Activation atlases (neurons colored by semantic role)
- Graph export for Gephi/NetworkX

Publication-quality styling for research reports.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch, FancyBboxPatch
    import matplotlib.patheffects as path_effects
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


# ============================================================================
# PUBLICATION-QUALITY STYLE CONFIGURATION
# ============================================================================

# Professional color palette (colorblind-friendly)
RESEARCH_COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Magenta
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'neutral': '#3B3B3B',      # Dark gray
    'background': '#FAFAFA',   # Light gray
    'grid': '#E0E0E0',         # Grid lines
}

# Community colors (distinct, publication-ready)
COMMUNITY_PALETTE = [
    '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B4C9A',
    '#4ECDC4', '#FF6B6B', '#95E1D3', '#DDA0DD', '#98D8C8',
]


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        
        # Figure settings
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': RESEARCH_COLORS['neutral'],
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': RESEARCH_COLORS['grid'],
        
        # Line settings
        'lines.linewidth': 2,
        'lines.markersize': 6,
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
    })
    
    if SEABORN_AVAILABLE:
        sns.set_style('whitegrid')
        sns.set_context('paper', font_scale=1.2)


def plot_topology(
    model: nn.Module,
    output_path: str = "output/topology.svg",
    top_k_percent: float = 2.0,
    figsize: Tuple[int, int] = (14, 14),
    node_size_scale: float = 80.0,
    max_nodes: int = 300,
    dark_theme: bool = False,
) -> str:
    """
    Generate publication-quality SVG visualization of learned graph structure.
    
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
    
    # Apply publication style
    set_publication_style()
    
    # Create figure with clean background
    print("  Drawing plot...")
    fig, ax = plt.subplots(figsize=figsize)
    
    if dark_theme:
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        edge_color = '#444466'
        title_color = 'white'
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        edge_color = '#cccccc'
        title_color = RESEARCH_COLORS['neutral']
    
    # Use professional color palette for communities
    n_communities = len(communities)
    community_colors = COMMUNITY_PALETTE[:n_communities] if n_communities <= len(COMMUNITY_PALETTE) else plt.cm.tab20.colors[:n_communities]
    node_colors = [community_colors[partition[node] % len(community_colors)] for node in G.nodes()]
    
    # Draw edges with gradient effect
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.15,
        width=0.5,
        edge_color=edge_color,
        style='solid',
    )
    
    # Draw nodes with glow effect
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors='white',
        linewidths=0.5,
    )
    
    # Professional title with metrics
    title = ax.set_title(
        f"BDH Neural Topology",
        fontsize=18,
        fontweight='bold',
        color=title_color,
        pad=20,
    )
    
    # Add subtitle with metrics
    ax.text(
        0.5, 1.02,
        f"Modularity Q = {tmi_result.modularity_q:.3f}  |  "
        f"Communities = {tmi_result.num_communities}  |  "
        f"Nodes = {n_nodes}",
        transform=ax.transAxes,
        ha='center',
        fontsize=11,
        color='gray' if not dark_theme else '#aaaaaa',
    )
    
    # Create professional legend
    top_communities = sorted(
        [(c, sum(1 for n, pc in partition.items() if pc == c)) 
         for c in communities],
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    
    legend_patches = [
        Patch(
            color=community_colors[c % len(community_colors)],
            label=f"Community {c+1} (n={size})",
            alpha=0.9,
        )
        for c, size in top_communities
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        edgecolor='lightgray',
        title="Communities",
        title_fontsize=11,
    )
    legend.get_frame().set_linewidth(0.5)
    
    ax.axis("off")
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Save in multiple formats
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format="svg", bbox_inches="tight", dpi=300)
    
    # Also save PNG for compatibility
    png_path = output_path.replace('.svg', '.png')
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"  Saved topology plot: {output_path}")
    print(f"  Saved PNG version: {png_path}")
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
    figsize: Tuple[int, int] = (14, 6),
) -> str:
    """
    Plot publication-quality Synaptic Persistence Score autocorrelation curve.
    
    Shows how synaptic state memory decays over time.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for SPS plotting")
    
    set_publication_style()
    
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
            s1 = snapshots[t]
            s2 = snapshots[t + lag]
            norm1 = np.linalg.norm(s1)
            norm2 = np.linalg.norm(s2)
            if norm1 > 0 and norm2 > 0:
                corr_sum += np.dot(s1, s2) / (norm1 * norm2)
                count += 1
        autocorr[i] = corr_sum / count if count > 0 else 0
    
    # Create professional two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Left: Autocorrelation decay with gradient fill
    ax1.plot(lags, autocorr, color=RESEARCH_COLORS['primary'], linewidth=2.5, 
             marker='o', markersize=5, markerfacecolor='white', markeredgewidth=2)
    ax1.fill_between(lags, autocorr, alpha=0.2, color=RESEARCH_COLORS['primary'])
    ax1.set_xlabel("Lag (snapshots)", fontsize=12, fontweight='medium')
    ax1.set_ylabel("Autocorrelation", fontsize=12, fontweight='medium')
    ax1.set_title(f"Synaptic Persistence Score", fontsize=14, fontweight='bold', pad=10)
    ax1.axhline(y=0, color=RESEARCH_COLORS['neutral'], linestyle='--', alpha=0.5, linewidth=1)
    
    # Add Hurst exponent annotation
    h_color = RESEARCH_COLORS['success'] if hurst_exponent > 0.5 else RESEARCH_COLORS['secondary']
    ax1.text(0.95, 0.95, f"H = {hurst_exponent:.3f}",
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=14, fontweight='bold', color=h_color,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                      edgecolor=h_color, alpha=0.9, linewidth=2))
    
    # Right: Log-log plot
    valid_mask = autocorr > 0
    if valid_mask.sum() > 1:
        ax2.loglog(lags[valid_mask], autocorr[valid_mask], 
                   color=RESEARCH_COLORS['secondary'], linewidth=2.5,
                   marker='s', markersize=5, markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel("Lag (log scale)", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Autocorrelation (log scale)", fontsize=12, fontweight='medium')
    ax2.set_title("Long-Range Dependence Analysis", fontsize=14, fontweight='bold', pad=10)
    
    # Add interpretation box
    if hurst_exponent > 0.5:
        interpretation = f"H > 0.5: Long-range synaptic memory"
        box_color = '#e8f5e9'
        text_color = RESEARCH_COLORS['success']
    else:
        interpretation = f"H ≤ 0.5: Short-range memory"
        box_color = '#fff3e0'
        text_color = RESEARCH_COLORS['accent']
    
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=12, fontweight='medium',
             color=text_color,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, 
                      edgecolor=text_color, alpha=0.9, linewidth=1.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
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
    Plot publication-quality Sparsity-Entropy Correlation scatter plot.
    
    Shows relationship between activation density and perplexity.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for SEC plotting")
    
    set_publication_style()
    
    if len(sparsity_values) != len(perplexity_values):
        print("Warning: Sparsity and perplexity arrays have different lengths")
        return output_path
    
    if len(sparsity_values) == 0:
        print("Warning: No data for SEC scatter plot")
        return output_path
    
    sparsity = np.array(sparsity_values)
    perplexity = np.array(perplexity_values)
    
    # Create professional figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')
    
    # Scatter plot with professional gradient
    scatter = ax.scatter(
        sparsity, 
        perplexity,
        c=np.arange(len(sparsity)),
        cmap='plasma',
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidths=0.5,
    )
    
    # Professional colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Time Step', fontsize=11, fontweight='medium')
    cbar.ax.tick_params(labelsize=9)
    
    # Trend line with confidence band
    if len(sparsity) > 1:
        z = np.polyfit(sparsity, perplexity, 1)
        p = np.poly1d(z)
        x_line = np.linspace(sparsity.min(), sparsity.max(), 100)
        ax.plot(x_line, p(x_line), color=RESEARCH_COLORS['success'], 
                linewidth=2.5, linestyle='--', alpha=0.8, label='Linear Trend')
    
    ax.set_xlabel("Activation Density (fraction non-zero)", fontsize=12, fontweight='medium')
    ax.set_ylabel("Perplexity", fontsize=12, fontweight='medium')
    ax.set_title("Sparsity-Entropy Correlation (SEC)", fontsize=14, fontweight='bold', pad=15)
    
    # Add correlation statistics box
    stats_text = f"r = {correlation:.3f}\np = {p_value:.2e}"
    if p_value < 0.05:
        box_color = '#e8f5e9' if correlation > 0 else '#fff3e0'
        text_color = RESEARCH_COLORS['primary'] if correlation > 0 else RESEARCH_COLORS['accent']
        sig_label = "Significant"
    else:
        box_color = '#f5f5f5'
        text_color = 'gray'
        sig_label = "Not Significant"
    
    # Statistics annotation box
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color,
                     edgecolor=text_color, alpha=0.95, linewidth=1.5))
    
    # Significance indicator
    ax.text(0.03, 0.97, sig_label,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=11, fontweight='bold', color=text_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=text_color, alpha=0.9, linewidth=1.5))
    
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    
    print(f"Saved SEC scatter: {output_path}")
    return output_path



def plot_metrics_dashboard(
    metrics_history: Dict[str, List[float]],
    output_path: str = "output/metrics_dashboard.png",
    figsize: Tuple[int, int] = (16, 12),
) -> str:
    """
    Generate publication-quality comprehensive metrics dashboard.
    
    Combines all frontier metrics into a single visualization
    for competition submission and research reports.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for dashboard")
    
    set_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    panel_colors = [RESEARCH_COLORS['primary'], RESEARCH_COLORS['secondary'],
                    RESEARCH_COLORS['accent'], RESEARCH_COLORS['neutral']]
    
    # TMI over training
    if "tmi" in metrics_history:
        ax = axes[0, 0]
        ax.set_facecolor('#fafafa')
        data = metrics_history["tmi"]
        ax.plot(data, color=panel_colors[0], linewidth=2.5)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=panel_colors[0])
        ax.set_title("Topological Modularity Index (TMI)", fontweight='bold', pad=10)
        ax.set_xlabel("Training Step", fontweight='medium')
        ax.set_ylabel("Modularity Q", fontweight='medium')
        ax.axhline(y=0.3, color='gray', linestyle=':', alpha=0.7, label='Target Q=0.3')
        ax.legend(loc='lower right', fontsize=9)
    
    # Sparsity over training
    if "sparsity" in metrics_history:
        ax = axes[0, 1]
        ax.set_facecolor('#fafafa')
        data = metrics_history["sparsity"]
        ax.plot(data, color=panel_colors[1], linewidth=2.5)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=panel_colors[1])
        ax.set_title("Activation Sparsity", fontweight='bold', pad=10)
        ax.set_xlabel("Training Step", fontweight='medium')
        ax.set_ylabel("Density (fraction non-zero)", fontweight='medium')
    
    # SEC correlation
    if "sec_correlation" in metrics_history:
        ax = axes[1, 0]
        ax.set_facecolor('#fafafa')
        data = metrics_history["sec_correlation"]
        ax.plot(data, color=panel_colors[2], linewidth=2.5)
        ax.axhline(y=0, color=RESEARCH_COLORS['neutral'], linestyle='--', alpha=0.6, linewidth=1.5)
        ax.set_title("Sparsity-Entropy Correlation (SEC)", fontweight='bold', pad=10)
        ax.set_xlabel("Training Step", fontweight='medium')
        ax.set_ylabel("Pearson r", fontweight='medium')
    
    # Loss over training
    if "loss" in metrics_history:
        ax = axes[1, 1]
        ax.set_facecolor('#fafafa')
        data = metrics_history["loss"]
        ax.plot(data, color=panel_colors[3], linewidth=2.5)
        ax.set_title("Training Loss", fontweight='bold', pad=10)
        ax.set_xlabel("Training Step", fontweight='medium')
        ax.set_ylabel("Cross-Entropy Loss", fontweight='medium')
        ax.set_yscale('log')
    
    # Professional main title
    fig.suptitle("BDH Training Metrics Dashboard", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white')
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
