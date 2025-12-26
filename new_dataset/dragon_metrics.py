"""
Dragon Metrics Library for BDH Competition

Implements the four Frontier Metrics from 
1. TMI: Topological Modularity Index (Section 4.1)
2. SPS: Synaptic Persistence Score (Section 4.2)
3. SEC: Sparsity-Entropy Correlation (Section 4.3)
4. MCL: Monosemanticity Concept Localization (Section 4.4)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    import networkx as nx
    from community import community_louvain
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    from scipy import stats
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class TMIResult:
    """Result of Topological Modularity Index computation."""
    modularity_q: float
    num_communities: int
    community_sizes: List[int]
    adjacency_matrix: np.ndarray


@dataclass
class SPSResult:
    """Result of Synaptic Persistence Score computation."""
    autocorrelation: np.ndarray  # SPS(k) for various lags
    lags: np.ndarray
    hurst_exponent: float  # H > 0.5 indicates long-range dependence


@dataclass
class SECResult:
    """Result of Sparsity-Entropy Correlation computation."""
    correlation: float  # Pearson r
    p_value: float
    sparsity_values: np.ndarray
    perplexity_values: np.ndarray


@dataclass
class MCLResult:
    """Result of Monosemanticity Concept Localization."""
    concept_neurons: Dict[str, List[int]]  # concept -> top neuron indices
    selectivity_scores: Dict[str, Dict[int, float]]  # concept -> {neuron: score}


# =============================================================================
# TMI: Topological Modularity Index (Section 4.1)
# =============================================================================

def compute_tmi(
    model: nn.Module,
    top_k_percent: float = 5.0,
) -> TMIResult:
    """
    Compute Topological Modularity Index.
    
    Verifies the claim that BDH evolves a "scale-free graph of neurons"
    with "high modularity" through training.
    
    Methodology (from ):
    1. Extract W_eff = decoder @ encoder as weighted adjacency matrix
    2. Sparsify to top k% edges by magnitude
    3. Apply Louvain community detection
    4. Compute Newman Modularity Q score
    
    Args:
        model: BDH model instance
        top_k_percent: Percentage of edges to retain
        
    Returns:
        TMIResult with modularity score and community structure
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx and python-louvain required for TMI computation")
    
    # Extract weight matrices
    # In BDH: decoder shape is (nh*N, D), encoder shape is (nh, D, N)
    decoder = model.decoder.detach().cpu().numpy()  # (nh*N, D)
    encoder = model.encoder.detach().cpu().numpy()  # (nh, D, N)
    
    # Reshape encoder to (nh*N, D) for matrix multiplication
    nh, D, N = encoder.shape
    encoder_flat = encoder.transpose(0, 2, 1).reshape(nh * N, D)  # (nh*N, D)
    
    # Compute effective adjacency: W_eff = decoder @ encoder.T
    # This gives (nh*N, nh*N) adjacency matrix
    w_eff = decoder @ encoder_flat.T
    
    # Take absolute values (we care about connection strength, not direction)
    w_eff = np.abs(w_eff)
    
    # Sparsify: keep only top k% edges
    threshold = np.percentile(w_eff.flatten(), 100 - top_k_percent)
    w_sparse = np.where(w_eff >= threshold, w_eff, 0)
    
    # Build NetworkX graph
    G = nx.from_numpy_array(w_sparse)
    
    # Remove isolated nodes for cleaner community detection
    G.remove_nodes_from(list(nx.isolates(G)))
    
    if len(G.nodes()) == 0:
        return TMIResult(
            modularity_q=0.0,
            num_communities=0,
            community_sizes=[],
            adjacency_matrix=w_sparse,
        )
    
    # Apply Louvain community detection
    partition = community_louvain.best_partition(G, weight="weight")
    
    # Compute Newman Modularity Q
    modularity_q = community_louvain.modularity(partition, G, weight="weight")
    
    # Count communities and their sizes
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    community_sizes = [len(nodes) for nodes in communities.values()]
    
    return TMIResult(
        modularity_q=modularity_q,
        num_communities=len(communities),
        community_sizes=sorted(community_sizes, reverse=True),
        adjacency_matrix=w_sparse,
    )


# =============================================================================
# SPS: Synaptic Persistence Score (Section 4.2)
# =============================================================================

class SynapticStateTracker:
    """
    Tracks synaptic state snapshots during inference.
    
    Used to compute the Synaptic Persistence Score (SPS).
    """
    
    def __init__(self, snapshot_interval: int = 100):
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[np.ndarray] = []
        self.token_count = 0
    
    def maybe_snapshot(self, sigma: torch.Tensor) -> None:
        """Save snapshot if interval reached."""
        self.token_count += 1
        if self.token_count % self.snapshot_interval == 0:
            self.snapshots.append(sigma.detach().cpu().numpy().flatten())
    
    def reset(self) -> None:
        """Clear all snapshots."""
        self.snapshots = []
        self.token_count = 0


def compute_sps(
    snapshots: List[np.ndarray],
    max_lag: Optional[int] = None,
) -> SPSResult:
    """
    Compute Synaptic Persistence Score.
    
    Measures the durability of Hebbian memory by computing
    autocorrelation of synaptic states over time.
    
    Methodology (from ):
    1. Take snapshots of σ at regular intervals
    2. Compute cosine similarity between σ_t and σ_{t+k}
    3. Analyze decay curve (power-law vs exponential)
    
    Args:
        snapshots: List of flattened synaptic state vectors
        max_lag: Maximum lag to compute (default: len/4)
        
    Returns:
        SPSResult with autocorrelation curve and Hurst exponent
    """
    if len(snapshots) < 4:
        return SPSResult(
            autocorrelation=np.array([1.0]),
            lags=np.array([0]),
            hurst_exponent=0.5,
        )
    
    n = len(snapshots)
    if max_lag is None:
        max_lag = n // 4
    
    lags = np.arange(1, max_lag + 1)
    autocorr = np.zeros(len(lags))
    
    for i, k in enumerate(lags):
        similarities = []
        for t in range(n - k):
            # Cosine similarity
            sigma_t = snapshots[t]
            sigma_tk = snapshots[t + k]
            
            norm_t = np.linalg.norm(sigma_t)
            norm_tk = np.linalg.norm(sigma_tk)
            
            if norm_t > 0 and norm_tk > 0:
                sim = np.dot(sigma_t, sigma_tk) / (norm_t * norm_tk)
                similarities.append(sim)
        
        if similarities:
            autocorr[i] = np.mean(similarities)
    
    # Estimate Hurst exponent via log-log regression
    # H > 0.5 indicates long-range dependence
    valid_mask = autocorr > 0
    if np.sum(valid_mask) > 2:
        log_lag = np.log(lags[valid_mask])
        log_acf = np.log(autocorr[valid_mask])
        
        # Linear regression: log(ACF) ~ -slope * log(lag)
        slope, _ = np.polyfit(log_lag, log_acf, 1)
        hurst = 1 - (-slope / 2)  # Approximate Hurst from decay slope
        hurst = np.clip(hurst, 0, 1)
    else:
        hurst = 0.5
    
    return SPSResult(
        autocorrelation=autocorr,
        lags=lags,
        hurst_exponent=hurst,
    )


# =============================================================================
# SEC: Sparsity-Entropy Correlation (Section 4.3)
# =============================================================================

class ActivationSparsityTracker:
    """
    Tracks activation sparsity and perplexity during inference.
    
    Used to compute the Sparsity-Entropy Correlation (SEC).
    """
    
    def __init__(self):
        self.sparsity_values: List[float] = []
        self.perplexity_values: List[float] = []
    
    def record(self, activations: torch.Tensor, loss: float) -> None:
        """
        Record activation sparsity and perplexity.
        
        Args:
            activations: Post-ReLU activations from BDH layer
            loss: Cross-entropy loss for the batch
        """
        # Sparsity: fraction of non-zero activations
        density = (activations > 0).float().mean().item()
        self.sparsity_values.append(density)
        
        # Perplexity = exp(loss)
        perplexity = np.exp(loss)
        self.perplexity_values.append(perplexity)
    
    def reset(self) -> None:
        """Clear recorded values."""
        self.sparsity_values = []
        self.perplexity_values = []


def compute_sec(
    sparsity_values: List[float],
    perplexity_values: List[float],
) -> SECResult:
    """
    Compute Sparsity-Entropy Correlation.
    
    Validates the "Reasoning you can see" claim by measuring
    correlation between activation density and input surprisal.
    
    Methodology (from ):
    1. Measure ρ_t = fraction of non-zero activations
    2. Measure perplexity as proxy for surprisal
    3. Compute Pearson correlation
    
    Hypothesis: Positive correlation (r > 0)
    - Low perplexity (predictable) -> sparse activations
    - High perplexity (surprising) -> dense activations
    
    Args:
        sparsity_values: Activation densities over time
        perplexity_values: Model perplexities over time
        
    Returns:
        SECResult with correlation coefficient and p-value
    """
    if len(sparsity_values) < 3:
        return SECResult(
            correlation=0.0,
            p_value=1.0,
            sparsity_values=np.array(sparsity_values),
            perplexity_values=np.array(perplexity_values),
        )
    
    sparsity = np.array(sparsity_values)
    perplexity = np.array(perplexity_values)
    
    # Clip extreme perplexity values
    perplexity = np.clip(perplexity, 0, 1000)
    
    if SCIPY_AVAILABLE:
        r, p_value = stats.pearsonr(sparsity, perplexity)
    else:
        # Manual Pearson correlation
        sparsity_centered = sparsity - np.mean(sparsity)
        perplexity_centered = perplexity - np.mean(perplexity)
        
        numerator = np.sum(sparsity_centered * perplexity_centered)
        denominator = np.sqrt(np.sum(sparsity_centered**2) * np.sum(perplexity_centered**2))
        
        r = numerator / (denominator + 1e-8)
        p_value = 0.0  # Cannot compute without scipy
    
    return SECResult(
        correlation=r,
        p_value=p_value,
        sparsity_values=sparsity,
        perplexity_values=perplexity,
    )


# =============================================================================
# MCL: Monosemanticity Concept Localization (Section 4.4)
# =============================================================================

def compute_mcl(
    activations_by_concept: Dict[str, List[np.ndarray]],
    activations_baseline: List[np.ndarray],
    top_k_neurons: int = 10,
) -> MCLResult:
    """
    Compute Monosemanticity Concept Localization.
    
    Empirically demonstrates that individual neurons encode
    specific financial concepts.
    
    Methodology (from ):
    1. Define concepts: {Inflation, AI, Dividend, Layoffs, ...}
    2. For each concept, identify neurons with max activation
    3. Compute selectivity score: S(n,c) = P(active|c) / P(active|¬c)
    
    Args:
        activations_by_concept: {concept: [activation_vectors]}
        activations_baseline: Activations when concept NOT present
        top_k_neurons: Number of top neurons to return per concept
        
    Returns:
        MCLResult with concept neurons and selectivity scores
    """
    concept_neurons = {}
    selectivity_scores = {}
    
    # Stack baseline activations
    if len(activations_baseline) == 0:
        return MCLResult(concept_neurons={}, selectivity_scores={})
    
    baseline_stack = np.stack(activations_baseline)
    baseline_active_prob = (baseline_stack > 0).mean(axis=0)  # P(active|¬c)
    
    for concept, activations in activations_by_concept.items():
        if len(activations) == 0:
            continue
        
        concept_stack = np.stack(activations)
        concept_active_prob = (concept_stack > 0).mean(axis=0)  # P(active|c)
        
        # Selectivity score: S(n,c) = P(active|c) / P(active|¬c)
        # Add small epsilon to avoid division by zero
        selectivity = concept_active_prob / (baseline_active_prob + 1e-8)
        
        # Find top-k most selective neurons
        top_indices = np.argsort(selectivity)[-top_k_neurons:][::-1]
        
        concept_neurons[concept] = top_indices.tolist()
        selectivity_scores[concept] = {
            int(idx): float(selectivity[idx]) for idx in top_indices
        }
    
    return MCLResult(
        concept_neurons=concept_neurons,
        selectivity_scores=selectivity_scores,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def create_activation_hooks(model: nn.Module) -> Tuple[List[torch.Tensor], List[Any]]:
    """
    Register forward hooks to capture activations.
    
    Uses model.register_forward_hook() as specified in 
    
    Returns:
        Tuple of (activation_list, hook_handles)
    """
    activations = []
    handles = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activations.append(output.detach())
    
    # Hook into BDH's ReLU activations (x_sparse in forward pass)
    # We'll need to modify BDH model to expose these, or hook specific layers
    for name, module in model.named_modules():
        if "relu" in name.lower() or isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(hook_fn)
            handles.append(handle)
    
    return activations, handles


def remove_hooks(handles: List[Any]) -> None:
    """Remove registered hooks."""
    for handle in handles:
        handle.remove()


def export_topology(
    model: nn.Module,
    epoch: int,
    output_dir: str = "output",
    top_k_percent: float = 5.0,
) -> str:
    """
    Export effective adjacency matrix as .gexf file.
    
    For analysis in Gephi or NetworkX.
    
    Returns:
        Path to exported file
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    tmi_result = compute_tmi(model, top_k_percent)
    
    G = nx.from_numpy_array(tmi_result.adjacency_matrix)
    output_path = os.path.join(output_dir, f"topology_epoch_{epoch}.gexf")
    nx.write_gexf(G, output_path)
    
    return output_path


if __name__ == "__main__":
    # Quick self-test
    print("Testing dragon_metrics module...")
    
    # Test SPS with synthetic snapshots
    snapshots = [np.random.randn(100) for _ in range(50)]
    sps_result = compute_sps(snapshots)
    print(f"SPS Hurst exponent: {sps_result.hurst_exponent:.3f}")
    
    # Test SEC with synthetic data
    sparsity = np.linspace(0.05, 0.15, 100) + np.random.randn(100) * 0.01
    perplexity = sparsity * 50 + np.random.randn(100) * 5
    sec_result = compute_sec(sparsity.tolist(), perplexity.tolist())
    print(f"SEC correlation: r={sec_result.correlation:.3f}, p={sec_result.p_value:.4f}")
    
    print("Dragon metrics ready!")
