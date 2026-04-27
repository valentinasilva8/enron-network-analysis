from __future__ import annotations

from collections import Counter

import networkx as nx
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_louvain(undirected_graph: nx.Graph, resolution: float = 1.0, seeds: list[int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Run Louvain repeatedly and summarize partition stability."""
    if seeds is None:
        seeds = [7, 21, 42]

    if undirected_graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "community"]), {
            "modularity": 0.0,
            "num_communities": 0,
            "largest_community_size": 0,
            "smallest_community_size": 0,
            "median_community_size": 0,
            "stability_majority_agreement_pct": 0.0,
        }

    # Imported lazily to keep module import resilient if optional dependency is absent.
    import community as community_louvain  # type: ignore

    partitions = []
    modularities = []
    for seed in seeds:
        part = community_louvain.best_partition(undirected_graph, weight="weight", resolution=resolution, random_state=seed)
        partitions.append(part)
        modularities.append(community_louvain.modularity(part, undirected_graph, weight="weight"))

    primary_partition = partitions[0]
    community_sizes = Counter(primary_partition.values())
    sizes = sorted(community_sizes.values())

    # Stability proxy: node-wise majority assignment agreement across runs.
    agreement_hits = 0
    for node in undirected_graph.nodes():
        labels = [part[node] for part in partitions]
        majority_count = Counter(labels).most_common(1)[0][1]
        if majority_count >= 2:
            agreement_hits += 1
    stability_pct = agreement_hits / undirected_graph.number_of_nodes() * 100.0

    community_df = pd.DataFrame(
        [{"node": node, "community": comm} for node, comm in primary_partition.items()]
    ).sort_values("community")

    summary = {
        "modularity": float(modularities[0]),
        "num_communities": len(community_sizes),
        "largest_community_size": max(sizes) if sizes else 0,
        "smallest_community_size": min(sizes) if sizes else 0,
        "median_community_size": sizes[len(sizes) // 2] if sizes else 0,
        "stability_majority_agreement_pct": stability_pct,
    }
    return community_df, summary


def run_optional_kmeans(node_features_df: pd.DataFrame, k_values: range = range(2, 7)) -> tuple[pd.DataFrame, dict]:
    """Optional role-archetype clustering on engineered features."""
    if node_features_df.empty:
        return pd.DataFrame(columns=["node", "kmeans_cluster"]), {"best_k": None, "best_silhouette": None}

    feature_cols = [
        "in_degree",
        "out_degree",
        "weighted_degree",
        "betweenness_centrality",
        "pagerank",
        "clustering_coefficient",
        "closeness_centrality",
    ]
    model_df = node_features_df[["node"] + feature_cols].copy()
    x = model_df[feature_cols].fillna(0.0)
    x_scaled = StandardScaler().fit_transform(x)

    best_k = None
    best_score = -1.0
    best_labels = None

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    result_df = pd.DataFrame({"node": model_df["node"], "kmeans_cluster": best_labels})
    return result_df, {"best_k": best_k, "best_silhouette": float(best_score)}

