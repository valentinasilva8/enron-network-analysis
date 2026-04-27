from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save_fig(output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_raw_email_volume(email_level_df: pd.DataFrame, output_dir: Path) -> None:
    """Raw visualization: number of parsed emails per mailbox user."""
    counts = email_level_df["mailbox_user"].value_counts()
    plt.figure(figsize=(8, 5))
    sns.histplot(counts, bins=30)
    plt.title("Raw Data: Email Volume per Mailbox User")
    plt.xlabel("Emails parsed per mailbox user")
    plt.ylabel("Count of users")
    _save_fig(output_dir, "raw_email_volume_per_user.png")


def plot_raw_temporal_trend(email_level_df: pd.DataFrame, output_dir: Path) -> None:
    """Raw visualization: monthly email trend before graph filtering."""
    month_counts = (
        email_level_df.dropna(subset=["date"])
        .assign(month=lambda d: d["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .size()
        .reset_index(name="count")
    )
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=month_counts, x="month", y="count", marker="o")
    plt.title("Raw Data: Emails per Month")
    plt.xlabel("Month")
    plt.ylabel("Email count")
    plt.xticks(rotation=60)
    _save_fig(output_dir, "raw_emails_per_month.png")


def plot_post_degree_distribution(undirected_graph: nx.Graph, output_dir: Path) -> None:
    """Post-preprocessing visualization: degree distribution in log-log space."""
    degrees = [d for _, d in undirected_graph.degree()]
    degree_counts = pd.Series(degrees).value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.scatter(degree_counts.index, degree_counts.values, alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Post-Preprocessing: Degree Distribution (Log-Log)")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    _save_fig(output_dir, "post_degree_distribution_loglog.png")


def plot_preprocessing_funnel(output_dir: Path) -> None:
    """Post-preprocessing visualization: data attrition funnel across pipeline stages."""
    import matplotlib.ticker as mticker

    stages = [
        ("Raw email files scanned",       126590, ""),
        ("Successfully parsed",            121726, "−3.8% parse failures dropped"),
        ("After broadcast drop (≤10)",    117154, "−3.6% broadcast noise removed"),
        ("Internal @enron.com pairs",      167281, "↑ edge expansion (1 email → multiple pairs)"),
        ("After aggregation",              19193,  "−88.5% one-off pairs collapsed"),
        ("Edges retained (weight ≥ 2)",   13627,  "−29.0% weak ties removed (weight < 2)"),
        ("Final graph nodes",              4996,   "final analysis network"),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    notes  = [s[2] for s in stages]
    max_val = max(values)

    palette = sns.color_palette("Blues_d", len(stages))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels[::-1], values[::-1], color=palette, height=0.6)

    for i, (val, note) in enumerate(zip(reversed(values), reversed(notes))):
        ann = f"  {val:,}   {note}" if note else f"  {val:,}"
        ax.text(val + max_val * 0.005, i, ann,
                va="center", ha="left", fontsize=8.5, color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)", fontsize=11)
    ax.set_title("Preprocessing Pipeline — Data Attrition at Each Stage",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(right=max_val * 8)
    ax.tick_params(axis="y", labelsize=10)
    _save_fig(output_dir, "result_preprocessing_funnel.png")


def plot_post_edge_weight_distribution(edges_df: pd.DataFrame, output_dir: Path) -> None:
    """Post-preprocessing visualization: distribution of aggregated edge weights."""
    plt.figure(figsize=(7, 5))
    sns.histplot(edges_df["weight"], bins=40)
    plt.title("Post-Preprocessing: Edge Weight Distribution")
    plt.xlabel("Edge weight (email count)")
    plt.ylabel("Edge frequency")
    _save_fig(output_dir, "post_edge_weight_distribution.png")


def _filtered_subgraph_for_display(
    undirected_graph: nx.Graph,
    node_features_df: pd.DataFrame,
) -> nx.Graph:
    """Filter nodes using top 5% degree OR top 100 PageRank for readability."""
    if undirected_graph.number_of_nodes() == 0:
        return undirected_graph.copy()

    degrees = dict(undirected_graph.degree())
    degree_values = np.array(list(degrees.values()))
    threshold = np.percentile(degree_values, 95)
    nodes_by_degree = {n for n, d in degrees.items() if d >= threshold}

    top_pagerank_nodes = set(node_features_df.sort_values("pagerank", ascending=False).head(100)["node"].tolist())
    selected = nodes_by_degree | top_pagerank_nodes
    if len(selected) < 30:
        selected = set(node_features_df.sort_values("pagerank", ascending=False).head(100)["node"].tolist())
    return undirected_graph.subgraph(selected).copy()


def plot_network_community_graph(
    undirected_graph: nx.Graph,
    node_features_df: pd.DataFrame,
    community_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Result visualization: filtered network graph colored by Louvain community."""
    subgraph = _filtered_subgraph_for_display(undirected_graph, node_features_df)
    if subgraph.number_of_nodes() == 0:
        return

    community_map = dict(zip(community_df["node"], community_df["community"]))
    pagerank_map = dict(zip(node_features_df["node"], node_features_df["pagerank"]))
    node_sizes = [3000 * pagerank_map.get(n, 0.001) + 20 for n in subgraph.nodes()]
    node_colors = [community_map.get(n, -1) for n in subgraph.nodes()]

    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(subgraph, seed=42, weight="weight", k=0.4)
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, cmap="tab20", alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.25)
    plt.title("Result: Filtered Enron Network (Color = Community, Size = PageRank)")
    plt.axis("off")
    _save_fig(output_dir, "result_network_community_graph.png")


def plot_top_central_nodes(node_features_df: pd.DataFrame, output_dir: Path) -> None:
    """Result visualization: top nodes by PageRank and betweenness."""
    top_pr = node_features_df.nlargest(15, "pagerank")[["node", "pagerank"]].copy()
    top_bw = node_features_df.nlargest(15, "betweenness_centrality")[["node", "betweenness_centrality"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=top_pr, x="pagerank", y="node", ax=axes[0], color="#4c72b0")
    axes[0].set_title("Top 15 by PageRank")
    axes[0].set_xlabel("PageRank")
    axes[0].set_ylabel("")

    sns.barplot(data=top_bw, x="betweenness_centrality", y="node", ax=axes[1], color="#dd8452")
    axes[1].set_title("Top 15 by Betweenness")
    axes[1].set_xlabel("Betweenness")
    axes[1].set_ylabel("")

    _save_fig(output_dir, "result_top_central_nodes.png")


def plot_bridge_nodes(
    undirected_graph: nx.Graph,
    node_features_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Result visualization: highlight top-5% betweenness bridge nodes."""
    if undirected_graph.number_of_nodes() == 0:
        return

    subgraph = _filtered_subgraph_for_display(undirected_graph, node_features_df)
    if subgraph.number_of_nodes() == 0:
        return

    bw_series = node_features_df.set_index("node")["betweenness_centrality"]
    threshold = bw_series.quantile(0.95)
    bridge_nodes = {n for n, v in bw_series.items() if v >= threshold}

    pos = nx.spring_layout(subgraph, seed=42, weight="weight", k=0.4)
    plt.figure(figsize=(11, 8))
    non_bridges = [n for n in subgraph.nodes() if n not in bridge_nodes]
    bridges = [n for n in subgraph.nodes() if n in bridge_nodes]

    nx.draw_networkx_edges(subgraph, pos, alpha=0.2)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=non_bridges, node_size=50, node_color="#b0b0b0", alpha=0.6)
    nx.draw_networkx_nodes(subgraph, pos, nodelist=bridges, node_size=180, node_color="#d62728", alpha=0.9)
    plt.title("Result: Bridge Nodes (Top 5% Betweenness)")
    plt.axis("off")
    _save_fig(output_dir, "result_bridge_nodes.png")


def plot_community_subgraph(
    undirected_graph: nx.Graph,
    node_features_df: pd.DataFrame,
    community_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Result visualization: zoom on the largest communities."""
    if community_df.empty:
        return
    largest_communities = (
        community_df["community"].value_counts().head(3).index.tolist()
    )
    selected_nodes = set(community_df[community_df["community"].isin(largest_communities)]["node"].tolist())
    subgraph = undirected_graph.subgraph(selected_nodes).copy()
    if subgraph.number_of_nodes() == 0:
        return

    community_map = dict(zip(community_df["node"], community_df["community"]))
    pagerank_map = dict(zip(node_features_df["node"], node_features_df["pagerank"]))
    node_sizes = [2500 * pagerank_map.get(n, 0.001) + 20 for n in subgraph.nodes()]
    node_colors = [community_map.get(n, -1) for n in subgraph.nodes()]

    plt.figure(figsize=(11, 8))
    pos = nx.spring_layout(subgraph, seed=42, weight="weight", k=0.45)
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, cmap="tab20", alpha=0.9)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.25)
    plt.title("Result: Largest Louvain Communities (Zoom)")
    plt.axis("off")
    _save_fig(output_dir, "result_community_zoom.png")

