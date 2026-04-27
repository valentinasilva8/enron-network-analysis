from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graphs(edges_df: pd.DataFrame) -> tuple[nx.DiGraph, nx.Graph]:
    """Build directed and undirected weighted graphs from edge list."""
    directed = nx.from_pandas_edgelist(
        edges_df,
        source="sender",
        target="receiver",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    undirected = directed.to_undirected()
    return directed, undirected


def compute_node_features(directed: nx.DiGraph, undirected: nx.Graph) -> pd.DataFrame:
    """Compute centrality features required by the project plan."""
    if directed.number_of_nodes() == 0:
        return pd.DataFrame(
            columns=[
                "node",
                "in_degree",
                "out_degree",
                "weighted_degree",
                "betweenness_centrality",
                "pagerank",
                "clustering_coefficient",
                "closeness_centrality",
            ]
        )

    in_degree = dict(directed.in_degree())
    out_degree = dict(directed.out_degree())
    weighted_degree = dict(undirected.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(undirected, weight="weight", normalized=True)
    pagerank = nx.pagerank(directed, weight="weight")
    clustering = nx.clustering(undirected, weight="weight")
    closeness = nx.closeness_centrality(undirected)

    rows = []
    for node in directed.nodes():
        rows.append(
            {
                "node": node,
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "clustering_coefficient": clustering.get(node, 0.0),
                "closeness_centrality": closeness.get(node, 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)

