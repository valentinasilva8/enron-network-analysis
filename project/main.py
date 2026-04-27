from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from modeling import run_louvain, run_optional_kmeans
from network_building import build_graphs, compute_node_features
from preprocessing import run_preprocessing, validation_report_text
from neural_network_model import run_neural_network_classification
from visualization import (
    plot_bridge_nodes,
    plot_community_subgraph,
    plot_network_community_graph,
    plot_post_degree_distribution,
    plot_post_edge_weight_distribution,
    plot_raw_email_volume,
    plot_raw_temporal_trend,
    plot_top_central_nodes,
)


SAMPLE_USERS = {"lay-k", "benson-r", "sanders-r"}


def _save_phase_outputs(output_dir: Path, artifacts, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.email_level_df.to_parquet(output_dir / f"email_level_{suffix}.parquet", index=False)
    artifacts.edges_df.to_csv(output_dir / f"edges_{suffix}.csv", index=False)
    (output_dir / f"validation_{suffix}.json").write_text(
        json.dumps(
            {
                "validation": {k: str(v) for k, v in artifacts.validation.items()},
                "sanity": artifacts.sanity,
            },
            indent=2,
        )
    )
    report = validation_report_text(artifacts.validation, artifacts.sanity)
    (output_dir / f"validation_report_{suffix}.txt").write_text(report)
    print(report)


def run_phase_1(maildir_root: Path, output_dir: Path):
    """Parse ~1K sent emails and validate parser behavior."""
    artifacts = run_preprocessing(maildir_root, max_emails=1000, allowed_mailbox_users=SAMPLE_USERS)
    _save_phase_outputs(output_dir, artifacts, "phase1_sample")
    return artifacts


def run_phase_2(phase1_artifacts, output_dir: Path):
    """Build a small graph and verify centrality + Louvain execution."""
    directed, undirected = build_graphs(phase1_artifacts.edges_df)
    node_features = compute_node_features(directed, undirected)
    community_df, louvain_summary = run_louvain(undirected)

    node_features.to_csv(output_dir / "nodes_phase2_sample.csv", index=False)
    community_df.to_csv(output_dir / "communities_phase2_sample.csv", index=False)
    (output_dir / "louvain_phase2_summary.json").write_text(json.dumps(louvain_summary, indent=2))
    print("=== PHASE 2 SMALL GRAPH SUMMARY ===")
    print(json.dumps(louvain_summary, indent=2))
    return directed, undirected, node_features, community_df, louvain_summary


def run_phase_3(maildir_root: Path, output_dir: Path):
    """Scale preprocessing to the full dataset and export graph-ready edges."""
    artifacts = run_preprocessing(maildir_root, max_emails=None, allowed_mailbox_users=None)
    _save_phase_outputs(output_dir, artifacts, "full")
    artifacts.edges_df.to_csv(output_dir / "edges.csv", index=False)
    return artifacts


def run_phase_4(full_artifacts, output_dir: Path, run_kmeans: bool):
    """Run Louvain and centrality modeling on full graph."""
    directed, undirected = build_graphs(full_artifacts.edges_df)
    node_features = compute_node_features(directed, undirected)
    community_df, louvain_summary = run_louvain(undirected)

    nodes_with_community = node_features.merge(community_df, on="node", how="left")
    nodes_with_community.to_csv(output_dir / "nodes.csv", index=False)
    community_df.to_csv(output_dir / "communities.csv", index=False)
    (output_dir / "louvain_summary.json").write_text(json.dumps(louvain_summary, indent=2))
    print("=== PHASE 4 FULL GRAPH LOUVAIN SUMMARY ===")
    print(json.dumps(louvain_summary, indent=2))

    kmeans_df = pd.DataFrame()
    kmeans_summary = {}
    if run_kmeans:
        kmeans_df, kmeans_summary = run_optional_kmeans(node_features)
        kmeans_df.to_csv(output_dir / "kmeans_clusters.csv", index=False)
        (output_dir / "kmeans_summary.json").write_text(json.dumps(kmeans_summary, indent=2))
        print("=== OPTIONAL K-MEANS SUMMARY ===")
        print(json.dumps(kmeans_summary, indent=2))

    return directed, undirected, node_features, community_df, louvain_summary, kmeans_df, kmeans_summary


def run_phase_5(email_level_df: pd.DataFrame, edges_df: pd.DataFrame, undirected, node_features, community_df, figures_dir: Path):
    """Generate core rubric-aligned visualizations."""
    plot_raw_email_volume(email_level_df, figures_dir)
    plot_raw_temporal_trend(email_level_df, figures_dir)
    plot_post_degree_distribution(undirected, figures_dir)
    plot_post_edge_weight_distribution(edges_df, figures_dir)
    plot_network_community_graph(undirected, node_features, community_df, figures_dir)
    plot_top_central_nodes(node_features, figures_dir)
    plot_community_subgraph(undirected, node_features, community_df, figures_dir)
    plot_bridge_nodes(undirected, node_features, figures_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enron Email Network Analysis pipeline")
    parser.add_argument("--maildir-root", type=Path, default=Path("/Users/valentinasilva/maildir"))
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/valentinasilva/maildir/project/output"))
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2", "phase3", "phase4", "phase4b_nn", "phase5", "all"],
        default="all",
        help="Which pipeline phase to run",
    )
    parser.add_argument("--run-kmeans", action="store_true", help="Run optional K-Means clustering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    phase1_artifacts = None
    full_artifacts = None
    model_bundle = None

    if args.phase in {"phase1", "all"}:
        phase1_artifacts = run_phase_1(args.maildir_root, args.output_dir)

    if args.phase in {"phase2", "all"}:
        if phase1_artifacts is None:
            phase1_artifacts = run_phase_1(args.maildir_root, args.output_dir)
        run_phase_2(phase1_artifacts, args.output_dir)

    if args.phase in {"phase3", "all"}:
        full_artifacts = run_phase_3(args.maildir_root, args.output_dir)

    if args.phase in {"phase4", "all"}:
        if full_artifacts is None:
            full_artifacts = run_phase_3(args.maildir_root, args.output_dir)
        model_bundle = run_phase_4(full_artifacts, args.output_dir, run_kmeans=args.run_kmeans)

    if args.phase in {"phase4b_nn", "all"}:
        nodes_csv = args.output_dir / "nodes.csv"
        if not nodes_csv.exists():
            if full_artifacts is None:
                full_artifacts = run_phase_3(args.maildir_root, args.output_dir)
            if model_bundle is None:
                model_bundle = run_phase_4(full_artifacts, args.output_dir, run_kmeans=args.run_kmeans)
        run_neural_network_classification(nodes_csv, args.output_dir, figures_dir)

    if args.phase in {"phase5", "all"}:
        if full_artifacts is None:
            full_artifacts = run_phase_3(args.maildir_root, args.output_dir)
        if model_bundle is None:
            model_bundle = run_phase_4(full_artifacts, args.output_dir, run_kmeans=args.run_kmeans)
        _, undirected, node_features, community_df, _, _, _ = model_bundle
        run_phase_5(full_artifacts.email_level_df, full_artifacts.edges_df, undirected, node_features, community_df, figures_dir)
        print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()

