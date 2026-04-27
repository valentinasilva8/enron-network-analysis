# Enron Email Network Analysis Report

## Front Page (Not Counted in 5 Pages)

**Project Title:** To What Extent Does Enron's Communication Structure Reflect Organizational Hierarchy?  
**Course:** Data Mining (Group Project)  
**Team Members:** [Fill in names]  
**Date:** [Fill in date]  

---

## 1) Problem Statement and Challenge

This project analyzes the Enron Email Dataset as a communication network.  
Core research question:

> To what extent does communication structure in the Enron email network reflect organizational hierarchy, and where does it diverge?

Challenge: naive email-network construction can produce misleading centrality and community structure due to duplicates, broad broadcast emails, and weak one-off ties. The project addresses this with a validation-first preprocessing pipeline.

---

## 2) Hardware and Software Details

- Machine: macOS workstation (Darwin 25.2.0)
- Python: 3.13
- Core libraries:
  - pandas 3.0.1
  - networkx 3.6.1
  - python-louvain 0.16
  - scikit-learn 1.8.0
  - matplotlib 3.10.8
  - seaborn 0.13.2
  - tqdm 4.67.1
  - pyarrow 24.0.0

---

## 3) Dataset Overview and Exploration

The Enron maildir contains 150 mailbox directories and 500K+ raw email files.  
Each message file includes headers such as `From`, `To`, `Cc`, `Bcc`, `Date`, `Message-ID`, plus Enron-specific `X-`* metadata.

### Raw Data Visualizations

1. `project/output/figures/raw_email_volume_per_user.png`
2. `project/output/figures/raw_emails_per_month.png`

These show long-tail communication volume and temporal concentration.

---

## 4) Preprocessing and Validation

### Pipeline

1. Parse only sent folders: `sent`, `sent_items`, `_sent`, `_sent_mail`
2. Extract sender/recipients/date/message-id
3. Normalize email addresses (lowercase, whitespace trim)
4. Identity strategy:
  - Use email address as node identity
  - Remove known system placeholders (`no.address@enron.com`, etc.)
  - Do **not** strip dot patterns (avoids false merges)
5. Drop messages with no recipients
6. Drop messages with >10 recipients (broadcast noise control)
7. Expand multi-recipient emails to pairwise edges
8. Filter to `@enron.com` for internal network focus
9. Aggregate edge weights by sender->receiver count
10. Keep edges with weight >= 2

### Validation and Sanity Checks

Validation output file: `project/output/validation_report_full.txt`

Key metrics from run:

- Total scanned: 126,590
- Successfully parsed: 121,726 (96.16%)
- Dropped: 4,864 (3.84%)
- Avg recipients/email: 1.73
- Edges retained (weight >= 2): 13,627
- Users after filtering: 6,382

Sanity checks include:

- Top 10 senders
- Random edge sample
- Node count + largest connected component + isolated nodes

### Post-Preprocessing Visualizations

1. `project/output/figures/post_degree_distribution_loglog.png`
2. `project/output/figures/post_edge_weight_distribution.png`

---

## 5) Algorithm, Pseudo-Code, and Implementation

### Main Algorithm (Clustering + Network Metrics)

Primary clustering: **Louvain community detection** on weighted undirected graph  
Secondary analysis: centrality ranking (PageRank, betweenness, degree)

```text
ALGORITHM: Enron Communication Network Analysis

INPUT: maildir/ (raw Enron email files)
OUTPUT: communities, centrality rankings, visualizations

1) Parse emails from sent* folders
2) Extract headers (From/To/Cc/Bcc/Date/Message-ID)
3) Normalize identities; remove known system addresses
4) Drop broadcasts (>10 recipients) and no-recipient emails
5) Expand to pairwise edges and aggregate weights
6) Keep edges with weight >= 2
7) Build NetworkX graph
8) Compute centrality metrics (PageRank, betweenness, degree, etc.)
9) Run Louvain community detection
10) Visualize and interpret: hubs, communities, bridges, divergence from hierarchy
```

Implementation modules:

- `project/data_loading.py`
- `project/preprocessing.py`
- `project/network_building.py`
- `project/modeling.py`
- `project/visualization.py`
- `project/main.py`

---

## 6) Results and Story

Generated result visualizations:

- `project/output/figures/result_network_community_graph.png`
- `project/output/figures/result_community_zoom.png`
- `project/output/figures/result_top_central_nodes.png`
- `project/output/figures/result_bridge_nodes.png`

Model summary (full graph):

- Modularity: 0.7422
- Communities: 18
- Largest community: 821 nodes
- Stability (majority agreement across 3 runs): 73.7%

Interpretation:

- The network is strongly structured into communities.
- Highly central individuals are not only top executives.
- Bridge roles (top betweenness) reveal structural influence that may diverge from formal hierarchy.

This supports a nuanced conclusion: hierarchy is reflected partially, but communication power also emerges from cross-team coordination roles.

### Neural Network Extension: Predicting High-Influence Nodes

As a secondary classification component, we trained an MLP neural network and a Random Forest baseline to predict whether a node is "high influence" (top 20% by PageRank OR betweenness centrality).

**Features used** (no label leakage — PageRank and betweenness excluded from inputs):
`in_degree`, `out_degree`, `weighted_degree`, `clustering_coefficient`, `closeness_centrality`, `community_size`

**Results:**


| Model                        | Accuracy | Precision | Recall | F1    |
| ---------------------------- | -------- | --------- | ------ | ----- |
| MLP Classifier (64-32, ReLU) | 85.3%    | 82.5%     | 63.8%  | 72.0% |
| Random Forest Baseline       | 91.0%    | 86.2%     | 83.0%  | 84.6% |


The Random Forest outperforms the MLP, likely due to the relatively small feature set and tabular data structure where tree-based models typically excel. Both models demonstrate that structural features alone can predict high-influence status with reasonable accuracy, supporting the hypothesis that network position encodes meaningful role information.

Confusion matrix: `project/output/figures/result_neural_network_confusion_matrix.png`
Metrics: `project/output/neural_network_metrics.json`

---

## 7) Future Work

1. Temporal community evolution (pre-/post-critical periods)
2. Inclusion of external-domain communication as layered graph
3. Content-aware analysis (topic/sentiment over communities)
4. Better entity resolution with curated alias map
5. Semi-supervised role prediction with validated labels

---

## 8) Appendix and Evidence

Appendix file: `docs/enron_report_appendix.md`

Includes:

- Full code references
- Terminal execution evidence commands
- Validation report extract
- Additional figures if needed

---

## 9) References (APA-Style)

- Klimt, B., & Yang, Y. (2004). Introducing the Enron corpus. In *CEAS 2004*.  
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.  
- Hagberg, A., Swart, P., & S Chult, D. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference*.

