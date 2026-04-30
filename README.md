# Enron Email Network Analysis — CSCI 3461/6683 Data Mining, Spring 2026

This project analyzes the Enron Email Dataset as a communication network to investigate whether email communication structure reflects organizational hierarchy.  We parse ~127K sent-folder emails from 150 mailboxes into a weighted directed graph, apply **Louvain community detection** to identify natural clusters, compute centrality metrics (PageRank, betweenness, degree) to rank influence, and train an **MLP neural network** alongside a Random Forest baseline to classify "high-influence" nodes from structural features alone.

## Setup

```bash
cd project
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# Full pipeline (parse → preprocess → model → visualize)
python3 main.py --phase all

# Neural network extension only (requires nodes.csv from a prior run)
python3 main.py --phase phase4b_nn

# Individual phases: phase1, phase2, phase3, phase4, phase4b_nn, phase5
python3 main.py --phase phase3
```

## Data

The raw Enron maildir dataset (~5 GB, 500K+ files) is **not included** in this repository.

Download it from <https://www.cs.cmu.edu/~./enron/> and extract so that user directories (e.g. `allen-p/`, `lay-k/`) sit at the repository root, or pass a custom path:

```bash
python3 main.py --maildir-root /path/to/your/maildir
```

## Output Files

After a full run, `project/output/` contains:

| File | Description |
|---|---|
| `edges.csv` | Weighted edge list (sender, receiver, weight, dates) |
| `nodes.csv` | Node features + community labels |
| `communities.csv` | Louvain community assignments |
| `louvain_summary.json` | Modularity, community count, stability |
| `neural_network_metrics.json` | MLP and RF accuracy, precision, recall, F1 |
| `validation_report_full.txt` | Preprocessing validation + sanity checks |
| `figures/*.png` | All visualizations (8 core + 1 NN confusion matrix) |

## Project Structure

```
project/
  data_loading.py          # Parse RFC822 emails from sent folders
  preprocessing.py         # Normalize, deduplicate, filter, build edge list
  network_building.py      # NetworkX graph construction + centrality
  modeling.py              # Louvain community detection (+ optional K-Means)
  neural_network_model.py  # MLP + Random Forest high-influence classifier
  visualization.py         # 8 core visualizations
  main.py                  # Pipeline orchestrator (phase1–phase5 + phase4b_nn)
  requirements.txt

docs/                      # Written report (Markdown)
slides/                    # Presentation (.pptx + outline)
notebooks/                 # Summary Jupyter notebook
```

## References

- Klimt, B., & Yang, Y. (2004). Introducing the Enron corpus. In *CEAS 2004*.
- Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.
- Hagberg, A., Swart, P., & S Chult, D. (2008). Exploring network structure, dynamics, and function using NetworkX. In *Proceedings of the 7th Python in Science Conference*.
