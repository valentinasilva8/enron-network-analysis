# Appendix — Enron Network Analysis

## A1) Execution Commands

Run full pipeline:

```bash
cd /Users/valentinasilva/maildir/project
python3 -m pip install -r requirements.txt
python3 main.py --phase all
```

Run optional K-Means:

```bash
python3 main.py --phase all --run-kmeans
```

## A2) Output Artifacts

- `project/output/validation_report_phase1_sample.txt`
- `project/output/validation_report_full.txt`
- `project/output/edges.csv`
- `project/output/nodes.csv`
- `project/output/communities.csv`
- `project/output/louvain_summary.json`
- `project/output/figures/*.png`

## A3) Screenshot Placeholders (Required by Rubric)

Add terminal screenshots in this section before final submission:

1. Screenshot of dependency installation command and success output
2. Screenshot of `python3 main.py --phase phase1` output
3. Screenshot of `python3 main.py --phase all` output
4. Screenshot of generated file list in `project/output/`

## A4) Code Documentation Requirement

All modules include:

- Function docstrings
- Focused comments for non-trivial logic blocks
- Explicit preprocessing rationale in code paths (broadcast drop, edge thresholding, identity policy)

## A5) Additional Notes

- The notebook summary is available at `notebooks/enron_network_analysis_summary.ipynb`.
- The slide deck source content is available in `slides/enron_presentation.md`.

