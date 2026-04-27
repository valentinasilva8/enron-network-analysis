# Enron Email Network Analysis — 15-Minute Slide Deck

## Slide 1 — Title
- To What Extent Does Enron Communication Reflect Hierarchy?
- Team members, course, date

## Slide 2 — Problem Statement
- Research question and hypothesis
- Why this matters (hidden influence vs formal rank)

## Slide 3 — Dataset and Challenges
- 150 mailbox directories, 500K+ messages
- Key risks: duplicates, broadcasts, weak ties, alias noise

## Slide 4 — Pipeline Overview
- Parse -> Validate -> Build Graph -> Louvain -> Centrality -> Story

## Slide 5 — Raw Data Exploration
- Email volume per user histogram
- Monthly trend line

## Slide 6 — Preprocessing Logic
- Sent-only parsing, identity strategy, broadcast removal (>10), edge threshold (>=2)
- Validation report highlights

## Slide 7 — Post-Processing Checks
- Degree distribution (log-log)
- Edge weight distribution

## Slide 8 — Algorithm: Louvain
- Community detection objective and rationale
- Pseudo-code summary

## Slide 9 — Algorithm: Centrality
- PageRank, betweenness, degree
- Bridge threshold (top 5% betweenness)

## Slide 10 — Network Result (Hero Figure)
- Filtered network (size = PageRank, color = community)

## Slide 11 — Community Result
- Largest-community zoom
- Modularity and community count

## Slide 12 — Centrality Result
- Top central nodes bar chart
- Executive vs non-executive pattern

## Slide 13 — Bridge Result
- Bridge nodes visualization
- Divergence finding: structural influence beyond hierarchy

## Slide 14 — Neural Network Extension
- Secondary model: predict "high influence" nodes from structural features only
- MLP (85.3% accuracy) vs Random Forest baseline (91.0%)
- Confirms: network position alone encodes role information
- Confusion matrix visualization

## Slide 15 — Conclusion + Future Work
- Where hierarchy holds
- Where hierarchy diverges
- Neural network supports: structural features predict influence
- Future extensions (temporal, external, content, label-based)

## Slide 16 — References
- Dataset, Louvain paper, NetworkX citation (APA style)
