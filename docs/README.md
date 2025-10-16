# Project Documentation & Knowledge Graph

## 📊 Knowledge Graph Visualization

This directory contains the project's knowledge graph visualization and data exports.

### Files

1. **`project_knowledge_graph.png`** (1.9 MB)
   - High-resolution visual graph (24" x 16", 300 DPI)
   - Color-coded nodes by file type
   - Shows all file dependencies and relationships
   - Best viewed with image viewer or browser

2. **`project_knowledge_graph.json`** (7 KB)
   - NetworkX graph data in JSON format
   - Machine-readable for programmatic analysis
   - Can be imported into NetworkX, Gephi, or other graph tools

3. **`project_relationships.csv`** (2.4 KB)
   - Edge list in CSV format
   - Columns: Source, Target, Relationship Type
   - Import into Excel, Google Sheets, or graph analysis tools

## 🎨 Node Categories

| Color | Category | Examples |
|-------|----------|----------|
| 🔵 Blue | Documentation | README.md, logbook entries, plans |
| 🟢 Green | Scripts | Training scripts, utilities |
| 🔴 Red | Core Modules | Model architectures, loss functions |
| 🟠 Orange | Data Files | Datasets, checkpoints, charsets |
| 🟣 Purple | Configuration | pyproject.toml, entrypoint.sh |

## 🔗 Relationship Types

| Type | Style | Meaning |
|------|-------|---------|
| DEPENDS_ON | Solid | Direct code dependency (import, execution) |
| REFERENCES | Dashed | Documentation reference |
| GENERATES | Dotted | Output/artifact creation |
| RELATED_TO | Thin | Topical or contextual relationship |

## 📈 Key Insights

### Most Connected Files (Hubs)
1. **`train32.py`** (12 connections) - Main training script, central to all experiments
2. **`train32_solution1_smoke_test_FIXED.sh`** (3) - Current active training
3. **`prepare_dataset.py`** (3) - Data pipeline entry point

### Critical Dependencies
- All training scripts → `train32.py`
- `train32.py` → Model architectures (generator, discriminator)
- `train32.py` → Loss functions and datasets
- Documentation → Standards and templates

## 🔄 Updating the Graph

When you add new files or change relationships:

```bash
# Re-generate knowledge graph
poetry run python scripts/generate_knowledge_graph.py

# Check the output
xdg-open docs/project_knowledge_graph.png
```

## 📊 Analysis Examples

### Find all dependents of a file:
```bash
grep ",train32.py," docs/project_relationships.csv
```

### Find what a file depends on:
```bash
grep "^\"train32.py\"," docs/project_relationships.csv
```

### Count relationships by type:
```bash
cut -d',' -f3 docs/project_relationships.csv | sort | uniq -c
```

## 🔍 Integration with Other Tools

### Import to Gephi:
1. Open Gephi
2. File → Open → Select `project_relationships.csv`
3. Graph type: Directed
4. Create missing nodes: Yes

### Import to NetworkX (Python):
```python
import json
import networkx as nx

# Load graph
with open('docs/project_knowledge_graph.json') as f:
    data = json.load(f)
G = nx.node_link_graph(data)

# Analyze
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

### Import to Pandas (Python):
```python
import pandas as pd

# Load relationships
df = pd.read_csv('docs/project_relationships.csv')
print(df.head())

# Analyze
print(df['Relationship Type'].value_counts())
```

---

**Last Updated:** 2025-10-16 11:10:00 WIB
**Generator:** `scripts/generate_knowledge_graph.py`
**Maintained by:** AI Assistant + Belekok
