#!/usr/bin/env python3
"""
Knowledge Graph Generator for Document Restoration GAN-HTR Project

Purpose: Generate visual knowledge graph showing file relationships
Author: AI Assistant + Belekok
Created: 2025-10-16 11:00:00 WIB
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from datetime import datetime
import os

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Node categories and colors
NODE_CATEGORIES = {
    'documentation': {'color': '#3498db', 'shape': 'o'},  # Blue circle
    'script': {'color': '#2ecc71', 'shape': 's'},          # Green square
    'core_module': {'color': '#e74c3c', 'shape': '^'},     # Red triangle
    'data': {'color': '#f39c12', 'shape': 'd'},            # Orange diamond
    'config': {'color': '#9b59b6', 'shape': 'v'},          # Purple inverted triangle
}

# Edge types
EDGE_TYPES = {
    'DEPENDS_ON': {'style': 'solid', 'width': 2.0, 'color': '#2c3e50'},
    'REFERENCES': {'style': 'dashed', 'width': 1.5, 'color': '#7f8c8d'},
    'GENERATES': {'style': 'dotted', 'width': 1.5, 'color': '#16a085'},
    'RELATED_TO': {'style': 'dashdot', 'width': 1.0, 'color': '#95a5a6'},
}


def create_knowledge_graph():
    """Create NetworkX directed graph with project file relationships."""
    G = nx.DiGraph()
    
    # ========== CORE NODES ==========
    
    # Documentation
    docs = {
        'README.md': 'documentation',
        'MANIFEST.md': 'documentation',
        'QUICKSTART.md': 'documentation',
        '.github/copilot-instructions.md': 'documentation',
    }
    
    # Logbook entries
    logbooks = {
        'logbook/20251016_bugfix_nan_loss_lr_alpha_zero.md': 'documentation',
        'logbook/20251016_solution1_lr_scheduling_smoke_test.md': 'documentation',
        'logbook/20251016_experiment2b_analysis_vs_contingency_plan.md': 'documentation',
        'logbook/20251015_experiment_2b_full_unlimited_training.md': 'documentation',
        'logbook/TEMPLATE_logbook.md': 'documentation',
        'logbook/LOGBOOK_STANDARDS.md': 'documentation',
    }
    
    # Planning documents
    planning = {
        'catatan/RENCANA_KONTINGENSI_POST_EXPERIMENT1.md': 'documentation',
        'catatan/implementasiNovelty.md': 'documentation',
        'catatan/StrategiOptimalisasi.md': 'documentation',
    }
    
    # Training scripts
    scripts = {
        'scripts/train32_solution1_smoke_test_FIXED.sh': 'script',
        'scripts/train32_solution1_smoke_test.sh': 'script',
        'scripts/monitor_solution1_smoke.sh': 'script',
        'scripts/analyze_solution1_results.sh': 'script',
        'scripts/train32_continue.sh': 'script',
        'quick_start_training.sh': 'script',
    }
    
    # Core Python modules
    core_modules = {
        'dual_modal_gan/scripts/train32.py': 'core_module',
        'dual_modal_gan/src/models/generator.py': 'core_module',
        'dual_modal_gan/src/models/discriminator.py': 'core_module',
        'dual_modal_gan/src/losses/dual_modal_loss.py': 'core_module',
        'dual_modal_gan/data/prepare_dataset.py': 'core_module',
    }
    
    # Data files
    data_files = {
        'dual_modal_gan/data/dataset_gan.tfrecord': 'data',
        'real_data_preparation/real_data_charlist.txt': 'data',
        'models/best_htr_recognizer/best_model.weights.h5': 'data',
        'dual_modal_gan/checkpoints/solution1_smoke_fixed/': 'data',
    }
    
    # Configuration
    configs = {
        'pyproject.toml': 'config',
        'entrypoint.sh': 'config',
    }
    
    # Add all nodes
    all_nodes = {**docs, **logbooks, **planning, **scripts, **core_modules, **data_files, **configs}
    for node, category in all_nodes.items():
        G.add_node(node, category=category)
    
    # ========== EDGES (Relationships) ==========
    
    # Solution 1 Workflow
    workflow_edges = [
        ('catatan/RENCANA_KONTINGENSI_POST_EXPERIMENT1.md', 
         'logbook/20251016_experiment2b_analysis_vs_contingency_plan.md', 'REFERENCES'),
        ('logbook/20251016_experiment2b_analysis_vs_contingency_plan.md',
         'scripts/train32_solution1_smoke_test.sh', 'GENERATES'),
        ('scripts/train32_solution1_smoke_test.sh',
         'logbook/20251016_solution1_lr_scheduling_smoke_test.md', 'GENERATES'),
        ('logbook/20251016_solution1_lr_scheduling_smoke_test.md',
         'logbook/20251016_bugfix_nan_loss_lr_alpha_zero.md', 'REFERENCES'),
        ('logbook/20251016_bugfix_nan_loss_lr_alpha_zero.md',
         'scripts/train32_solution1_smoke_test_FIXED.sh', 'GENERATES'),
    ]
    
    # Script dependencies
    script_edges = [
        ('scripts/train32_solution1_smoke_test_FIXED.sh', 'dual_modal_gan/scripts/train32.py', 'DEPENDS_ON'),
        ('scripts/train32_solution1_smoke_test.sh', 'dual_modal_gan/scripts/train32.py', 'DEPENDS_ON'),
        ('scripts/train32_continue.sh', 'dual_modal_gan/scripts/train32.py', 'DEPENDS_ON'),
        ('quick_start_training.sh', 'dual_modal_gan/scripts/train32.py', 'DEPENDS_ON'),
        ('scripts/monitor_solution1_smoke.sh', 'scripts/train32_solution1_smoke_test_FIXED.sh', 'RELATED_TO'),
        ('scripts/analyze_solution1_results.sh', 'scripts/monitor_solution1_smoke.sh', 'RELATED_TO'),
    ]
    
    # Core module dependencies
    module_edges = [
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/src/models/generator.py', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/src/models/discriminator.py', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/src/losses/dual_modal_loss.py', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/data/prepare_dataset.py', 'DEPENDS_ON'),
    ]
    
    # Data dependencies
    data_edges = [
        ('dual_modal_gan/data/prepare_dataset.py', 'dual_modal_gan/data/dataset_gan.tfrecord', 'GENERATES'),
        ('dual_modal_gan/data/prepare_dataset.py', 'real_data_preparation/real_data_charlist.txt', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/data/dataset_gan.tfrecord', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'models/best_htr_recognizer/best_model.weights.h5', 'DEPENDS_ON'),
        ('dual_modal_gan/scripts/train32.py', 'dual_modal_gan/checkpoints/solution1_smoke_fixed/', 'GENERATES'),
    ]
    
    # Documentation relationships
    doc_edges = [
        ('README.md', 'QUICKSTART.md', 'REFERENCES'),
        ('QUICKSTART.md', 'quick_start_training.sh', 'REFERENCES'),
        ('MANIFEST.md', 'logbook/LOGBOOK_STANDARDS.md', 'REFERENCES'),
        ('logbook/LOGBOOK_STANDARDS.md', 'logbook/TEMPLATE_logbook.md', 'REFERENCES'),
        ('.github/copilot-instructions.md', 'MANIFEST.md', 'REFERENCES'),
    ]
    
    # Configuration dependencies
    config_edges = [
        ('pyproject.toml', 'dual_modal_gan/scripts/train32.py', 'RELATED_TO'),
        ('README.md', 'pyproject.toml', 'REFERENCES'),
    ]
    
    # Add all edges
    all_edges = (workflow_edges + script_edges + module_edges + 
                 data_edges + doc_edges + config_edges)
    for source, target, edge_type in all_edges:
        if source in all_nodes and target in all_nodes:
            G.add_edge(source, target, edge_type=edge_type)
    
    return G


def visualize_graph(G, output_path='project_knowledge_graph.png'):
    """Create visual representation of knowledge graph."""
    plt.figure(figsize=(24, 16))
    
    # Use hierarchical layout for better readability
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes by category
    for category, style in NODE_CATEGORIES.items():
        nodes = [node for node, data in G.nodes(data=True) if data.get('category') == category]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=style['color'],
            node_shape=style['shape'],
            node_size=1500,
            alpha=0.9,
            linewidths=2,
            edgecolors='#34495e'
        )
    
    # Draw edges by type
    for edge_type, style in EDGE_TYPES.items():
        edges = [(u, v) for u, v, data in G.edges(data=True) if data.get('edge_type') == edge_type]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=style['color'],
            style=style['style'],
            width=style['width'],
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1'
        )
    
    # Draw labels with smaller font for readability
    labels = {node: Path(node).name for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=7,
        font_weight='bold',
        font_color='#2c3e50'
    )
    
    # Create legend
    legend_elements = []
    
    # Node types
    for category, style in NODE_CATEGORIES.items():
        legend_elements.append(
            mpatches.Patch(color=style['color'], label=category.replace('_', ' ').title())
        )
    
    # Edge types
    for edge_type, style in EDGE_TYPES.items():
        legend_elements.append(
            mpatches.Patch(color=style['color'], label=edge_type.replace('_', ' ').title(), 
                         linestyle=style['style'])
        )
    
    plt.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=10,
        title='Legend',
        title_fontsize=12,
        framealpha=0.9
    )
    
    plt.title('Document Restoration GAN-HTR - Project Knowledge Graph', 
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    # Save
    output_file = OUTPUT_DIR / output_path
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Knowledge graph saved to: {output_file}")
    
    return output_file


def export_graph_data(G, json_path='project_knowledge_graph.json', csv_path='project_relationships.csv'):
    """Export graph data to JSON and CSV formats."""
    
    # JSON export (NetworkX format)
    graph_data = nx.node_link_data(G)
    json_file = OUTPUT_DIR / json_path
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Graph data exported to: {json_file}")
    
    # CSV export (edge list)
    csv_file = OUTPUT_DIR / csv_path
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Source,Target,Relationship Type\n")
        for source, target, data in G.edges(data=True):
            edge_type = data.get('edge_type', 'UNKNOWN')
            f.write(f'"{source}","{target}","{edge_type}"\n')
    print(f"✅ Relationship CSV exported to: {csv_file}")
    
    return json_file, csv_file


def print_graph_statistics(G):
    """Print graph statistics and metrics."""
    print("\n" + "="*70)
    print("📊 KNOWLEDGE GRAPH STATISTICS")
    print("="*70)
    
    print(f"\n🔢 Basic Metrics:")
    print(f"   Total Nodes: {G.number_of_nodes()}")
    print(f"   Total Edges: {G.number_of_edges()}")
    print(f"   Graph Density: {nx.density(G):.4f}")
    
    print(f"\n📦 Nodes by Category:")
    categories = {}
    for node, data in G.nodes(data=True):
        cat = data.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"   {cat.replace('_', ' ').title()}: {count}")
    
    print(f"\n🔗 Edges by Type:")
    edge_types = {}
    for u, v, data in G.edges(data=True):
        etype = data.get('edge_type', 'UNKNOWN')
        edge_types[etype] = edge_types.get(etype, 0) + 1
    for etype, count in sorted(edge_types.items()):
        print(f"   {etype.replace('_', ' ')}: {count}")
    
    print(f"\n🔝 Most Connected Nodes (Top 10):")
    degree_centrality = nx.degree_centrality(G)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (node, centrality) in enumerate(top_nodes, 1):
        short_name = Path(node).name
        print(f"   {i:2d}. {short_name:50s} (degree: {G.degree(node)})")
    
    print(f"\n🎯 Hub Nodes (High Out-Degree):")
    out_degrees = dict(G.out_degree())
    hubs = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, out_deg in hubs:
        if out_deg > 0:
            print(f"   {Path(node).name:50s} → {out_deg} dependencies")
    
    print(f"\n📥 Sink Nodes (High In-Degree):")
    in_degrees = dict(G.in_degree())
    sinks = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, in_deg in sinks:
        if in_deg > 0:
            print(f"   {Path(node).name:50s} ← {in_deg} dependents")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main execution function."""
    print("🔧 Generating Project Knowledge Graph...")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"📂 Output Directory: {OUTPUT_DIR}")
    print()
    
    # Create graph
    G = create_knowledge_graph()
    
    # Print statistics
    print_graph_statistics(G)
    
    # Visualize
    img_path = visualize_graph(G)
    
    # Export data
    json_path, csv_path = export_graph_data(G)
    
    print("\n✅ Knowledge graph generation completed!")
    print(f"\n📊 Outputs:")
    print(f"   - Visual Graph: {img_path}")
    print(f"   - JSON Data: {json_path}")
    print(f"   - CSV Relationships: {csv_path}")
    print(f"\n💡 View the graph: open {img_path}")
    print()


if __name__ == "__main__":
    main()
