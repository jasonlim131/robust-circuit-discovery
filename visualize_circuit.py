#!/usr/bin/env python3
"""
Circuit Visualization Script
Plots the discovered robust circuit as a DAG and saves as PNG
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path

def load_circuit_data():
    """Load the discovered circuit data from JSON files"""
    circuit_file = Path("circuit_cache/circuits_1753052092.json")
    results_file = Path("robust_induction_results.json")
    
    with open(circuit_file, 'r') as f:
        circuit_data = json.load(f)
    
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    return circuit_data, results_data

def create_circuit_dag(components):
    """Create a directed acyclic graph representing the circuit"""
    G = nx.DiGraph()
    
    # Add nodes for each component
    for component in components:
        G.add_node(component)
    
    # Create edges based on layer dependencies
    # MLP outputs feed into block outputs, block outputs feed into next layer
    for i in range(12):  # 12 layers in GPT-2
        mlp_node = f"layer_{i}_mlp_output"
        block_node = f"layer_{i}_block_output"
        
        # MLP feeds into block output
        if mlp_node in components and block_node in components:
            G.add_edge(mlp_node, block_node)
        
        # Block output feeds into next layer's MLP (if exists)
        if i < 11:
            next_mlp = f"layer_{i+1}_mlp_output"
            if block_node in components and next_mlp in components:
                G.add_edge(block_node, next_mlp)
    
    return G

def plot_circuit_dag(G, circuit_name, robustness_score):
    """Plot the circuit DAG with custom layout and styling"""
    plt.figure(figsize=(16, 12))
    
    # Create a hierarchical layout
    pos = {}
    layer_width = 2
    
    for i in range(12):
        mlp_node = f"layer_{i}_mlp_output"
        block_node = f"layer_{i}_block_output"
        
        if mlp_node in G.nodes():
            pos[mlp_node] = (i * layer_width, 1)
        if block_node in G.nodes():
            pos[block_node] = (i * layer_width, 0)
    
    # Define node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if "mlp" in node:
            node_colors.append('#FF6B6B')  # Red for MLP
            node_sizes.append(800)
        else:
            node_colors.append('#4ECDC4')  # Teal for block output
            node_sizes.append(1000)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, 
                          arrowstyle='->', alpha=0.6, width=2)
    
    # Add labels with rotation for better readability
    labels = {node: node.replace('_', '\n') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Add title and formatting
    plt.title(f'Robust Circuit DAG: {circuit_name}\nRobustness Score: {robustness_score:.2%}', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', 
                                 markersize=15, label='MLP Output'),
                      plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4', 
                                 markersize=15, label='Block Output')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add layer labels
    for i in range(12):
        plt.text(i * layer_width, -0.7, f'Layer {i}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig('robust_circuit_dag.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('robust_circuit_dag.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Circuit DAG saved as 'robust_circuit_dag.png' and 'robust_circuit_dag.pdf'")
    
    return G

def main():
    """Main function to create and save circuit visualization"""
    print("Loading circuit data...")
    circuit_data, results_data = load_circuit_data()
    
    # Extract circuit information
    circuit = circuit_data['circuits'][0]
    components = circuit['components']
    circuit_name = circuit['name']
    robustness_score = circuit['robustness_score']
    
    print(f"Creating DAG for circuit: {circuit_name}")
    print(f"Components: {len(components)}")
    print(f"Robustness Score: {robustness_score:.2%}")
    
    # Create and plot the DAG
    G = create_circuit_dag(components)
    plot_circuit_dag(G, circuit_name, robustness_score)
    
    # Print graph statistics
    print(f"\nGraph Statistics:")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.3f}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")

if __name__ == "__main__":
    main()