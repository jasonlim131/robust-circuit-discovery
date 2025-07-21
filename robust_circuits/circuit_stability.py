"""
Circuit Stability Analysis: Methods for measuring the consistency of circuits
across different contexts and variable instantiations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score

import pyvene as pv
from pyvene import IntervenableModel


@dataclass
class StabilityMetrics:
    """Metrics for measuring circuit stability across contexts."""
    structural_consistency: float  # How consistent is the circuit structure?
    functional_consistency: float  # How consistent is the circuit function?
    robustness_score: float       # Overall robustness measure
    variance_explained: float     # How much behavior variance is explained?
    context_independence: float   # How independent is the circuit from context?
    

class CircuitStabilityAnalyzer:
    """
    Analyzes circuit stability across different contexts and variable instantiations.
    
    This class implements methods to measure how robust circuits are to changes in:
    - Context (different sentences/scenarios)
    - Variables (different names, objects, etc.)
    - Distributional shifts
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def analyze_stability(
        self, 
        circuit_config: Dict,
        test_contexts: List[str],
        behavioral_metric: Callable,
        intervention_strength: float = 1.0
    ) -> StabilityMetrics:
        """
        Analyze how stable a circuit is across different contexts.
        
        Args:
            circuit_config: Configuration defining the circuit to analyze
            test_contexts: List of different contexts to test
            behavioral_metric: Function that measures circuit behavior
            intervention_strength: Strength of interventions for testing
            
        Returns:
            StabilityMetrics object with various stability measures
        """
        # Extract circuit across contexts
        circuit_behaviors = []
        circuit_structures = []
        
        for context in test_contexts:
            behavior, structure = self._extract_circuit_info(
                context, circuit_config, behavioral_metric, intervention_strength
            )
            circuit_behaviors.append(behavior)
            circuit_structures.append(structure)
            
        # Compute stability metrics
        structural_consistency = self._compute_structural_consistency(circuit_structures)
        functional_consistency = self._compute_functional_consistency(circuit_behaviors)
        robustness_score = self._compute_robustness_score(circuit_behaviors, circuit_structures)
        variance_explained = self._compute_variance_explained(circuit_behaviors)
        context_independence = self._compute_context_independence(circuit_behaviors, test_contexts)
        
        return StabilityMetrics(
            structural_consistency=structural_consistency,
            functional_consistency=functional_consistency,
            robustness_score=robustness_score,
            variance_explained=variance_explained,
            context_independence=context_independence
        )
    
    def find_invariant_subgraph(
        self,
        circuit_configs: List[Dict],
        contexts: List[str],
        min_consistency_threshold: float = 0.8
    ) -> Dict:
        """
        Find the subgraph of a circuit that remains consistent across contexts.
        
        This implements the core idea that robust circuits should have a stable
        core that works regardless of surface-level changes.
        """
        # Build graphs for each context
        context_graphs = []
        for context in contexts:
            graph = self._build_circuit_graph(context, circuit_configs[0])
            context_graphs.append(graph)
            
        # Find common subgraph across all contexts
        invariant_nodes = set(context_graphs[0].nodes())
        for graph in context_graphs[1:]:
            invariant_nodes &= set(graph.nodes())
            
        # Filter by consistency threshold
        consistent_nodes = set()
        for node in invariant_nodes:
            consistencies = []
            for i, graph in enumerate(context_graphs):
                for j, other_graph in enumerate(context_graphs[i+1:], i+1):
                    if node in graph.nodes() and node in other_graph.nodes():
                        # Compare node properties/activations
                        consistency = self._compute_node_consistency(
                            graph.nodes[node], other_graph.nodes[node]
                        )
                        consistencies.append(consistency)
                        
            if np.mean(consistencies) >= min_consistency_threshold:
                consistent_nodes.add(node)
                
        # Extract invariant subgraph
        invariant_subgraph = context_graphs[0].subgraph(consistent_nodes)
        
        return {
            "nodes": list(consistent_nodes),
            "edges": list(invariant_subgraph.edges()),
            "consistency_scores": {node: np.mean([
                self._compute_node_consistency(
                    context_graphs[i].nodes[node], 
                    context_graphs[j].nodes[node]
                ) for i in range(len(context_graphs)) 
                  for j in range(i+1, len(context_graphs))
                  if node in context_graphs[i].nodes() and node in context_graphs[j].nodes()
            ]) for node in consistent_nodes}
        }
    
    def _extract_circuit_info(
        self, 
        context: str, 
        circuit_config: Dict, 
        behavioral_metric: Callable,
        intervention_strength: float
    ) -> Tuple[Dict, nx.Graph]:
        """Extract both behavioral and structural information about a circuit."""
        
        # Create interventions based on circuit config
        intervenable_config = pv.IntervenableConfig(
            model_type=type(self.model),
            representations=circuit_config.get("representations", []),
            intervention_types=circuit_config.get("intervention_types", [])
        )
        
        intervenable = pv.IntervenableModel(
            intervenable_config, 
            self.model
        )
        
        # Test behavior with interventions
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        # Get baseline behavior
        with torch.no_grad():
            baseline_output = self.model(**inputs)
            
        # Test with different interventions
        behavior_profile = {}
        for intervention_name, intervention_config in circuit_config.get("interventions", {}).items():
            try:
                # Apply intervention
                intervention_output = intervenable(
                    base=inputs,
                    unit_locations=intervention_config.get("locations"),
                    subspaces=intervention_config.get("subspaces")
                )
                
                # Measure behavioral change
                behavior_change = behavioral_metric(baseline_output, intervention_output)
                behavior_profile[intervention_name] = behavior_change
                
            except Exception as e:
                print(f"Failed intervention {intervention_name}: {e}")
                behavior_profile[intervention_name] = 0.0
                
        # Build structural graph
        structure_graph = self._build_circuit_graph(context, circuit_config)
        
        return behavior_profile, structure_graph
    
    def _build_circuit_graph(self, context: str, circuit_config: Dict) -> nx.Graph:
        """Build a graph representation of the circuit structure."""
        graph = nx.DiGraph()
        
        # Add nodes for each component
        for repr_config in circuit_config.get("representations", []):
            node_id = f"{repr_config['layer']}.{repr_config['component']}"
            graph.add_node(node_id, **repr_config)
            
        # Add edges based on information flow
        # This is simplified - in practice, you'd analyze actual activations
        representations = circuit_config.get("representations", [])
        for i, repr1 in enumerate(representations):
            for j, repr2 in enumerate(representations[i+1:], i+1):
                if repr2["layer"] > repr1["layer"]:
                    node1 = f"{repr1['layer']}.{repr1['component']}"
                    node2 = f"{repr2['layer']}.{repr2['component']}"
                    graph.add_edge(node1, node2, weight=1.0)
                    
        return graph
    
    def _compute_structural_consistency(self, structures: List[nx.Graph]) -> float:
        """Compute how consistent circuit structures are across contexts."""
        if len(structures) < 2:
            return 1.0
            
        consistencies = []
        for i, struct1 in enumerate(structures):
            for struct2 in structures[i+1:]:
                # Compare graph structure
                common_nodes = set(struct1.nodes()) & set(struct2.nodes())
                total_nodes = set(struct1.nodes()) | set(struct2.nodes())
                
                if len(total_nodes) == 0:
                    node_consistency = 1.0
                else:
                    node_consistency = len(common_nodes) / len(total_nodes)
                
                common_edges = set(struct1.edges()) & set(struct2.edges())
                total_edges = set(struct1.edges()) | set(struct2.edges())
                
                if len(total_edges) == 0:
                    edge_consistency = 1.0
                else:
                    edge_consistency = len(common_edges) / len(total_edges)
                
                consistencies.append((node_consistency + edge_consistency) / 2)
                
        return np.mean(consistencies) if consistencies else 1.0
    
    def _compute_functional_consistency(self, behaviors: List[Dict]) -> float:
        """Compute how consistent circuit functions are across contexts."""
        if len(behaviors) < 2:
            return 1.0
            
        # Get common intervention types
        all_keys = set()
        for behavior in behaviors:
            all_keys.update(behavior.keys())
            
        consistencies = []
        for key in all_keys:
            values = [behavior.get(key, 0.0) for behavior in behaviors]
            if len(set(values)) == 1:
                consistency = 1.0
            else:
                # Compute coefficient of variation (inverse of consistency)
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val == 0:
                    consistency = 1.0 if std_val == 0 else 0.0
                else:
                    consistency = 1.0 - (std_val / abs(mean_val))
                    consistency = max(0.0, consistency)
            consistencies.append(consistency)
            
        return np.mean(consistencies) if consistencies else 1.0
    
    def _compute_robustness_score(self, behaviors: List[Dict], structures: List[nx.Graph]) -> float:
        """Compute overall robustness as combination of structural and functional consistency."""
        structural = self._compute_structural_consistency(structures)
        functional = self._compute_functional_consistency(behaviors)
        return (structural + functional) / 2
    
    def _compute_variance_explained(self, behaviors: List[Dict]) -> float:
        """Compute how much of the behavioral variance is explained by the circuit."""
        if not behaviors:
            return 0.0
            
        # This is a simplified metric - in practice you'd compare against
        # random interventions or null models
        all_values = []
        for behavior in behaviors:
            all_values.extend(behavior.values())
            
        if not all_values:
            return 0.0
            
        # High values indicate the circuit explains significant variance
        return min(1.0, np.mean(np.abs(all_values)))
    
    def _compute_context_independence(self, behaviors: List[Dict], contexts: List[str]) -> float:
        """Compute how independent the circuit is from specific context features."""
        # This would ideally analyze semantic similarity of contexts
        # vs similarity of behaviors - for now, simplified
        return self._compute_functional_consistency(behaviors)
    
    def _compute_node_consistency(self, node1_props: Dict, node2_props: Dict) -> float:
        """Compute consistency between two nodes."""
        # Compare node properties
        common_keys = set(node1_props.keys()) & set(node2_props.keys())
        if not common_keys:
            return 0.0
            
        consistencies = []
        for key in common_keys:
            val1, val2 = node1_props[key], node2_props[key]
            if val1 == val2:
                consistencies.append(1.0)
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric comparison
                max_val = max(abs(val1), abs(val2))
                if max_val == 0:
                    consistencies.append(1.0)
                else:
                    consistencies.append(1.0 - abs(val1 - val2) / max_val)
            else:
                consistencies.append(0.0)
                
        return np.mean(consistencies) if consistencies else 0.0