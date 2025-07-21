"""
Robust Circuit Discovery Engine: The main orchestrator for finding robust circuits
that are context and variable invariant.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import time
from collections import defaultdict

import pyvene as pv
from .circuit_stability import CircuitStabilityAnalyzer, StabilityMetrics
from .invariant_detection import InvariantDetector, InvariantPattern
from .robust_interventions import RobustInterventionTechnique
from .evaluation import RobustnessEvaluator


@dataclass
class RobustCircuit:
    """Represents a discovered robust circuit."""
    name: str
    components: List[str]
    stability_metrics: StabilityMetrics
    invariant_patterns: List[InvariantPattern]
    robustness_score: float
    contexts_tested: List[str]
    discovery_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "components": self.components,
            "stability_metrics": asdict(self.stability_metrics),
            "invariant_patterns": [asdict(p) for p in self.invariant_patterns],
            "robustness_score": self.robustness_score,
            "contexts_tested": self.contexts_tested,
            "discovery_metadata": self.discovery_metadata
        }


class RobustCircuitDiscovery:
    """
    Main engine for discovering robust circuits that work across contexts and variables.
    
    This addresses the core challenge in circuit literature that circuits should be
    stable, generalizable, and not dependent on specific surface-level features.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        tokenizer,
        device: str = "auto",
        cache_dir: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize analyzers
        self.stability_analyzer = CircuitStabilityAnalyzer(model, tokenizer, device)
        self.invariant_detector = InvariantDetector(model, tokenizer, device)
        self.intervention_technique = RobustInterventionTechnique(model, tokenizer, device)
        self.evaluator = RobustnessEvaluator(model, tokenizer, device)
        
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./circuit_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.discovered_circuits = []
        
    def discover_robust_circuits(
        self,
        task_templates: List[str],
        variable_sets: Dict[str, List[str]], 
        behavioral_tests: List[Callable],
        robustness_threshold: float = 0.8,
        min_contexts: int = 20,
        max_circuits: int = 10
    ) -> List[RobustCircuit]:
        """
        Discover robust circuits for a given task.
        
        Args:
            task_templates: Template strings defining the task (with {variables})
            variable_sets: Variables to substitute in templates
            behavioral_tests: Functions that measure task-relevant behavior
            robustness_threshold: Minimum robustness score to consider circuit valid
            min_contexts: Minimum contexts a circuit must work in
            max_circuits: Maximum number of circuits to return
            
        Returns:
            List of discovered robust circuits, ranked by robustness
        """
        print("🔍 Starting robust circuit discovery...")
        
        # Step 1: Generate diverse test contexts
        print("📝 Generating test contexts...")
        test_contexts = self._generate_comprehensive_contexts(
            task_templates, variable_sets, min_contexts * 3
        )
        
        # Step 2: Find invariant patterns
        print("🔍 Detecting invariant patterns...")
        invariant_patterns = self.invariant_detector.find_invariant_circuits(
            task_templates, variable_sets, behavioral_tests, 
            min_contexts, consistency_threshold=0.8
        )
        
        if not invariant_patterns:
            print("⚠️ No invariant patterns found!")
            return []
            
        print(f"✅ Found {len(invariant_patterns)} invariant patterns")
        
        # Step 3: For each pattern, build and validate full circuit
        robust_circuits = []
        for i, pattern in enumerate(invariant_patterns):
            print(f"🔧 Building circuit {i+1}/{len(invariant_patterns)}...")
            
            circuit = self._build_robust_circuit(
                pattern, test_contexts, behavioral_tests, robustness_threshold
            )
            
            if circuit and circuit.robustness_score >= robustness_threshold:
                robust_circuits.append(circuit)
                print(f"✅ Circuit '{circuit.name}' - Robustness: {circuit.robustness_score:.3f}")
            else:
                print(f"❌ Circuit failed robustness threshold")
                
        # Step 4: Rank and return top circuits
        robust_circuits.sort(key=lambda c: c.robustness_score, reverse=True)
        final_circuits = robust_circuits[:max_circuits]
        
        # Step 5: Cache results
        self._cache_circuits(final_circuits, task_templates, variable_sets)
        
        self.discovered_circuits.extend(final_circuits)
        
        print(f"🎉 Discovery complete! Found {len(final_circuits)} robust circuits")
        return final_circuits
    
    def validate_circuit_robustness(
        self,
        circuit: RobustCircuit,
        new_contexts: List[str],
        behavioral_tests: List[Callable]
    ) -> Dict[str, float]:
        """
        Validate how robust a circuit is on new, unseen contexts.
        
        This is crucial for testing the core hypothesis that robust circuits
        should generalize to contexts they weren't trained on.
        """
        print(f"🧪 Validating circuit '{circuit.name}' on {len(new_contexts)} new contexts...")
        
        validation_results = {}
        
        # Test each behavioral aspect
        for i, behavioral_test in enumerate(behavioral_tests):
            test_name = getattr(behavioral_test, '__name__', f'test_{i}')
            
            scores = []
            for context in new_contexts:
                score = self._test_circuit_on_context(
                    circuit, context, behavioral_test
                )
                if score is not None:
                    scores.append(score)
                    
            if scores:
                avg_score = np.mean(scores)
                validation_results[test_name] = avg_score
                print(f"  {test_name}: {avg_score:.3f}")
            else:
                validation_results[test_name] = 0.0
                print(f"  {test_name}: FAILED")
                
        overall_score = np.mean(list(validation_results.values()))
        validation_results['overall'] = overall_score
        
        print(f"🎯 Overall validation score: {overall_score:.3f}")
        return validation_results
    
    def compare_circuit_families(
        self,
        circuit_family_A: List[RobustCircuit],
        circuit_family_B: List[RobustCircuit],
        test_contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Compare two families of circuits to understand their differences and similarities.
        
        This helps understand what makes some circuits more robust than others.
        """
        print("🔬 Comparing circuit families...")
        
        comparison = {
            "family_A_stats": self._compute_family_stats(circuit_family_A),
            "family_B_stats": self._compute_family_stats(circuit_family_B),
            "shared_components": self._find_shared_components(circuit_family_A, circuit_family_B),
            "unique_patterns": self._find_unique_patterns(circuit_family_A, circuit_family_B),
            "robustness_comparison": self._compare_robustness(
                circuit_family_A, circuit_family_B, test_contexts
            )
        }
        
        return comparison
    
    def explain_circuit_behavior(
        self,
        circuit: RobustCircuit,
        example_contexts: List[str],
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a human-readable explanation of how a circuit works.
        
        This is crucial for interpretability and building trust in discovered circuits.
        """
        print(f"📖 Explaining circuit '{circuit.name}'...")
        
        explanation = {
            "circuit_summary": {
                "name": circuit.name,
                "robustness_score": circuit.robustness_score,
                "num_components": len(circuit.components),
                "contexts_tested": len(circuit.contexts_tested)
            },
            "functional_description": self._generate_functional_description(circuit),
            "key_components": self._identify_key_components(circuit),
            "robustness_factors": self._identify_robustness_factors(circuit),
            "example_behaviors": []
        }
        
        if detailed:
            # Generate detailed behavioral examples
            for context in example_contexts[:5]:  # Limit to 5 examples
                behavior_explanation = self._explain_behavior_on_context(circuit, context)
                explanation["example_behaviors"].append({
                    "context": context,
                    "explanation": behavior_explanation
                })
                
        return explanation
    
    def find_universal_motifs(
        self,
        discovered_circuits: Optional[List[RobustCircuit]] = None
    ) -> Dict[str, Any]:
        """
        Find universal circuit motifs that appear across multiple robust circuits.
        
        This addresses the literature's claim that there should be universal
        computational patterns that generalize across tasks.
        """
        if discovered_circuits is None:
            discovered_circuits = self.discovered_circuits
            
        print(f"🔍 Searching for universal motifs in {len(discovered_circuits)} circuits...")
        
        # Convert circuits to a format suitable for pattern analysis
        circuit_dicts = [
            {
                "name": c.name,
                "components": c.components,
                "patterns": c.invariant_patterns
            }
            for c in discovered_circuits
        ]
        
        universal_patterns = self.invariant_detector.detect_universal_patterns(
            circuit_dicts, 
            [c.contexts_tested[0] for c in discovered_circuits if c.contexts_tested]
        )
        
        motif_analysis = {
            "universal_patterns": universal_patterns,
            "pattern_frequency": self._compute_pattern_frequency(discovered_circuits),
            "robustness_correlation": self._correlate_patterns_with_robustness(discovered_circuits),
            "suggested_motifs": self._suggest_circuit_motifs(discovered_circuits)
        }
        
        print(f"✅ Found {len(universal_patterns)} universal patterns")
        return motif_analysis
    
    def _generate_comprehensive_contexts(
        self,
        templates: List[str],
        variable_sets: Dict[str, List[str]],
        num_contexts: int
    ) -> List[str]:
        """Generate a comprehensive set of test contexts."""
        
        # Use the invariant detector's context generation, but add more variety
        base_contexts = self.invariant_detector._generate_test_contexts(
            templates, variable_sets, num_contexts // 2
        )
        
        # Add synthetic variations to test edge cases
        varied_contexts = []
        for context in base_contexts[:10]:  # Vary first 10 contexts
            variations = self._create_context_variations(context)
            varied_contexts.extend(variations)
            
        all_contexts = base_contexts + varied_contexts
        return list(set(all_contexts))[:num_contexts]  # Remove duplicates
    
    def _create_context_variations(self, base_context: str) -> List[str]:
        """Create variations of a context to test robustness."""
        variations = []
        
        # Add punctuation variations
        variations.append(base_context + ".")
        variations.append(base_context + "!")
        variations.append(base_context + "?")
        
        # Add case variations
        variations.append(base_context.upper())
        variations.append(base_context.lower())
        
        # Add prefix/suffix variations  
        variations.append("Actually, " + base_context.lower())
        variations.append(base_context + " obviously")
        
        return variations
    
    def _build_robust_circuit(
        self,
        pattern: InvariantPattern,
        test_contexts: List[str],
        behavioral_tests: List[Callable],
        robustness_threshold: float
    ) -> Optional[RobustCircuit]:
        """Build a full robust circuit from an invariant pattern."""
        
        # Create circuit configuration
        circuit_config = self._pattern_to_circuit_config(pattern)
        
        # Test stability across contexts
        stability_metrics = self.stability_analyzer.analyze_stability(
            circuit_config, test_contexts[:15], behavioral_tests[0]
        )
        
        # Compute overall robustness score
        robustness_score = (
            stability_metrics.structural_consistency * 0.3 +
            stability_metrics.functional_consistency * 0.4 + 
            stability_metrics.context_independence * 0.3
        )
        
        if robustness_score < robustness_threshold:
            return None
            
        # Create robust circuit
        circuit = RobustCircuit(
            name=f"robust_circuit_{pattern.pattern_type}_{int(time.time())}",
            components=pattern.components,
            stability_metrics=stability_metrics,
            invariant_patterns=[pattern],
            robustness_score=robustness_score,
            contexts_tested=test_contexts[:15],
            discovery_metadata={
                "discovery_time": time.time(),
                "pattern_confidence": pattern.confidence,
                "num_behavioral_tests": len(behavioral_tests)
            }
        )
        
        return circuit
    
    def _pattern_to_circuit_config(self, pattern: InvariantPattern) -> Dict:
        """Convert an invariant pattern to a circuit configuration."""
        
        representations = []
        interventions = {}
        
        for i, component in enumerate(pattern.components):
            # Parse component string (e.g., "layer_5_attention")
            parts = component.split('_')
            if len(parts) >= 3:
                layer = int(parts[1])
                comp_type = '_'.join(parts[2:])
                
                representations.append({
                    "layer": layer,
                    "component": comp_type,
                    "intervention_type": pv.VanillaIntervention
                })
                
                interventions[f"intervention_{i}"] = {
                    "locations": {"sources->base": [(None, [layer])]},
                    "subspaces": None
                }
                
        return {
            "representations": representations,
            "intervention_types": [pv.VanillaIntervention] * len(representations),
            "interventions": interventions
        }
    
    def _test_circuit_on_context(
        self,
        circuit: RobustCircuit,
        context: str,
        behavioral_test: Callable
    ) -> Optional[float]:
        """Test how well a circuit works on a specific context."""
        try:
            # This is simplified - in practice you'd reconstruct the full circuit
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = behavioral_test(outputs, inputs)
                
            return score.item() if isinstance(score, torch.Tensor) else score
            
        except Exception as e:
            print(f"Error testing circuit on context: {e}")
            return None
    
    def _compute_family_stats(self, circuits: List[RobustCircuit]) -> Dict:
        """Compute statistics for a family of circuits."""
        if not circuits:
            return {}
            
        robustness_scores = [c.robustness_score for c in circuits]
        num_components = [len(c.components) for c in circuits]
        
        return {
            "count": len(circuits),
            "avg_robustness": np.mean(robustness_scores),
            "std_robustness": np.std(robustness_scores),
            "avg_components": np.mean(num_components),
            "component_range": [min(num_components), max(num_components)]
        }
    
    def _find_shared_components(
        self, 
        family_A: List[RobustCircuit], 
        family_B: List[RobustCircuit]
    ) -> List[str]:
        """Find components shared between circuit families."""
        components_A = set()
        for circuit in family_A:
            components_A.update(circuit.components)
            
        components_B = set()
        for circuit in family_B:
            components_B.update(circuit.components)
            
        return list(components_A & components_B)
    
    def _find_unique_patterns(
        self,
        family_A: List[RobustCircuit],
        family_B: List[RobustCircuit]
    ) -> Dict[str, List[str]]:
        """Find patterns unique to each family."""
        patterns_A = set()
        for circuit in family_A:
            for pattern in circuit.invariant_patterns:
                patterns_A.add(pattern.pattern_type)
                
        patterns_B = set()
        for circuit in family_B:
            for pattern in circuit.invariant_patterns:
                patterns_B.add(pattern.pattern_type)
                
        return {
            "unique_to_A": list(patterns_A - patterns_B),
            "unique_to_B": list(patterns_B - patterns_A),
            "shared": list(patterns_A & patterns_B)
        }
    
    def _compare_robustness(
        self,
        family_A: List[RobustCircuit],
        family_B: List[RobustCircuit], 
        test_contexts: List[str]
    ) -> Dict[str, float]:
        """Compare robustness between circuit families."""
        
        avg_robustness_A = np.mean([c.robustness_score for c in family_A]) if family_A else 0
        avg_robustness_B = np.mean([c.robustness_score for c in family_B]) if family_B else 0
        
        return {
            "family_A_avg": avg_robustness_A,
            "family_B_avg": avg_robustness_B,
            "difference": avg_robustness_A - avg_robustness_B,
            "better_family": "A" if avg_robustness_A > avg_robustness_B else "B"
        }
    
    def _generate_functional_description(self, circuit: RobustCircuit) -> str:
        """Generate a human-readable description of what the circuit does."""
        
        # This is simplified - in practice you'd analyze the actual circuit behavior
        if len(circuit.invariant_patterns) > 0:
            primary_pattern = circuit.invariant_patterns[0]
            return primary_pattern.functional_description
        else:
            return f"Circuit with {len(circuit.components)} components"
    
    def _identify_key_components(self, circuit: RobustCircuit) -> List[str]:
        """Identify the most important components in a circuit."""
        # Simplified - would analyze actual importance scores
        return circuit.components[:3]  # Return top 3 components
    
    def _identify_robustness_factors(self, circuit: RobustCircuit) -> List[str]:
        """Identify what makes this circuit robust."""
        factors = []
        
        if circuit.stability_metrics.structural_consistency > 0.8:
            factors.append("Consistent structure across contexts")
            
        if circuit.stability_metrics.functional_consistency > 0.8:
            factors.append("Stable functional behavior")
            
        if circuit.stability_metrics.context_independence > 0.8:
            factors.append("Independent of surface-level context features")
            
        return factors
    
    def _explain_behavior_on_context(self, circuit: RobustCircuit, context: str) -> str:
        """Explain how the circuit behaves on a specific context."""
        # Simplified explanation
        return f"Circuit processes '{context}' using {len(circuit.components)} components"
    
    def _compute_pattern_frequency(self, circuits: List[RobustCircuit]) -> Dict[str, int]:
        """Compute how frequently different patterns appear."""
        pattern_counts = defaultdict(int)
        
        for circuit in circuits:
            for pattern in circuit.invariant_patterns:
                pattern_counts[pattern.pattern_type] += 1
                
        return dict(pattern_counts)
    
    def _correlate_patterns_with_robustness(self, circuits: List[RobustCircuit]) -> Dict[str, float]:
        """Correlate pattern types with robustness scores."""
        pattern_robustness = defaultdict(list)
        
        for circuit in circuits:
            for pattern in circuit.invariant_patterns:
                pattern_robustness[pattern.pattern_type].append(circuit.robustness_score)
                
        correlations = {}
        for pattern, scores in pattern_robustness.items():
            correlations[pattern] = np.mean(scores)
            
        return correlations
    
    def _suggest_circuit_motifs(self, circuits: List[RobustCircuit]) -> List[str]:
        """Suggest potential universal circuit motifs."""
        # This would analyze actual circuit structures - simplified for now
        common_components = defaultdict(int)
        
        for circuit in circuits:
            for component in circuit.components:
                common_components[component] += 1
                
        # Find components that appear in >50% of circuits
        total_circuits = len(circuits)
        motifs = []
        
        for component, count in common_components.items():
            if count >= total_circuits * 0.5:
                motifs.append(f"Universal component: {component}")
                
        return motifs
    
    def _cache_circuits(
        self,
        circuits: List[RobustCircuit],
        templates: List[str],
        variables: Dict[str, List[str]]
    ):
        """Cache discovered circuits for future use."""
        cache_data = {
            "circuits": [circuit.to_dict() for circuit in circuits],
            "discovery_config": {
                "templates": templates,
                "variables": variables,
                "timestamp": time.time()
            }
        }
        
        cache_file = self.cache_dir / f"circuits_{int(time.time())}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
        print(f"💾 Cached {len(circuits)} circuits to {cache_file}")
    
    def load_cached_circuits(self, cache_file: str) -> List[RobustCircuit]:
        """Load previously discovered circuits from cache."""
        cache_path = self.cache_dir / cache_file
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
            
        circuits = []
        for circuit_data in cache_data["circuits"]:
            # Reconstruct stability metrics
            stability_data = circuit_data["stability_metrics"]
            stability_metrics = StabilityMetrics(**stability_data)
            
            # Reconstruct invariant patterns
            patterns = []
            for pattern_data in circuit_data["invariant_patterns"]:
                pattern = InvariantPattern(**pattern_data)
                patterns.append(pattern)
                
            # Reconstruct circuit
            circuit = RobustCircuit(
                name=circuit_data["name"],
                components=circuit_data["components"],
                stability_metrics=stability_metrics,
                invariant_patterns=patterns,
                robustness_score=circuit_data["robustness_score"],
                contexts_tested=circuit_data["contexts_tested"],
                discovery_metadata=circuit_data["discovery_metadata"]
            )
            circuits.append(circuit)
            
        print(f"📂 Loaded {len(circuits)} circuits from cache")
        return circuits