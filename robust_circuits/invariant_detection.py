"""
Invariant Detection: Methods for identifying circuit components that remain
consistent across different variable instantiations and contexts.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass
from itertools import product, combinations
import re
from collections import defaultdict

import pyvene as pv
from pyvene import IntervenableModel


@dataclass 
class InvariantPattern:
    """Represents a pattern that remains invariant across contexts."""
    pattern_type: str          # e.g., "induction", "name_mover", "position_invariant"
    components: List[str]      # Circuit components involved
    confidence: float          # How confident we are this is truly invariant
    contexts_tested: int       # Number of contexts where pattern held
    failure_modes: List[str]   # Contexts where pattern failed
    functional_description: str # What the pattern does
    

class InvariantDetector:
    """
    Detects circuit patterns that are invariant across different contexts
    and variable instantiations.
    
    This addresses the core challenge that circuits should work regardless of
    surface-level changes like different names, objects, or sentence structures.
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def find_invariant_circuits(
        self,
        base_templates: List[str],
        variable_sets: Dict[str, List[str]],
        behavioral_tests: List[Callable],
        min_contexts: int = 10,
        consistency_threshold: float = 0.85
    ) -> List[InvariantPattern]:
        """
        Find circuits that remain invariant across different variable instantiations.
        
        Args:
            base_templates: Template strings with placeholders (e.g., "When {name} went to {place}")
            variable_sets: Dictionary mapping placeholder names to possible values
            behavioral_tests: Functions that test specific behaviors
            min_contexts: Minimum number of contexts a pattern must work in
            consistency_threshold: Minimum consistency score to be considered invariant
            
        Returns:
            List of invariant patterns found
        """
        invariant_patterns = []
        
        # Generate test contexts by filling templates
        test_contexts = self._generate_test_contexts(base_templates, variable_sets, min_contexts * 2)
        
        # For each behavioral test, find invariant components
        for i, behavioral_test in enumerate(behavioral_tests):
            print(f"Testing behavioral pattern {i+1}/{len(behavioral_tests)}")
            
            # Find components that consistently affect this behavior
            consistent_components = self._find_consistent_components(
                test_contexts, behavioral_test, consistency_threshold
            )
            
            if len(consistent_components) > 0:
                # Validate the pattern across more contexts
                pattern = self._validate_invariant_pattern(
                    consistent_components,
                    test_contexts,
                    behavioral_test,
                    consistency_threshold
                )
                
                if pattern.contexts_tested >= min_contexts:
                    invariant_patterns.append(pattern)
                    
        return invariant_patterns
    
    def test_context_independence(
        self,
        circuit_config: Dict,
        context_variations: List[Tuple[str, str]],  # (original, variation) pairs
        behavioral_metric: Callable
    ) -> Dict[str, float]:
        """
        Test how independent a circuit is from specific contextual features.
        
        This tests the core hypothesis that robust circuits should work 
        regardless of surface-level context changes.
        """
        independence_scores = {}
        
        for original_context, varied_context in context_variations:
            # Test circuit on original context
            original_behavior = self._measure_circuit_behavior(
                original_context, circuit_config, behavioral_metric
            )
            
            # Test circuit on varied context  
            varied_behavior = self._measure_circuit_behavior(
                varied_context, circuit_config, behavioral_metric
            )
            
            # Compute similarity (independence means high similarity)
            if original_behavior is not None and varied_behavior is not None:
                similarity = self._compute_behavior_similarity(original_behavior, varied_behavior)
                independence_scores[f"{original_context} -> {varied_context}"] = similarity
                
        return independence_scores
    
    def detect_universal_patterns(
        self,
        known_circuits: List[Dict],
        test_scenarios: List[str],
        pattern_library: Optional[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        Detect universal patterns that appear across multiple known circuits.
        
        This implements the idea that there might be universal circuit "motifs"
        that appear across different tasks and contexts.
        """
        if pattern_library is None:
            pattern_library = self._get_default_pattern_library()
            
        universal_patterns = defaultdict(list)
        
        for pattern_name, pattern_signature in pattern_library.items():
            matching_circuits = []
            
            for circuit in known_circuits:
                if self._circuit_matches_pattern(circuit, pattern_signature):
                    # Test if pattern works in this circuit across scenarios
                    success_rate = self._test_pattern_robustness(
                        circuit, pattern_signature, test_scenarios
                    )
                    
                    if success_rate > 0.8:  # Pattern works robustly
                        matching_circuits.append(circuit.get("name", "unnamed"))
                        
            if len(matching_circuits) >= 2:  # Universal = appears in multiple circuits
                universal_patterns[pattern_name] = matching_circuits
                
        return dict(universal_patterns)
    
    def _generate_test_contexts(
        self, 
        templates: List[str], 
        variable_sets: Dict[str, List[str]], 
        num_contexts: int
    ) -> List[str]:
        """Generate test contexts by filling templates with different variables."""
        contexts = []
        
        # Get all placeholder names from templates
        all_placeholders = set()
        for template in templates:
            placeholders = re.findall(r'\{(\w+)\}', template)
            all_placeholders.update(placeholders)
            
        # Generate combinations
        available_vars = {}
        for placeholder in all_placeholders:
            if placeholder in variable_sets:
                available_vars[placeholder] = variable_sets[placeholder]
            else:
                print(f"Warning: No variables provided for placeholder '{placeholder}'")
                available_vars[placeholder] = [f"default_{placeholder}"]
                
        # Create combinations
        placeholder_names = list(available_vars.keys())
        if not placeholder_names:
            return templates  # No placeholders to fill
            
        # Generate combinations up to num_contexts
        combinations_generated = 0
        for template in templates:
            if combinations_generated >= num_contexts:
                break
                
            # Get relevant placeholders for this template
            template_placeholders = re.findall(r'\{(\w+)\}', template)
            
            if not template_placeholders:
                contexts.append(template)
                combinations_generated += 1
                continue
                
            # Generate variable combinations for this template
            relevant_vars = [available_vars[p] for p in template_placeholders]
            
            for combo in product(*relevant_vars):
                if combinations_generated >= num_contexts:
                    break
                    
                # Fill template
                filled_template = template
                for placeholder, value in zip(template_placeholders, combo):
                    filled_template = filled_template.replace(f"{{{placeholder}}}", value)
                    
                contexts.append(filled_template)
                combinations_generated += 1
                
        return contexts[:num_contexts]
    
    def _find_consistent_components(
        self,
        contexts: List[str],
        behavioral_test: Callable,
        threshold: float
    ) -> List[str]:
        """Find model components that consistently affect the behavioral test."""
        
        # Get model architecture info
        component_importance = defaultdict(list)
        
        for context in contexts:
            # Test each potential component
            for layer in range(getattr(self.model.config, 'n_layer', 12)):
                for component in ['attention', 'mlp_output', 'block_output']:
                    component_id = f"layer_{layer}_{component}"
                    
                    # Test importance via ablation
                    importance = self._test_component_importance(
                        context, layer, component, behavioral_test
                    )
                    
                    component_importance[component_id].append(importance)
                    
        # Find consistently important components
        consistent_components = []
        for component_id, importances in component_importance.items():
            if len(importances) > 0:
                consistency = np.mean([1.0 if imp > 0.1 else 0.0 for imp in importances])
                if consistency >= threshold:
                    consistent_components.append(component_id)
                    
        return consistent_components
    
    def _test_component_importance(
        self,
        context: str,
        layer: int,
        component: str,
        behavioral_test: Callable
    ) -> float:
        """Test how important a component is for a specific behavior."""
        try:
            # Create intervention config
            config = pv.IntervenableConfig(
                model_type=type(self.model),
                representations=[{
                    "layer": layer,
                    "component": component,
                    "intervention_type": pv.ZeroIntervention
                }]
            )
            
            intervenable = pv.IntervenableModel(config, self.model)
            
            # Get baseline behavior
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                baseline_output = self.model(**inputs)
                baseline_behavior = behavioral_test(baseline_output, inputs)
                
                # Get behavior with intervention
                intervened_output = intervenable(base=inputs)
                intervened_behavior = behavioral_test(intervened_output, inputs)
                
            # Measure difference
            if baseline_behavior is not None and intervened_behavior is not None:
                if isinstance(baseline_behavior, torch.Tensor):
                    baseline_behavior = baseline_behavior.item()
                if isinstance(intervened_behavior, torch.Tensor):
                    intervened_behavior = intervened_behavior.item()
                    
                importance = abs(baseline_behavior - intervened_behavior)
                return importance
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error testing component {layer}_{component}: {e}")
            return 0.0
    
    def _validate_invariant_pattern(
        self,
        components: List[str],
        test_contexts: List[str],
        behavioral_test: Callable,
        threshold: float
    ) -> InvariantPattern:
        """Validate that a pattern is truly invariant across contexts."""
        
        successes = 0
        failures = []
        
        for context in test_contexts:
            # Test if pattern works in this context
            success = self._test_pattern_in_context(
                components, context, behavioral_test, threshold
            )
            
            if success:
                successes += 1
            else:
                failures.append(context)
                
        confidence = successes / len(test_contexts) if test_contexts else 0.0
        
        return InvariantPattern(
            pattern_type="discovered",
            components=components,
            confidence=confidence,
            contexts_tested=len(test_contexts),
            failure_modes=failures[:5],  # Keep first 5 failures
            functional_description=f"Pattern involving {len(components)} components"
        )
    
    def _test_pattern_in_context(
        self,
        components: List[str],
        context: str,
        behavioral_test: Callable,
        threshold: float
    ) -> bool:
        """Test if a pattern works in a specific context."""
        try:
            # This is a simplified test - in practice you'd do more sophisticated validation
            for component in components:
                parts = component.split('_')
                if len(parts) >= 3:
                    layer = int(parts[1])
                    comp_type = '_'.join(parts[2:])
                    
                    importance = self._test_component_importance(
                        context, layer, comp_type, behavioral_test
                    )
                    
                    if importance < 0.1:  # Component not important in this context
                        return False
                        
            return True
            
        except Exception:
            return False
    
    def _measure_circuit_behavior(
        self,
        context: str,
        circuit_config: Dict,
        behavioral_metric: Callable
    ) -> Optional[float]:
        """Measure how a circuit behaves in a specific context."""
        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                behavior = behavioral_metric(outputs, inputs)
                
            return behavior if isinstance(behavior, (int, float)) else behavior.item()
            
        except Exception:
            return None
    
    def _compute_behavior_similarity(self, behavior1: float, behavior2: float) -> float:
        """Compute similarity between two behavioral measurements."""
        if behavior1 == behavior2:
            return 1.0
            
        max_val = max(abs(behavior1), abs(behavior2))
        if max_val == 0:
            return 1.0
            
        return 1.0 - abs(behavior1 - behavior2) / max_val
    
    def _get_default_pattern_library(self) -> Dict[str, Dict]:
        """Get default library of known circuit patterns."""
        return {
            "induction_pattern": {
                "description": "Copies tokens based on previous occurrence",
                "required_components": ["prev_token_head", "induction_head"],
                "layer_pattern": "early_to_mid"
            },
            "name_mover_pattern": {
                "description": "Moves name information to final position",
                "required_components": ["attention_head", "mlp"],
                "layer_pattern": "mid_to_late"
            },
            "suppression_pattern": {
                "description": "Suppresses repeated information",
                "required_components": ["negative_heads"],
                "layer_pattern": "late"
            }
        }
    
    def _circuit_matches_pattern(self, circuit: Dict, pattern: Dict) -> bool:
        """Check if a circuit matches a known pattern."""
        # Simplified pattern matching - in practice this would be more sophisticated
        circuit_components = circuit.get("components", [])
        required_components = pattern.get("required_components", [])
        
        # Check if circuit has similar components
        matches = 0
        for req_comp in required_components:
            for circuit_comp in circuit_components:
                if req_comp.lower() in circuit_comp.lower():
                    matches += 1
                    break
                    
        return matches >= len(required_components) * 0.7  # 70% match required
    
    def _test_pattern_robustness(
        self,
        circuit: Dict,
        pattern: Dict,
        scenarios: List[str]
    ) -> float:
        """Test how robust a pattern is across different scenarios."""
        # Simplified robustness test
        successes = 0
        
        for scenario in scenarios:
            # In practice, this would test the specific pattern on each scenario
            # For now, we'll use a simple heuristic
            if len(circuit.get("components", [])) > 0:
                successes += 1
                
        return successes / len(scenarios) if scenarios else 0.0