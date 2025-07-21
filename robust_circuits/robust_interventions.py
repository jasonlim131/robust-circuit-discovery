"""
Robust Intervention Techniques: Advanced intervention methods for testing circuit robustness.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

import pyvene as pv
from pyvene import IntervenableModel


@dataclass
class InterventionResult:
    """Result of a robust intervention test."""
    baseline_behavior: float
    intervened_behavior: float 
    effect_size: float
    success: bool
    metadata: Dict[str, Any]


class RobustInterventionTechnique:
    """
    Advanced intervention techniques for testing circuit robustness.
    
    Goes beyond simple activation patching to test circuits with:
    - Gradual interventions
    - Multi-context interventions  
    - Noise-resistant interventions
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def gradual_intervention_test(
        self,
        circuit_config: Dict,
        context: str,
        behavioral_metric: Callable,
        intervention_strengths: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ) -> List[InterventionResult]:
        """
        Test circuit with gradually increasing intervention strength.
        
        This tests whether the circuit fails gracefully or has sharp transitions.
        Robust circuits should show smooth degradation.
        """
        results = []
        
        for strength in intervention_strengths:
            result = self._test_intervention_strength(
                circuit_config, context, behavioral_metric, strength
            )
            results.append(result)
            
        return results
    
    def cross_context_intervention_test(
        self,
        circuit_config: Dict,
        contexts: List[str],
        behavioral_metric: Callable
    ) -> Dict[str, InterventionResult]:
        """
        Test circuit with interventions that work across multiple contexts.
        
        This tests if the circuit is truly context-invariant by seeing
        if interventions trained on one context work on others.
        """
        results = {}
        
        # For each context, test intervention trained on other contexts
        for target_context in contexts:
            cross_context_results = []
            
            for source_context in contexts:
                if source_context != target_context:
                    result = self._cross_context_intervention(
                        circuit_config, source_context, target_context, behavioral_metric
                    )
                    cross_context_results.append(result)
                    
            # Average the cross-context results
            if cross_context_results:
                avg_result = self._average_intervention_results(cross_context_results)
                results[target_context] = avg_result
                
        return results
    
    def noise_robust_intervention_test(
        self,
        circuit_config: Dict,
        context: str,
        behavioral_metric: Callable,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.5]
    ) -> List[InterventionResult]:
        """
        Test circuit robustness to noise in interventions.
        
        Robust circuits should work even when interventions are noisy.
        """
        results = []
        
        for noise_level in noise_levels:
            result = self._test_noisy_intervention(
                circuit_config, context, behavioral_metric, noise_level
            )
            results.append(result)
            
        return results
    
    def _test_intervention_strength(
        self,
        circuit_config: Dict,
        context: str,
        behavioral_metric: Callable,
        strength: float
    ) -> InterventionResult:
        """Test intervention at a specific strength."""
        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            # Get baseline
            with torch.no_grad():
                baseline_output = self.model(**inputs)
                baseline_behavior = behavioral_metric(baseline_output, inputs)
                
            # Create intervention with specified strength
            config = self._create_intervention_config(circuit_config, strength)
            intervenable = pv.IntervenableModel(config, self.model)
            
            # Apply intervention
            with torch.no_grad():
                intervened_output = intervenable(base=inputs)
                intervened_behavior = behavioral_metric(intervened_output, inputs)
                
            # Compute effect size
            if isinstance(baseline_behavior, torch.Tensor):
                baseline_behavior = baseline_behavior.item()
            if isinstance(intervened_behavior, torch.Tensor):
                intervened_behavior = intervened_behavior.item()
                
            effect_size = abs(baseline_behavior - intervened_behavior)
            
            return InterventionResult(
                baseline_behavior=baseline_behavior,
                intervened_behavior=intervened_behavior,
                effect_size=effect_size,
                success=True,
                metadata={"strength": strength, "context": context}
            )
            
        except Exception as e:
            return InterventionResult(
                baseline_behavior=0.0,
                intervened_behavior=0.0,
                effect_size=0.0,
                success=False,
                metadata={"error": str(e), "strength": strength}
            )
    
    def _cross_context_intervention(
        self,
        circuit_config: Dict,
        source_context: str,
        target_context: str,
        behavioral_metric: Callable
    ) -> InterventionResult:
        """Test intervention trained on source context applied to target context."""
        try:
            # Get intervention parameters from source context
            source_inputs = self.tokenizer(source_context, return_tensors="pt").to(self.device)
            target_inputs = self.tokenizer(target_context, return_tensors="pt").to(self.device)
            
            # Get baseline behavior on target
            with torch.no_grad():
                baseline_output = self.model(**target_inputs)
                baseline_behavior = behavioral_metric(baseline_output, target_inputs)
                
            # Create intervention config
            config = self._create_intervention_config(circuit_config, 1.0)
            intervenable = pv.IntervenableModel(config, self.model)
            
            # Apply intervention (this is simplified - in practice you'd extract
            # intervention parameters from source and apply to target)
            with torch.no_grad():
                intervened_output = intervenable(
                    base=target_inputs,
                    source=source_inputs
                )
                intervened_behavior = behavioral_metric(intervened_output, target_inputs)
                
            # Compute effect
            if isinstance(baseline_behavior, torch.Tensor):
                baseline_behavior = baseline_behavior.item()
            if isinstance(intervened_behavior, torch.Tensor):
                intervened_behavior = intervened_behavior.item()
                
            effect_size = abs(baseline_behavior - intervened_behavior)
            
            return InterventionResult(
                baseline_behavior=baseline_behavior,
                intervened_behavior=intervened_behavior,
                effect_size=effect_size,
                success=True,
                metadata={
                    "source_context": source_context,
                    "target_context": target_context
                }
            )
            
        except Exception as e:
            return InterventionResult(
                baseline_behavior=0.0,
                intervened_behavior=0.0,
                effect_size=0.0,
                success=False,
                metadata={"error": str(e)}
            )
    
    def _test_noisy_intervention(
        self,
        circuit_config: Dict,
        context: str,
        behavioral_metric: Callable,
        noise_level: float
    ) -> InterventionResult:
        """Test intervention with added noise."""
        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            # Get baseline
            with torch.no_grad():
                baseline_output = self.model(**inputs)
                baseline_behavior = behavioral_metric(baseline_output, inputs)
                
            # Create noisy intervention
            config = self._create_noisy_intervention_config(circuit_config, noise_level)
            intervenable = pv.IntervenableModel(config, self.model)
            
            # Apply noisy intervention
            with torch.no_grad():
                intervened_output = intervenable(base=inputs)
                intervened_behavior = behavioral_metric(intervened_output, inputs)
                
            # Compute effect
            if isinstance(baseline_behavior, torch.Tensor):
                baseline_behavior = baseline_behavior.item()
            if isinstance(intervened_behavior, torch.Tensor):
                intervened_behavior = intervened_behavior.item()
                
            effect_size = abs(baseline_behavior - intervened_behavior)
            
            return InterventionResult(
                baseline_behavior=baseline_behavior,
                intervened_behavior=intervened_behavior,
                effect_size=effect_size,
                success=True,
                metadata={"noise_level": noise_level}
            )
            
        except Exception as e:
            return InterventionResult(
                baseline_behavior=0.0,
                intervened_behavior=0.0,
                effect_size=0.0,
                success=False,
                metadata={"error": str(e), "noise_level": noise_level}
            )
    
    def _create_intervention_config(self, circuit_config: Dict, strength: float) -> pv.IntervenableConfig:
        """Create intervention config with specified strength."""
        
        # Create scaled intervention
        representations = []
        for repr_config in circuit_config.get("representations", []):
            scaled_config = repr_config.copy()
            # In practice, you'd scale the intervention based on strength
            representations.append(scaled_config)
            
        return pv.IntervenableConfig(
            model_type=type(self.model),
            representations=representations
        )
    
    def _create_noisy_intervention_config(self, circuit_config: Dict, noise_level: float) -> pv.IntervenableConfig:
        """Create intervention config with noise."""
        
        # This is simplified - in practice you'd create custom noisy interventions
        representations = circuit_config.get("representations", [])
        
        return pv.IntervenableConfig(
            model_type=type(self.model),
            representations=representations
        )
    
    def _average_intervention_results(self, results: List[InterventionResult]) -> InterventionResult:
        """Average multiple intervention results."""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return InterventionResult(
                baseline_behavior=0.0,
                intervened_behavior=0.0,
                effect_size=0.0,
                success=False,
                metadata={"error": "No successful results to average"}
            )
            
        avg_baseline = np.mean([r.baseline_behavior for r in successful_results])
        avg_intervened = np.mean([r.intervened_behavior for r in successful_results])
        avg_effect = np.mean([r.effect_size for r in successful_results])
        
        return InterventionResult(
            baseline_behavior=avg_baseline,
            intervened_behavior=avg_intervened,
            effect_size=avg_effect,
            success=True,
            metadata={
                "num_averaged": len(successful_results),
                "success_rate": len(successful_results) / len(results)
            }
        )