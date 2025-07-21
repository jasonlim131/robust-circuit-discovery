#!/usr/bin/env python3
"""
Causal Abstraction System for IOI Circuit Discovery

This module implements a proper causal abstraction approach to discover 
circuits in the Indirect Object Identification (IOI) task, using discrete
causal variables as units of intervention rather than just layer-level flow.

Based on:
- Causal abstraction literature (Geiger et al.)
- IOI circuit analysis (Wang et al.)
- PyVene intervention framework
"""

import torch
import numpy as np
import json
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools

# PyVene imports
from pyvene import (
    IntervenableModel, 
    IntervenableConfig, 
    RepresentationConfig,
    LowRankRotatedSpaceIntervention,
    VanillaIntervention,
    CausalModel,
    create_gpt2
)

# Import simplified IOI utilities
from ioi_utils_simplified import (
    PromptDistribution, 
    NAMES, OBJECTS, PLACES, TEMPLATES,
    compute_metrics,
    calculate_loss,
    get_last_token
)

@dataclass
class CausalVariable:
    """Represents a discrete causal variable in the IOI task"""
    name: str
    description: str
    possible_values: List[str]
    dependencies: List[str]  # Parent variables
    neural_implementation: Dict[str, Any]  # How this is implemented neurally
    
@dataclass
class IOICircuitComponent:
    """Represents a component in the IOI circuit with causal role"""
    component_type: str  # "attention_head", "mlp_neuron", "residual_stream"
    layer: int
    position: int  # Token position
    head: Optional[int] = None  # For attention heads
    neuron_idx: Optional[int] = None  # For MLP neurons
    causal_role: str = ""  # E.g., "name_mover", "duplicate_token_head", etc.
    strength: float = 0.0  # Intervention effect strength
    
@dataclass
class CausalAbstractionResult:
    """Results from causal abstraction analysis"""
    circuit_components: List[IOICircuitComponent]
    causal_variables: Dict[str, CausalVariable] 
    alignment_scores: Dict[str, float]  # Variable -> neural implementation alignment
    intervention_effects: Dict[str, Dict[str, float]]  # Component -> Variable -> Effect
    robustness_metrics: Dict[str, float]
    
class IOICausalModel:
    """Defines the causal model for the IOI task"""
    
    def __init__(self):
        self.variables = self._define_causal_variables()
        self.causal_graph = self._build_causal_graph()
        
    def _define_causal_variables(self) -> Dict[str, CausalVariable]:
        """Define the discrete causal variables for IOI task"""
        variables = {
            # Input variables
            "subject_token": CausalVariable(
                name="subject_token",
                description="The subject name token (e.g., 'John')",
                possible_values=NAMES[:20],  # Subset for tractability
                dependencies=[],
                neural_implementation={"positions": [2], "layers": "input"}  # Subject is 3rd token
            ),
            
            "io_token": CausalVariable(
                name="io_token", 
                description="The indirect object name token",
                possible_values=NAMES[:20],
                dependencies=[],
                neural_implementation={"positions": [0, 1], "layers": "input"}  # IO in positions 0 or 1
            ),
            
            "duplicate_position": CausalVariable(
                name="duplicate_position",
                description="Which position contains the duplicate of the subject",
                possible_values=["position_0", "position_1"],
                dependencies=["subject_token", "io_token"],
                neural_implementation={"positions": [0, 1], "layers": "all"}
            ),
            
            # Intermediate computational variables
            "previous_token_head": CausalVariable(
                name="previous_token_head",
                description="Attention head that identifies previous token",
                possible_values=["active", "inactive"],
                dependencies=["subject_token"],
                neural_implementation={"component_type": "attention", "layers": [7, 8, 9]}
            ),
            
            "duplicate_token_head": CausalVariable(
                name="duplicate_token_head", 
                description="Attention head that identifies duplicate tokens",
                possible_values=["active", "inactive"],
                dependencies=["subject_token", "io_token", "duplicate_position"],
                neural_implementation={"component_type": "attention", "layers": [7, 8, 9, 10]}
            ),
            
            "induction_head": CausalVariable(
                name="induction_head",
                description="Induction head that copies from duplicate position",
                possible_values=["active", "inactive"], 
                dependencies=["previous_token_head", "duplicate_token_head"],
                neural_implementation={"component_type": "attention", "layers": [9, 10, 11]}
            ),
            
            "name_mover_head": CausalVariable(
                name="name_mover_head",
                description="Head that moves name information to final position",
                possible_values=["moves_subject", "moves_io", "inactive"],
                dependencies=["induction_head", "duplicate_position"],
                neural_implementation={"component_type": "attention", "layers": [9, 10, 11]}
            ),
            
            # Output variable
            "prediction": CausalVariable(
                name="prediction",
                description="Final model prediction",
                possible_values=["correct_io", "incorrect_subject"],
                dependencies=["name_mover_head"],
                neural_implementation={"positions": [-1], "layers": "output"}
            )
        }
        return variables
        
    def _build_causal_graph(self) -> nx.DiGraph:
        """Build directed graph of causal dependencies"""
        G = nx.DiGraph()
        
        # Add nodes
        for var_name in self.variables:
            G.add_node(var_name)
            
        # Add edges based on dependencies
        for var_name, var in self.variables.items():
            for parent in var.dependencies:
                G.add_edge(parent, var_name)
                
        return G
        
    def get_intervention_targets(self) -> Dict[str, List[str]]:
        """Get components that should be targeted for each causal variable"""
        targets = {}
        
        for var_name, var in self.variables.items():
            impl = var.neural_implementation
            components = []
            
            if impl.get("component_type") == "attention":
                # Target attention heads in specified layers
                layers = impl.get("layers", [])
                for layer in layers:
                    for head in range(12):  # GPT-2 has 12 heads per layer
                        components.append(f"blocks.{layer}.attn.c_attn.{head}")
                        
            elif impl.get("component_type") == "mlp":
                # Target MLP neurons
                layers = impl.get("layers", [])
                for layer in layers:
                    components.append(f"blocks.{layer}.mlp")
                    
            elif "positions" in impl:
                # Target residual stream at specific positions
                positions = impl["positions"]
                layers = impl.get("layers", "all")
                if layers == "all":
                    layers = list(range(12))
                elif layers == "input":
                    layers = [0]
                elif layers == "output": 
                    layers = [11]
                    
                for layer in layers:
                    for pos in positions:
                        components.append(f"blocks.{layer}.hook_resid_post.{pos}")
                        
            targets[var_name] = components
            
        return targets


class IOICausalCircuitDiscovery:
    """Main class for discovering IOI circuits using causal abstraction"""
    
    def __init__(self, model_name="gpt2", device="cuda"):
        self.device = device
        self.config, self.tokenizer, self.model = create_gpt2(model_name)
        self.model = self.model.to(device)
        
        # Set padding token for GPT-2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.causal_model = IOICausalModel()
        self.intervention_targets = self.causal_model.get_intervention_targets()
        
        # Setup data distributions 
        self.train_distribution = PromptDistribution(
            names=NAMES[:15],
            objects=OBJECTS[:10], 
            places=PLACES[:10],
            templates=TEMPLATES[:2]
        )
        
        self.test_distribution = PromptDistribution(
            names=NAMES[15:25],
            objects=OBJECTS[:10],  # Use first 10 since we only have 10 total
            places=PLACES[:10],    # Use first 10 since we only have 10 total
            templates=TEMPLATES[2:]  # Use remaining templates
        )
        
    def discover_neuron_level_components(self) -> List[IOICircuitComponent]:
        """Discover specific neuron-level components for each causal variable"""
        components = []
        
        # Scan attention heads
        print("🔍 Scanning attention heads...")
        attention_components = self._scan_attention_heads()
        components.extend(attention_components)
        
        # Scan MLP neurons
        print("🔍 Scanning MLP neurons...")
        mlp_components = self._scan_mlp_neurons()
        components.extend(mlp_components)
        
        # Scan residual stream positions
        print("🔍 Scanning residual stream positions...")
        residual_components = self._scan_residual_positions()
        components.extend(residual_components)
        
        return components
        
    def _scan_attention_heads(self) -> List[IOICircuitComponent]:
        """Systematically scan all attention heads to find functionally relevant ones"""
        components = []
        
        for layer in range(self.config.n_layer):
            for head in range(self.config.n_head):
                print(f"  Analyzing attention head {layer}.{head}")
                
                # Test this head's effect on IOI task
                effect_strength = self._test_attention_head_effect(layer, head)
                
                if effect_strength > 0.1:  # Threshold for significance
                    # Determine causal role
                    causal_role = self._classify_attention_head_role(layer, head, effect_strength)
                    
                    component = IOICircuitComponent(
                        component_type="attention_head",
                        layer=layer,
                        position=-1,  # Heads operate across positions
                        head=head,
                        causal_role=causal_role,
                        strength=effect_strength
                    )
                    components.append(component)
                    
        return components
        
    def _test_attention_head_effect(self, layer: int, head: int) -> float:
        """Test the effect of intervening on a specific attention head"""
        
        # Generate test data
        test_data = self.test_distribution.sample_das(
            tokenizer=self.tokenizer,
            base_patterns=["ABB"],
            source_patterns=["BAB"], 
            labels="position",
            samples_per_combination=20
        )
        
        # Create intervention config for this head
        config = IntervenableConfig([
            RepresentationConfig(
                layer=layer,
                component="head_attention_value_output", 
                unit="h.pos",
                low_rank_dimension=1
            )
        ])
        
        try:
            intervenable = IntervenableModel(config, self.model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            # Test intervention effect
            total_effect = 0.0
            num_batches = 0
            
            for batch_data in test_data.batches(batch_size=10):
                base_inputs = batch_data.base.tokens
                source_inputs = batch_data.source.tokens
                labels = batch_data.patched_answer_tokens[:, 0]
                
                # Move to device
                for k, v in base_inputs.items():
                    if isinstance(v, torch.Tensor):
                        base_inputs[k] = v.to(self.device)
                for k, v in source_inputs.items():
                    if isinstance(v, torch.Tensor):
                        source_inputs[k] = v.to(self.device)
                labels = labels.to(self.device)
                
                # Run intervention
                _, counterfactual = intervenable(
                    base_inputs,
                    [source_inputs],
                    {"sources->base": ([[[head] * len(labels), [[-1]] * len(labels)]], 
                                     [[[head] * len(labels), [[-1]] * len(labels)]])}
                )
                
                # Compute effect (change in prediction accuracy)
                baseline_logits = self.model(base_inputs["input_ids"]).logits
                intervention_logits = counterfactual.logits
                
                baseline_logits_last = get_last_token(baseline_logits, base_inputs["attention_mask"])
                intervention_logits_last = get_last_token(intervention_logits, base_inputs["attention_mask"])
                
                baseline_probs = torch.softmax(baseline_logits_last, dim=-1)
                intervention_probs = torch.softmax(intervention_logits_last, dim=-1)
                
                # Effect is change in probability mass on target tokens
                effect = torch.abs(intervention_probs[torch.arange(len(labels)), labels] - 
                                 baseline_probs[torch.arange(len(labels)), labels]).mean().item()
                
                total_effect += effect
                num_batches += 1
                
            return total_effect / num_batches if num_batches > 0 else 0.0
            
        except Exception as e:
            print(f"    Error testing head {layer}.{head}: {e}")
            return 0.0
            
    def _classify_attention_head_role(self, layer: int, head: int, strength: float) -> str:
        """Classify the causal role of an attention head based on its pattern"""
        
        # Heuristic classification based on layer and effect strength
        if layer <= 3:
            return "early_processing"
        elif layer <= 7: 
            if strength > 0.3:
                return "duplicate_token_head"
            else:
                return "previous_token_head"
        elif layer <= 10:
            if strength > 0.4:
                return "induction_head" 
            else:
                return "name_mover_head"
        else:
            return "output_head"
            
    def _scan_mlp_neurons(self) -> List[IOICircuitComponent]:
        """Scan MLP neurons for causal relevance"""
        components = []
        
        # Sample a subset of neurons due to computational constraints
        for layer in range(0, self.config.n_layer, 2):  # Every other layer
            print(f"  Analyzing MLP layer {layer}")
            
            # Test effect of entire MLP layer first
            effect_strength = self._test_mlp_layer_effect(layer)
            
            if effect_strength > 0.05:  # Threshold for MLP layers
                component = IOICircuitComponent(
                    component_type="mlp_layer",
                    layer=layer,
                    position=-1,
                    causal_role="nonlinear_processing",
                    strength=effect_strength
                )
                components.append(component)
                
        return components
        
    def _test_mlp_layer_effect(self, layer: int) -> float:
        """Test the effect of intervening on an MLP layer"""
        
        # Generate test data
        test_data = self.test_distribution.sample_das(
            tokenizer=self.tokenizer,
            base_patterns=["ABB"],
            source_patterns=["BAB"],
            labels="position", 
            samples_per_combination=10
        )
        
        # Create intervention config for MLP
        config = IntervenableConfig([
            RepresentationConfig(
                layer=layer,
                component="mlp_output",
                unit="pos",
                low_rank_dimension=8  # Higher rank for MLPs
            )
        ])
        
        try:
            intervenable = IntervenableModel(config, self.model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            total_effect = 0.0
            num_batches = 0
            
            for batch_data in test_data.batches(batch_size=5):
                base_inputs = batch_data.base.tokens
                source_inputs = batch_data.source.tokens
                labels = batch_data.patched_answer_tokens[:, 0]
                
                # Move to device
                for k, v in base_inputs.items():
                    if isinstance(v, torch.Tensor):
                        base_inputs[k] = v.to(self.device)
                for k, v in source_inputs.items():
                    if isinstance(v, torch.Tensor):
                        source_inputs[k] = v.to(self.device)
                labels = labels.to(self.device)
                
                # Run intervention
                _, counterfactual = intervenable(
                    base_inputs,
                    [source_inputs], 
                    {"sources->base": ([[[-1]] * len(labels)], [[[-1]] * len(labels)])}
                )
                
                # Compute effect
                baseline_logits = self.model(base_inputs["input_ids"]).logits
                intervention_logits = counterfactual.logits
                
                baseline_logits_last = get_last_token(baseline_logits, base_inputs["attention_mask"])
                intervention_logits_last = get_last_token(intervention_logits, base_inputs["attention_mask"])
                
                baseline_probs = torch.softmax(baseline_logits_last, dim=-1)
                intervention_probs = torch.softmax(intervention_logits_last, dim=-1)
                
                effect = torch.abs(intervention_probs[torch.arange(len(labels)), labels] -
                                 baseline_probs[torch.arange(len(labels)), labels]).mean().item()
                
                total_effect += effect
                num_batches += 1
                
            return total_effect / num_batches if num_batches > 0 else 0.0
            
        except Exception as e:
            print(f"    Error testing MLP layer {layer}: {e}")
            return 0.0
            
    def _scan_residual_positions(self) -> List[IOICircuitComponent]:
        """Scan residual stream at key token positions"""
        components = []
        
        key_positions = [0, 1, 2, -1]  # First names, subject, final prediction
        
        for layer in range(0, self.config.n_layer, 3):  # Every 3rd layer
            for pos in key_positions:
                print(f"  Analyzing residual position {layer}.{pos}")
                
                effect_strength = self._test_residual_position_effect(layer, pos)
                
                if effect_strength > 0.08:
                    component = IOICircuitComponent(
                        component_type="residual_stream",
                        layer=layer,
                        position=pos,
                        causal_role=f"information_routing_pos_{pos}",
                        strength=effect_strength
                    )
                    components.append(component)
                    
        return components
        
    def _test_residual_position_effect(self, layer: int, position: int) -> float:
        """Test effect of intervening on residual stream at specific position"""
        
        # Generate test data
        test_data = self.test_distribution.sample_das(
            tokenizer=self.tokenizer,
            base_patterns=["ABB"],
            source_patterns=["BAB"],
            labels="position",
            samples_per_combination=8
        )
        
        # Create intervention config
        config = IntervenableConfig([
            RepresentationConfig(
                layer=layer,
                component="block_output",
                unit="pos", 
                low_rank_dimension=4
            )
        ])
        
        try:
            intervenable = IntervenableModel(config, self.model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            total_effect = 0.0
            num_batches = 0
            
            for batch_data in test_data.batches(batch_size=4):
                base_inputs = batch_data.base.tokens
                source_inputs = batch_data.source.tokens
                labels = batch_data.patched_answer_tokens[:, 0]
                
                # Move to device
                for k, v in base_inputs.items():
                    if isinstance(v, torch.Tensor):
                        base_inputs[k] = v.to(self.device)
                for k, v in source_inputs.items():
                    if isinstance(v, torch.Tensor):
                        source_inputs[k] = v.to(self.device)
                labels = labels.to(self.device)
                
                # Run intervention
                _, counterfactual = intervenable(
                    base_inputs,
                    [source_inputs],
                    {"sources->base": ([[[position]] * len(labels)], [[[position]] * len(labels)])}
                )
                
                # Compute effect
                baseline_logits = self.model(base_inputs["input_ids"]).logits
                intervention_logits = counterfactual.logits
                
                baseline_logits_last = get_last_token(baseline_logits, base_inputs["attention_mask"])
                intervention_logits_last = get_last_token(intervention_logits, base_inputs["attention_mask"])
                
                baseline_probs = torch.softmax(baseline_logits_last, dim=-1)
                intervention_probs = torch.softmax(intervention_logits_last, dim=-1)
                
                effect = torch.abs(intervention_probs[torch.arange(len(labels)), labels] -
                                 baseline_probs[torch.arange(len(labels)), labels]).mean().item()
                
                total_effect += effect
                num_batches += 1
                
            return total_effect / num_batches if num_batches > 0 else 0.0
            
        except Exception as e:
            print(f"    Error testing residual {layer}.{position}: {e}")
            return 0.0
    
    def test_causal_alignment(self, components: List[IOICircuitComponent]) -> Dict[str, float]:
        """Test how well discovered components align with causal variables"""
        alignment_scores = {}
        
        for var_name, variable in self.causal_model.variables.items():
            print(f"🧪 Testing alignment for causal variable: {var_name}")
            
            # Find components that should implement this variable
            relevant_components = [c for c in components 
                                 if self._component_matches_variable(c, variable)]
            
            if not relevant_components:
                alignment_scores[var_name] = 0.0
                continue
                
            # Test if intervening on these components has the expected causal effect
            score = self._test_variable_implementation(variable, relevant_components)
            alignment_scores[var_name] = score
            
        return alignment_scores
        
    def _component_matches_variable(self, component: IOICircuitComponent, 
                                  variable: CausalVariable) -> bool:
        """Check if a component could implement a causal variable"""
        impl = variable.neural_implementation
        
        # Check component type matches
        if "component_type" in impl:
            if impl["component_type"] == "attention" and component.component_type != "attention_head":
                return False
            if impl["component_type"] == "mlp" and "mlp" not in component.component_type:
                return False
                
        # Check layer range
        if "layers" in impl:
            layers = impl["layers"]
            if isinstance(layers, list) and component.layer not in layers:
                return False
                
        # Check positions
        if "positions" in impl:
            positions = impl["positions"]
            if component.position not in positions and component.position != -1:
                return False
                
        return True
        
    def _test_variable_implementation(self, variable: CausalVariable, 
                                    components: List[IOICircuitComponent]) -> float:
        """Test how well components implement the causal variable"""
        
        # Create test scenarios that should activate this variable differently
        if variable.name == "duplicate_position":
            # Test ABB vs BAB patterns
            base_patterns = ["ABB"]
            source_patterns = ["BAB"] 
        elif variable.name in ["previous_token_head", "duplicate_token_head", "induction_head"]:
            # Test attention-based variables with position changes
            base_patterns = ["ABB", "BAB"]
            source_patterns = ["ABC", "BCA"]  # Break the duplication
        else:
            # Default test 
            base_patterns = ["ABB"]
            source_patterns = ["BAB"]
            
        test_data = self.test_distribution.sample_das(
            tokenizer=self.tokenizer,
            base_patterns=base_patterns,
            source_patterns=source_patterns,
            labels="position",
            samples_per_combination=5
        )
        
        total_score = 0.0
        num_tests = 0
        
        # Test each component's contribution to the variable
        for component in components:
            try:
                score = self._test_component_variable_effect(component, test_data)
                total_score += score
                num_tests += 1
            except Exception as e:
                print(f"    Error testing component {component}: {e}")
                
        return total_score / num_tests if num_tests > 0 else 0.0
        
    def _test_component_variable_effect(self, component: IOICircuitComponent, 
                                      test_data) -> float:
        """Test a specific component's effect on variable behavior"""
        
        # Create intervention config for this component
        if component.component_type == "attention_head":
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=component.layer,
                    component="head_attention_value_output",
                    unit="h.pos",
                    low_rank_dimension=1
                )
            ])
        elif "mlp" in component.component_type:
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=component.layer,
                    component="mlp_output", 
                    unit="pos",
                    low_rank_dimension=4
                )
            ])
        else:  # residual stream
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=component.layer,
                    component="block_output",
                    unit="pos",
                    low_rank_dimension=2
                )
            ])
            
        intervenable = IntervenableModel(config, self.model)
        intervenable.set_device(self.device)
        intervenable.disable_model_gradients()
        
        effects = []
        
        for batch_data in test_data.batches(batch_size=3):
            base_inputs = batch_data.base.tokens
            source_inputs = batch_data.source.tokens
            labels = batch_data.patched_answer_tokens[:, 0]
            
            # Move to device
            for k, v in base_inputs.items():
                if isinstance(v, torch.Tensor):
                    base_inputs[k] = v.to(self.device)
            for k, v in source_inputs.items():
                if isinstance(v, torch.Tensor):
                    source_inputs[k] = v.to(self.device)
            labels = labels.to(self.device)
            
            # Configure intervention based on component type
            if component.component_type == "attention_head":
                intervention_spec = {
                    "sources->base": ([[[component.head] * len(labels), [[-1]] * len(labels)]], 
                                    [[[component.head] * len(labels), [[-1]] * len(labels)]])
                }
            else:
                pos = component.position if component.position != -1 else -1
                intervention_spec = {
                    "sources->base": ([[[pos]] * len(labels)], [[[pos]] * len(labels)])
                }
            
            # Run intervention
            _, counterfactual = intervenable(base_inputs, [source_inputs], intervention_spec)
            
            # Compute effect
            baseline_logits = self.model(base_inputs["input_ids"]).logits
            intervention_logits = counterfactual.logits
            
            baseline_logits_last = get_last_token(baseline_logits, base_inputs["attention_mask"])
            intervention_logits_last = get_last_token(intervention_logits, base_inputs["attention_mask"])
            
            baseline_probs = torch.softmax(baseline_logits_last, dim=-1)
            intervention_probs = torch.softmax(intervention_logits_last, dim=-1)
            
            effect = torch.abs(intervention_probs[torch.arange(len(labels)), labels] -
                             baseline_probs[torch.arange(len(labels)), labels]).mean().item()
            effects.append(effect)
            
        return np.mean(effects) if effects else 0.0
    
    def run_full_discovery(self) -> CausalAbstractionResult:
        """Run complete causal abstraction discovery pipeline"""
        print("🚀 Starting IOI Causal Circuit Discovery...")
        
        # Step 1: Discover neuron-level components
        print("\n🔍 Phase 1: Discovering neuron-level circuit components...")
        components = self.discover_neuron_level_components()
        print(f"Found {len(components)} significant components")
        
        # Step 2: Test causal alignment
        print("\n🧪 Phase 2: Testing causal variable alignment...")
        alignment_scores = self.test_causal_alignment(components)
        
        # Step 3: Compute intervention effects
        print("\n⚡ Phase 3: Computing intervention effects...")
        intervention_effects = self._compute_intervention_effects(components)
        
        # Step 4: Evaluate robustness
        print("\n🛡️ Phase 4: Evaluating circuit robustness...")
        robustness_metrics = self._evaluate_robustness(components)
        
        result = CausalAbstractionResult(
            circuit_components=components,
            causal_variables=self.causal_model.variables,
            alignment_scores=alignment_scores,
            intervention_effects=intervention_effects,
            robustness_metrics=robustness_metrics
        )
        
        print("\n✅ Discovery complete!")
        return result
        
    def _compute_intervention_effects(self, components: List[IOICircuitComponent]) -> Dict[str, Dict[str, float]]:
        """Compute effect matrix: components x causal variables"""
        effects = {}
        
        for component in components:
            component_key = f"{component.component_type}_{component.layer}"
            if component.head is not None:
                component_key += f"_h{component.head}"
            if component.position != -1:
                component_key += f"_pos{component.position}"
                
            effects[component_key] = {}
            
            for var_name in self.causal_model.variables:
                # Simplified effect computation
                if self._component_matches_variable(component, self.causal_model.variables[var_name]):
                    effects[component_key][var_name] = component.strength
                else:
                    effects[component_key][var_name] = 0.0
                    
        return effects
        
    def _evaluate_robustness(self, components: List[IOICircuitComponent]) -> Dict[str, float]:
        """Evaluate robustness of the discovered circuit"""
        
        # Test on held-out templates and names
        holdout_distribution = PromptDistribution(
            names=NAMES[25:30],  # Only have 32 names total
            objects=OBJECTS[:5],  # Use subset since we only have 10 total
            places=PLACES[:5],    # Use subset since we only have 10 total  
            templates=TEMPLATES[3:]  # Use remaining templates
        )
        
        holdout_data = holdout_distribution.sample_das(
            tokenizer=self.tokenizer,
            base_patterns=["ABB", "BAB"],
            source_patterns=["ABB", "BAB"],
            labels="position",
            samples_per_combination=10
        )
        
        # Compute baseline performance
        baseline_acc = self._compute_baseline_accuracy(holdout_data)
        
        # Test circuit performance by ablating discovered components
        circuit_performance = self._test_circuit_performance(components, holdout_data)
        
        return {
            "baseline_accuracy": baseline_acc,
            "circuit_performance": circuit_performance,
            "robustness_score": circuit_performance / baseline_acc if baseline_acc > 0 else 0.0,
            "num_components": len(components),
            "component_diversity": len(set(c.component_type for c in components)) / 3.0  # 3 types
        }
        
    def _compute_baseline_accuracy(self, test_data) -> float:
        """Compute baseline model accuracy without interventions"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in test_data.batches(batch_size=10):
                base_inputs = batch_data.base.tokens
                labels = batch_data.patched_answer_tokens[:, 0]
                
                for k, v in base_inputs.items():
                    if isinstance(v, torch.Tensor):
                        base_inputs[k] = v.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(base_inputs["input_ids"]).logits
                logits_last = get_last_token(logits, base_inputs["attention_mask"])
                predictions = torch.argmax(logits_last, dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += len(labels)
                
        return correct / total if total > 0 else 0.0
        
    def _test_circuit_performance(self, components: List[IOICircuitComponent], 
                                test_data) -> float:
        """Test performance when circuit components are ablated"""
        
        # For simplicity, test the strongest component
        if not components:
            return 0.0
            
        strongest_component = max(components, key=lambda c: c.strength)
        
        # Create ablation intervention (zero out the component)
        if strongest_component.component_type == "attention_head":
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=strongest_component.layer,
                    component="head_attention_value_output",
                    unit="h.pos",
                    low_rank_dimension=1
                )
            ])
        else:
            config = IntervenableConfig([
                RepresentationConfig(
                    layer=strongest_component.layer,
                    component="block_output",
                    unit="pos",
                    low_rank_dimension=1
                )
            ])
            
        try:
            intervenable = IntervenableModel(config, self.model)
            intervenable.set_device(self.device)
            intervenable.disable_model_gradients()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data in test_data.batches(batch_size=5):
                    base_inputs = batch_data.base.tokens
                    labels = batch_data.patched_answer_tokens[:, 0]
                    
                    for k, v in base_inputs.items():
                        if isinstance(v, torch.Tensor):
                            base_inputs[k] = v.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero intervention (ablation)
                    zero_source = {k: torch.zeros_like(v) for k, v in base_inputs.items()}
                    
                    if strongest_component.component_type == "attention_head":
                        intervention_spec = {
                            "sources->base": ([[[strongest_component.head] * len(labels), [[-1]] * len(labels)]], 
                                            [[[strongest_component.head] * len(labels), [[-1]] * len(labels)]])
                        }
                    else:
                        intervention_spec = {
                            "sources->base": ([[[-1]] * len(labels)], [[[-1]] * len(labels)])
                        }
                    
                    _, counterfactual = intervenable(base_inputs, [zero_source], intervention_spec)
                    
                    counterfactual_logits_last = get_last_token(counterfactual.logits, base_inputs["attention_mask"])
                    predictions = torch.argmax(counterfactual_logits_last, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += len(labels)
                    
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            print(f"Error in circuit performance test: {e}")
            return 0.0


def visualize_causal_circuit(result: CausalAbstractionResult, save_path: str = "causal_ioi_circuit.png"):
    """Visualize the discovered causal circuit"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Causal graph
    G = nx.DiGraph()
    for var_name, var in result.causal_variables.items():
        G.add_node(var_name)
        for parent in var.dependencies:
            G.add_edge(parent, var_name)
            
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, arrows=True)
    ax1.set_title("Causal Variable Graph")
    
    # 2. Component types distribution
    component_types = [c.component_type for c in result.circuit_components]
    type_counts = {t: component_types.count(t) for t in set(component_types)}
    ax2.bar(type_counts.keys(), type_counts.values())
    ax2.set_title("Circuit Component Types")
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Alignment scores
    variables = list(result.alignment_scores.keys())
    scores = list(result.alignment_scores.values())
    ax3.barh(variables, scores)
    ax3.set_title("Causal Variable Alignment Scores")
    ax3.set_xlabel("Alignment Score")
    
    # 4. Layer distribution of components
    layers = [c.layer for c in result.circuit_components]
    ax4.hist(layers, bins=range(0, 13), alpha=0.7, edgecolor='black')
    ax4.set_title("Component Distribution by Layer")
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Number of Components")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Causal circuit visualization saved to {save_path}")


def main():
    """Main function to run IOI causal circuit discovery"""
    
    print("🎯 IOI Causal Circuit Discovery")
    print("=" * 50)
    
    # Initialize discovery system
    discovery = IOICausalCircuitDiscovery(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Run full discovery pipeline
    result = discovery.run_full_discovery()
    
    # Print summary
    print("\n📊 DISCOVERY SUMMARY")
    print("=" * 30)
    print(f"Components found: {len(result.circuit_components)}")
    print(f"Causal variables: {len(result.causal_variables)}")
    print(f"Average alignment score: {np.mean(list(result.alignment_scores.values())):.3f}")
    print(f"Circuit robustness: {result.robustness_metrics['robustness_score']:.3f}")
    
    print("\n🔍 TOP COMPONENTS:")
    top_components = sorted(result.circuit_components, key=lambda c: c.strength, reverse=True)[:5]
    for i, comp in enumerate(top_components):
        print(f"{i+1}. {comp.component_type} L{comp.layer} "
              f"{'H'+str(comp.head) if comp.head is not None else ''} "
              f"- {comp.causal_role} (strength: {comp.strength:.3f})")
              
    print("\n🧪 CAUSAL ALIGNMENT:")
    for var_name, score in result.alignment_scores.items():
        print(f"  {var_name}: {score:.3f}")
        
    # Visualize results
    visualize_causal_circuit(result)
    
    # Save results
    results_dict = {
        "components": [
            {
                "type": c.component_type,
                "layer": c.layer,
                "position": c.position,
                "head": c.head,
                "neuron_idx": c.neuron_idx,
                "causal_role": c.causal_role,
                "strength": c.strength
            }
            for c in result.circuit_components
        ],
        "alignment_scores": result.alignment_scores,
        "robustness_metrics": result.robustness_metrics,
        "intervention_effects": result.intervention_effects
    }
    
    with open("causal_ioi_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
        
    print("\n💾 Results saved to causal_ioi_results.json")
    print("✅ Causal IOI circuit discovery complete!")


if __name__ == "__main__":
    main()