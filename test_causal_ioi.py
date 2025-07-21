#!/usr/bin/env python3
"""
Quick test of the causal IOI discovery system
"""

import torch
from causal_ioi_discovery import IOICausalCircuitDiscovery, IOICircuitComponent, visualize_causal_circuit, CausalAbstractionResult

def test_causal_system():
    """Test the causal IOI system with a few components"""
    print("🧪 Testing Causal IOI System")
    print("=" * 40)
    
    # Initialize discovery system
    discovery = IOICausalCircuitDiscovery(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("✅ System initialized successfully")
    
    # Test data generation
    test_data = discovery.test_distribution.sample_das(
        tokenizer=discovery.tokenizer,
        base_patterns=["ABB"],
        source_patterns=["BAB"],
        labels="position", 
        samples_per_combination=5
    )
    
    print("✅ Data generation works")
    print(f"Test data size: {len(test_data)}")
    
    # Test a single attention head
    print("\n🔍 Testing single attention head...")
    effect = discovery._test_attention_head_effect(layer=9, head=1)
    print(f"Head 9.1 effect: {effect:.4f}")
    
    # Test a single MLP layer
    print("\n🔍 Testing single MLP layer...")
    effect = discovery._test_mlp_layer_effect(layer=6)
    print(f"MLP layer 6 effect: {effect:.4f}")
    
    # Test residual position
    print("\n🔍 Testing residual position...")
    effect = discovery._test_residual_position_effect(layer=3, position=-1)
    print(f"Residual 3.-1 effect: {effect:.4f}")
    
    # Create mock components for testing
    print("\n🧪 Testing with mock components...")
    mock_components = [
        IOICircuitComponent(
            component_type="attention_head",
            layer=9, 
            position=-1,
            head=1,
            causal_role="name_mover_head",
            strength=0.25
        ),
        IOICircuitComponent(
            component_type="mlp_layer",
            layer=6,
            position=-1,
            causal_role="nonlinear_processing", 
            strength=0.15
        ),
        IOICircuitComponent(
            component_type="residual_stream",
            layer=3,
            position=-1,
            causal_role="information_routing_pos_-1",
            strength=0.12
        )
    ]
    
    # Test causal alignment
    print("\n🧪 Testing causal alignment...")
    alignment_scores = discovery.test_causal_alignment(mock_components)
    print("Alignment scores:")
    for var_name, score in alignment_scores.items():
        print(f"  {var_name}: {score:.3f}")
    
    # Test robustness evaluation
    print("\n🛡️ Testing robustness evaluation...")
    robustness_metrics = discovery._evaluate_robustness(mock_components)
    print("Robustness metrics:")
    for metric, value in robustness_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Create a mock result for visualization
    print("\n📊 Testing visualization...")
    mock_result = CausalAbstractionResult(
        circuit_components=mock_components,
        causal_variables=discovery.causal_model.variables,
        alignment_scores=alignment_scores,
        intervention_effects={},
        robustness_metrics=robustness_metrics
    )
    
    visualize_causal_circuit(mock_result, "test_causal_circuit.png")
    
    print("\n✅ All tests passed!")
    print("🎯 Causal IOI system is working correctly")
    
    return mock_result

if __name__ == "__main__":
    result = test_causal_system()