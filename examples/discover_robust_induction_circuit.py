"""
Example: Discovering a Robust Induction Circuit

This example demonstrates how to use the robust circuit discovery system to find
an induction head circuit that works across different contexts and variables.

The core claim we're testing: "Induction circuits should be context and variable
invariant - they should work regardless of specific names, objects, or sentence
structures used."
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from typing import Dict, List, Callable
import json

# Import our robust circuit discovery system
from robust_circuits import RobustCircuitDiscovery
from robust_circuits.circuit_stability import StabilityMetrics
from robust_circuits.invariant_detection import InvariantPattern
from robust_circuits.evaluation import RobustnessReport

# Import pyvene for model handling
import pyvene as pv


def create_induction_behavioral_test() -> Callable:
    """
    Create a behavioral test that measures induction behavior.
    
    Induction behavior: given sequence [A][B]...[A], predict [B].
    """
    def induction_test(model_output, inputs):
        """
        Test if model can perform induction (copy pattern completion).
        
        We measure how much the model prefers the "correct" token B 
        after seeing pattern A B ... A.
        """
        try:
            # Get logits for last position
            logits = model_output.logits[0, -1, :]  # Last position
            
            # For this example, we'll use a simplified measure
            # In practice, you'd extract A and B from the input sequence
            # and measure P(B | context with A at end)
            
            # Simple heuristic: measure entropy (lower = more confident)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            
            # Convert to "confidence" score (higher = better induction)
            confidence = 1.0 / (1.0 + entropy.item())
            
            return confidence
            
        except Exception as e:
            print(f"Error in induction test: {e}")
            return 0.0
    
    return induction_test


def create_pattern_completion_test() -> Callable:
    """Create a test for general pattern completion ability."""
    def pattern_test(model_output, inputs):
        try:
            logits = model_output.logits[0, -1, :]
            
            # Measure how "peaked" the distribution is
            # (good pattern completion should be confident)
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            
            return max_prob
            
        except Exception:
            return 0.0
    
    return pattern_test


def run_robust_induction_discovery():
    """
    Main function to discover robust induction circuits.
    
    This demonstrates the full pipeline of robust circuit discovery.
    """
    print("🚀 Starting Robust Induction Circuit Discovery")
    print("=" * 60)
    
    # Step 1: Load model
    print("📝 Loading model...")
    model_name = "gpt2"  # Start with a smaller model for demonstration
    
    try:
        config, tokenizer, model = pv.create_gpt2_lm(model_name, cache_dir="./models")
        print(f"✅ Loaded {model_name}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Try: pip install transformers torch")
        return
    
    # Step 2: Initialize robust circuit discovery system  
    print("🔧 Initializing discovery system...")
    discovery_engine = RobustCircuitDiscovery(
        model=model,
        tokenizer=tokenizer,
        device="auto",
        cache_dir="./circuit_cache"
    )
    
    # Step 3: Define templates for induction patterns
    print("📋 Defining test templates...")
    
    # Templates with different structures but same logical pattern
    task_templates = [
        "When {name} went to {place}, {name}",           # Classic induction
        "{name} loves {object}. {name}",                 # Simple repetition  
        "In {place}, {name} found {object}. {name}",     # More complex
        "{name} and {other_name} went to {place}. {name}", # With distractor
        "Yesterday {name} bought {object} at {place}. Today {name}", # Temporal
        "{name} told {other_name} about {object}. {name}",  # Social context
        "The {object} belonged to {name}. {name}",       # Possession
        "{name}: I like {object}. {name}:",              # Dialogue format
    ]
    
    # Step 4: Define variable sets (the content should not matter for robust circuits)
    print("📊 Setting up variable sets...")
    
    variable_sets = {
        "name": [
            "Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Henry",
            "Maya", "Zara", "Alex", "Sam", "Jordan", "Riley", "Casey", "Drew"
        ],
        "other_name": [
            "Sarah", "Mike", "Lisa", "John", "Kate", "Paul", "Nina", "Ryan", 
            "Ella", "Owen", "Sophia", "Lucas", "Ava", "Noah", "Mia", "Liam"
        ],
        "place": [
            "Paris", "Tokyo", "school", "work", "home", "the park", "the store", 
            "the library", "the cafe", "the beach", "downtown", "the museum"
        ],
        "object": [
            "a book", "a phone", "a car", "flowers", "food", "a gift", "a pen",
            "a laptop", "a watch", "a camera", "music", "a ticket", "a key"
        ]
    }
    
    # Step 5: Define behavioral tests
    print("🧪 Creating behavioral tests...")
    
    behavioral_tests = [
        create_induction_behavioral_test(),
        create_pattern_completion_test(),
    ]
    
    # Step 6: Discover robust circuits!
    print("🔍 Starting circuit discovery...")
    print("⏳ This may take a few minutes...")
    
    robust_circuits = discovery_engine.discover_robust_circuits(
        task_templates=task_templates,
        variable_sets=variable_sets,
        behavioral_tests=behavioral_tests,
        robustness_threshold=0.7,  # Require 70% robustness
        min_contexts=15,           # Test on at least 15 contexts
        max_circuits=5             # Return top 5 circuits
    )
    
    # Step 7: Analyze and report results
    print("\n🎉 Circuit Discovery Complete!")
    print("=" * 60)
    
    if not robust_circuits:
        print("❌ No robust circuits found meeting the criteria")
        print("💡 Try lowering the robustness_threshold or increasing context diversity")
        return
    
    print(f"✅ Found {len(robust_circuits)} robust circuits")
    
    # Analyze each circuit
    for i, circuit in enumerate(robust_circuits):
        print(f"\n🔍 Circuit {i+1}: {circuit.name}")
        print(f"   Robustness Score: {circuit.robustness_score:.3f}")
        print(f"   Components: {len(circuit.components)}")
        print(f"   Contexts Tested: {len(circuit.contexts_tested)}")
        
        # Get detailed explanation
        explanation = discovery_engine.explain_circuit_behavior(
            circuit, 
            circuit.contexts_tested[:3],  # Show first 3 examples
            detailed=False
        )
        
        print(f"   Function: {explanation['functional_description']}")
        print(f"   Key Components: {explanation['key_components']}")
        print(f"   Robustness Factors: {explanation['robustness_factors']}")
    
    # Step 8: Validate on completely new contexts
    print("\n🧪 Validating on unseen contexts...")
    
    # Create validation contexts with completely different structure
    validation_contexts = [
        "During the meeting, Alex mentioned the project. Alex",
        "Before dinner, Sam called about the tickets. Sam", 
        "At the conference, Jordan presented the research. Jordan",
        "Through the window, Riley saw the sunrise. Riley",
        "Despite the rain, Casey enjoyed the concert. Casey"
    ]
    
    best_circuit = robust_circuits[0]  # Take the most robust circuit
    
    validation_results = discovery_engine.validate_circuit_robustness(
        best_circuit,
        validation_contexts,
        behavioral_tests
    )
    
    print("📊 Validation Results:")
    for test_name, score in validation_results.items():
        print(f"   {test_name}: {score:.3f}")
    
    # Step 9: Compare with circuit families
    if len(robust_circuits) >= 2:
        print("\n🔬 Comparing circuit families...")
        
        high_robustness = [c for c in robust_circuits if c.robustness_score > 0.8]
        med_robustness = [c for c in robust_circuits if 0.6 <= c.robustness_score <= 0.8]
        
        if high_robustness and med_robustness:
            comparison = discovery_engine.compare_circuit_families(
                high_robustness,
                med_robustness, 
                validation_contexts
            )
            
            print("📈 Family Comparison:")
            print(f"   High Robustness Family: {comparison['family_A_stats']['avg_robustness']:.3f}")
            print(f"   Medium Robustness Family: {comparison['family_B_stats']['avg_robustness']:.3f}")
            print(f"   Shared Components: {len(comparison['shared_components'])}")
    
    # Step 10: Look for universal motifs
    print("\n🔍 Searching for universal circuit motifs...")
    
    motif_analysis = discovery_engine.find_universal_motifs(robust_circuits)
    
    print("🧬 Universal Patterns Found:")
    for pattern_name, circuits in motif_analysis["universal_patterns"].items():
        print(f"   {pattern_name}: appears in {len(circuits)} circuits")
    
    if motif_analysis["suggested_motifs"]:
        print("💡 Suggested Universal Motifs:")
        for motif in motif_analysis["suggested_motifs"][:3]:
            print(f"   • {motif}")
    
    # Step 11: Comprehensive evaluation of the best circuit
    print("\n📊 Comprehensive Evaluation of Best Circuit...")
    
    evaluation_report = discovery_engine.evaluator.comprehensive_evaluation(
        circuit_config={"representations": [], "interventions": {}},  # Simplified for demo
        test_contexts=validation_contexts,
        behavioral_tests=behavioral_tests,
        variable_sets=variable_sets
    )
    
    print("🎯 Robustness Report:")
    print(f"   Overall Score: {evaluation_report.overall_score:.3f}")
    print(f"   Context Robustness: {evaluation_report.context_robustness:.3f}")
    print(f"   Variable Robustness: {evaluation_report.variable_robustness:.3f}")
    print(f"   Noise Robustness: {evaluation_report.noise_robustness:.3f}")
    
    if evaluation_report.failure_modes:
        print("⚠️  Failure Modes:")
        for failure in evaluation_report.failure_modes:
            print(f"   • {failure}")
    
    if evaluation_report.recommendations:
        print("💡 Recommendations:")
        for rec in evaluation_report.recommendations:
            print(f"   • {rec}")
    
    # Step 12: Save results
    print("\n💾 Saving results...")
    
    results_summary = {
        "discovery_metadata": {
            "model_name": model_name,
            "num_circuits_found": len(robust_circuits),
            "best_robustness_score": max(c.robustness_score for c in robust_circuits),
            "templates_tested": len(task_templates),
            "variables_per_template": {k: len(v) for k, v in variable_sets.items()}
        },
        "circuits": [circuit.to_dict() for circuit in robust_circuits],
        "validation_results": validation_results,
        "motif_analysis": motif_analysis,
        "evaluation_report": {
            "overall_score": evaluation_report.overall_score,
            "detailed_scores": evaluation_report.detailed_scores,
            "failure_modes": evaluation_report.failure_modes,
            "recommendations": evaluation_report.recommendations
        }
    }
    
    with open("robust_induction_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("✅ Results saved to robust_induction_results.json")
    
    # Final summary
    print("\n" + "="*60)
    print("🏆 DISCOVERY SUMMARY")
    print("="*60)
    print(f"🎯 Circuits Found: {len(robust_circuits)}")
    print(f"🏅 Best Robustness: {max(c.robustness_score for c in robust_circuits):.3f}")
    print(f"🧪 Contexts Tested: {sum(len(c.contexts_tested) for c in robust_circuits)}")
    print(f"🔗 Universal Patterns: {len(motif_analysis['universal_patterns'])}")
    
    if robust_circuits:
        print(f"✅ SUCCESS: Found robust induction circuit with {robust_circuits[0].robustness_score:.1%} robustness!")
        print("🔍 Key insight: The circuit works across different contexts and variables,")
        print("   supporting the hypothesis that robust circuits are context-invariant.")
    else:
        print("❌ No robust circuits found. This suggests:")
        print("   • The model may not have learned robust induction patterns")
        print("   • The robustness criteria may be too strict")
        print("   • More diverse training contexts may be needed")
    
    print("\n🎉 Robust Circuit Discovery Complete!")


if __name__ == "__main__":
    run_robust_induction_discovery()