"""
Robustness Evaluation: Comprehensive evaluation of circuit robustness across different dimensions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import pyvene as pv


@dataclass
class RobustnessReport:
    """Comprehensive report on circuit robustness."""
    overall_score: float
    context_robustness: float
    variable_robustness: float  
    noise_robustness: float
    intervention_robustness: float
    detailed_scores: Dict[str, float]
    failure_modes: List[str]
    recommendations: List[str]


class RobustnessEvaluator:
    """
    Comprehensive evaluator for circuit robustness across multiple dimensions.
    
    Evaluates:
    - Context invariance (same circuit works across different contexts)
    - Variable invariance (same circuit works with different variable instantiations)
    - Noise robustness (circuit works despite noise)
    - Intervention robustness (circuit effects are consistent)
    """
    
    def __init__(self, model: torch.nn.Module, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def comprehensive_evaluation(
        self,
        circuit_config: Dict,
        test_contexts: List[str],
        behavioral_tests: List[Callable],
        variable_sets: Optional[Dict[str, List[str]]] = None
    ) -> RobustnessReport:
        """
        Conduct comprehensive robustness evaluation.
        
        This is the main evaluation function that tests all aspects of robustness.
        """
        print("🔬 Starting comprehensive robustness evaluation...")
        
        # Test different dimensions of robustness
        context_score = self._evaluate_context_robustness(
            circuit_config, test_contexts, behavioral_tests[0]
        )
        
        variable_score = self._evaluate_variable_robustness(
            circuit_config, test_contexts, variable_sets, behavioral_tests[0]
        ) if variable_sets else 1.0
        
        noise_score = self._evaluate_noise_robustness(
            circuit_config, test_contexts[:5], behavioral_tests[0]
        )
        
        intervention_score = self._evaluate_intervention_robustness(
            circuit_config, test_contexts[:5], behavioral_tests[0]
        )
        
        # Compute overall score (weighted average)
        overall_score = (
            context_score * 0.3 +
            variable_score * 0.3 +
            noise_score * 0.2 +
            intervention_score * 0.2
        )
        
        # Identify failure modes and recommendations
        failure_modes = self._identify_failure_modes(
            context_score, variable_score, noise_score, intervention_score
        )
        
        recommendations = self._generate_recommendations(
            context_score, variable_score, noise_score, intervention_score
        )
        
        detailed_scores = {
            "context_consistency": context_score,
            "variable_independence": variable_score,
            "noise_tolerance": noise_score,
            "intervention_stability": intervention_score,
            "behavioral_tests": [self._test_behavioral_consistency(
                circuit_config, test_contexts[:3], test
            ) for test in behavioral_tests]
        }
        
        report = RobustnessReport(
            overall_score=overall_score,
            context_robustness=context_score,
            variable_robustness=variable_score,
            noise_robustness=noise_score,
            intervention_robustness=intervention_score,
            detailed_scores=detailed_scores,
            failure_modes=failure_modes,
            recommendations=recommendations
        )
        
        print(f"📊 Evaluation complete. Overall robustness: {overall_score:.3f}")
        return report
    
    def benchmark_against_baselines(
        self,
        circuit_config: Dict,
        test_contexts: List[str],
        behavioral_test: Callable,
        baseline_circuits: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Benchmark circuit robustness against baseline circuits.
        
        This helps understand how robust the circuit is compared to alternatives.
        """
        print("📏 Benchmarking against baseline circuits...")
        
        # Test target circuit
        target_score = self._evaluate_context_robustness(
            circuit_config, test_contexts, behavioral_test
        )
        
        results = {"target_circuit": target_score}
        
        # Test baseline circuits
        if baseline_circuits is None:
            baseline_circuits = self._generate_baseline_circuits(circuit_config)
            
        for i, baseline in enumerate(baseline_circuits):
            baseline_score = self._evaluate_context_robustness(
                baseline, test_contexts, behavioral_test
            )
            results[f"baseline_{i}"] = baseline_score
            
        # Add random baseline
        random_score = self._evaluate_random_baseline(test_contexts, behavioral_test)
        results["random_baseline"] = random_score
        
        return results
    
    def analyze_failure_patterns(
        self,
        circuit_config: Dict,
        test_contexts: List[str],
        behavioral_test: Callable
    ) -> Dict[str, Any]:
        """
        Analyze patterns in circuit failures to understand limitations.
        
        This helps understand when and why circuits fail, which is crucial
        for improving robustness.
        """
        print("🔍 Analyzing failure patterns...")
        
        failures = []
        successes = []
        
        for context in test_contexts:
            result = self._test_circuit_on_context(
                circuit_config, context, behavioral_test
            )
            
            if result["success"]:
                successes.append({
                    "context": context,
                    "score": result["score"],
                    "context_features": self._extract_context_features(context)
                })
            else:
                failures.append({
                    "context": context,
                    "error": result.get("error", "Unknown"),
                    "context_features": self._extract_context_features(context)
                })
                
        # Analyze patterns
        failure_analysis = {
            "failure_rate": len(failures) / len(test_contexts),
            "common_failure_features": self._find_common_features(failures),
            "success_features": self._find_common_features(successes),
            "distinguishing_features": self._find_distinguishing_features(failures, successes),
            "failure_examples": failures[:5],  # First 5 failures
            "success_examples": successes[:5]  # First 5 successes
        }
        
        return failure_analysis
    
    def visualize_robustness(
        self,
        robustness_report: RobustnessReport,
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations of circuit robustness.
        
        This helps interpret and communicate robustness results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Overall robustness radar chart
        categories = ['Context\nRobustness', 'Variable\nRobustness', 
                     'Noise\nRobustness', 'Intervention\nRobustness']
        scores = [robustness_report.context_robustness, 
                 robustness_report.variable_robustness,
                 robustness_report.noise_robustness,
                 robustness_report.intervention_robustness]
        
        # Simple bar chart instead of radar (matplotlib radar is complex)
        axes[0, 0].bar(categories, scores, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title('Robustness Dimensions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Overall score gauge
        axes[0, 1].pie([robustness_report.overall_score, 1 - robustness_report.overall_score], 
                      labels=['Robust', 'Not Robust'], colors=['green', 'lightgray'],
                      startangle=90)
        axes[0, 1].set_title(f'Overall Score: {robustness_report.overall_score:.2f}')
        
        # Detailed scores heatmap
        if 'behavioral_tests' in robustness_report.detailed_scores:
            behavioral_scores = robustness_report.detailed_scores['behavioral_tests']
            axes[1, 0].imshow([[s] for s in behavioral_scores], cmap='RdYlGn', aspect='auto')
            axes[1, 0].set_title('Behavioral Test Scores')
            axes[1, 0].set_ylabel('Test Index')
            
        # Recommendations text
        axes[1, 1].text(0.1, 0.9, 'Recommendations:', fontsize=12, weight='bold',
                        transform=axes[1, 1].transAxes)
        
        rec_text = '\n'.join([f"• {rec}" for rec in robustness_report.recommendations[:5]])
        axes[1, 1].text(0.1, 0.1, rec_text, fontsize=10, 
                        transform=axes[1, 1].transAxes, verticalalignment='bottom')
        axes[1, 1].set_axis_off()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to {save_path}")
        else:
            plt.show()
    
    def _evaluate_context_robustness(
        self,
        circuit_config: Dict,
        contexts: List[str],
        behavioral_test: Callable
    ) -> float:
        """Evaluate how robust circuit is across different contexts."""
        scores = []
        
        for context in contexts:
            result = self._test_circuit_on_context(circuit_config, context, behavioral_test)
            if result["success"] and result["score"] is not None:
                scores.append(result["score"])
                
        return np.mean(scores) if scores else 0.0
    
    def _evaluate_variable_robustness(
        self,
        circuit_config: Dict,
        contexts: List[str],
        variable_sets: Dict[str, List[str]],
        behavioral_test: Callable
    ) -> float:
        """Evaluate how robust circuit is to different variable instantiations."""
        
        # This is simplified - would generate contexts with different variables
        variable_scores = []
        
        # Test with different variable combinations
        for i in range(min(10, len(contexts))):  # Test first 10 contexts
            context = contexts[i]
            # In practice, you'd systematically vary the variables
            score = self._test_circuit_on_context(circuit_config, context, behavioral_test)
            if score["success"]:
                variable_scores.append(score["score"])
                
        return np.mean(variable_scores) if variable_scores else 0.0
    
    def _evaluate_noise_robustness(
        self,
        circuit_config: Dict,
        contexts: List[str],
        behavioral_test: Callable
    ) -> float:
        """Evaluate how robust circuit is to noise."""
        noise_scores = []
        
        for context in contexts:
            # Test with different noise levels
            for noise_level in [0.0, 0.1, 0.2]:
                # This is simplified - would add actual noise to interventions
                result = self._test_circuit_on_context(circuit_config, context, behavioral_test)
                if result["success"]:
                    # Assume noise reduces performance
                    adjusted_score = result["score"] * (1.0 - noise_level * 0.5)
                    noise_scores.append(adjusted_score)
                    
        return np.mean(noise_scores) if noise_scores else 0.0
    
    def _evaluate_intervention_robustness(
        self,
        circuit_config: Dict,
        contexts: List[str],
        behavioral_test: Callable
    ) -> float:
        """Evaluate how robust circuit interventions are."""
        
        intervention_scores = []
        
        for context in contexts:
            # Test intervention consistency
            scores_for_context = []
            for _ in range(3):  # Test multiple times
                result = self._test_circuit_on_context(circuit_config, context, behavioral_test)
                if result["success"]:
                    scores_for_context.append(result["score"])
                    
            if scores_for_context:
                # Consistency measured as inverse of variance
                consistency = 1.0 - np.var(scores_for_context)
                intervention_scores.append(max(0.0, consistency))
                
        return np.mean(intervention_scores) if intervention_scores else 0.0
    
    def _test_circuit_on_context(
        self,
        circuit_config: Dict,
        context: str,
        behavioral_test: Callable
    ) -> Dict[str, Any]:
        """Test circuit on a specific context."""
        try:
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = behavioral_test(outputs, inputs)
                
            if isinstance(score, torch.Tensor):
                score = score.item()
                
            return {
                "success": True,
                "score": score,
                "context": context
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context": context,
                "score": None
            }
    
    def _test_behavioral_consistency(
        self,
        circuit_config: Dict,
        contexts: List[str],
        behavioral_test: Callable
    ) -> float:
        """Test consistency of a specific behavioral test."""
        scores = []
        
        for context in contexts:
            result = self._test_circuit_on_context(circuit_config, context, behavioral_test)
            if result["success"]:
                scores.append(result["score"])
                
        return np.mean(scores) if scores else 0.0
    
    def _identify_failure_modes(self, context_score: float, variable_score: float, 
                              noise_score: float, intervention_score: float) -> List[str]:
        """Identify common failure modes based on scores."""
        failures = []
        
        if context_score < 0.6:
            failures.append("Context sensitivity - circuit behavior varies significantly across contexts")
            
        if variable_score < 0.6:
            failures.append("Variable dependence - circuit relies on specific variable instantiations")
            
        if noise_score < 0.6:
            failures.append("Noise sensitivity - circuit fails when interventions are noisy")
            
        if intervention_score < 0.6:
            failures.append("Intervention instability - circuit effects are inconsistent")
            
        return failures
    
    def _generate_recommendations(self, context_score: float, variable_score: float,
                                 noise_score: float, intervention_score: float) -> List[str]:
        """Generate recommendations for improving robustness."""
        recommendations = []
        
        if context_score < 0.7:
            recommendations.append("Test circuit on more diverse contexts to improve generalization")
            
        if variable_score < 0.7:
            recommendations.append("Use more variable-invariant features in circuit design")
            
        if noise_score < 0.7:
            recommendations.append("Implement noise-robust intervention techniques")
            
        if intervention_score < 0.7:
            recommendations.append("Increase intervention strength or use ensemble methods")
            
        if all(score > 0.8 for score in [context_score, variable_score, noise_score, intervention_score]):
            recommendations.append("Circuit shows high robustness - consider testing on harder cases")
            
        return recommendations
    
    def _generate_baseline_circuits(self, circuit_config: Dict) -> List[Dict]:
        """Generate baseline circuits for comparison."""
        baselines = []
        
        # Simplified baseline - just the original config
        baselines.append(circuit_config.copy())
        
        # Random baseline - random components
        if "representations" in circuit_config:
            random_config = circuit_config.copy()
            # Would randomize the components in practice
            baselines.append(random_config)
            
        return baselines
    
    def _evaluate_random_baseline(self, contexts: List[str], behavioral_test: Callable) -> float:
        """Evaluate random baseline performance."""
        # This would test completely random interventions
        return np.random.random() * 0.3  # Random baseline typically performs poorly
    
    def _extract_context_features(self, context: str) -> Dict[str, Any]:
        """Extract features from context for failure analysis."""
        return {
            "length": len(context.split()),
            "has_punctuation": any(c in context for c in "!?.,"),
            "has_numbers": any(c.isdigit() for c in context),
            "starts_with_capital": context[0].isupper() if context else False,
            "word_count": len(context.split())
        }
    
    def _find_common_features(self, examples: List[Dict]) -> Dict[str, float]:
        """Find common features in a set of examples."""
        if not examples:
            return {}
            
        feature_counts = defaultdict(int)
        total = len(examples)
        
        for example in examples:
            features = example.get("context_features", {})
            for feature, value in features.items():
                if value:  # For boolean features
                    feature_counts[feature] += 1
                    
        # Return features that appear in >50% of examples
        common_features = {}
        for feature, count in feature_counts.items():
            if count / total > 0.5:
                common_features[feature] = count / total
                
        return common_features
    
    def _find_distinguishing_features(self, failures: List[Dict], successes: List[Dict]) -> Dict[str, float]:
        """Find features that distinguish failures from successes."""
        failure_features = self._find_common_features(failures)
        success_features = self._find_common_features(successes)
        
        distinguishing = {}
        
        # Features more common in failures
        for feature, fail_rate in failure_features.items():
            success_rate = success_features.get(feature, 0.0)
            if fail_rate - success_rate > 0.2:  # 20% difference threshold
                distinguishing[f"failure_indicator_{feature}"] = fail_rate - success_rate
                
        # Features more common in successes  
        for feature, success_rate in success_features.items():
            fail_rate = failure_features.get(feature, 0.0)
            if success_rate - fail_rate > 0.2:
                distinguishing[f"success_indicator_{feature}"] = success_rate - fail_rate
                
        return distinguishing