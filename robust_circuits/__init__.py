"""
Robust Circuit Discovery: A library for finding context and variable-invariant circuits.

This library extends pyvene to discover robust circuits that maintain their behavior
across different contexts and variables, addressing the core challenge in circuit
literature that circuits should be stable and generalizable.
"""

from .circuit_stability import CircuitStabilityAnalyzer
from .invariant_detection import InvariantDetector
from .robust_interventions import RobustInterventionTechnique
from .evaluation import RobustnessEvaluator
from .discovery_engine import RobustCircuitDiscovery

__version__ = "0.1.0"
__all__ = [
    "CircuitStabilityAnalyzer",
    "InvariantDetector", 
    "RobustInterventionTechnique",
    "RobustnessEvaluator",
    "RobustCircuitDiscovery"
]