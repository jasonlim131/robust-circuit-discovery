<br />
<div align="center">
  <h1 align="center"><img src="https://i.ibb.co/BNkhQH3/pyvene-logo.png" height="100"></h1>
  <a href="https://arxiv.org/abs/2403.07809"><strong>Read our paper »</strong></a> | <a href="https://stanfordnlp.github.io/pyvene/"><strong>Read the docs »</strong></a>
</div>     

<br />
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pepy/dt/pyvene?color=green"></img></a>
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pypi/v/pyvene?color=red"></img></a> 
<a href="https://pypi.org/project/pyvene/"><img src="https://img.shields.io/pypi/l/pyvene?color=blue"></img></a>

# Robust Circuit Discovery

**Finding Context and Variable-Invariant Circuits in Neural Networks**

This library extends [pyvene](https://github.com/stanfordnlp/pyvene) to discover **robust circuits** that maintain their behavior across different contexts and variables. It addresses a core challenge in mechanistic interpretability: circuits should be stable and generalizable, not dependent on specific surface-level features.

## 🎯 Core Hypothesis

> **Robust circuits should work regardless of context and variable instantiations.**

The circuit literature suggests that there exist universal computational patterns that generalize across tasks. This library provides tools to find and validate such circuits.

## 🚀 Quick Start

```python
import pyvene as pv
from robust_circuits import RobustCircuitDiscovery

# Load model
config, tokenizer, model = pv.create_gpt2_lm("gpt2")

# Initialize discovery engine
discovery_engine = RobustCircuitDiscovery(model, tokenizer)

# Define templates with variables
templates = [
    "When {name} went to {place}, {name}",
    "{name} loves {object}. {name}",
]

variables = {
    "name": ["Alice", "Bob", "Charlie"],
    "place": ["Paris", "school", "home"],
    "object": ["books", "music", "food"]
}

# Define behavioral tests
def induction_test(model_output, inputs):
    # Test induction behavior
    return confidence_score

# Discover robust circuits
circuits = discovery_engine.discover_robust_circuits(
    task_templates=templates,
    variable_sets=variables,
    behavioral_tests=[induction_test],
    robustness_threshold=0.8
)

print(f"Found {len(circuits)} robust circuits!")
```

## 🏗️ Architecture

### Core Components

1. **CircuitStabilityAnalyzer** - Measures consistency across contexts
2. **InvariantDetector** - Finds patterns that remain stable
3. **RobustInterventionTechnique** - Advanced intervention methods
4. **RobustnessEvaluator** - Comprehensive evaluation metrics
5. **RobustCircuitDiscovery** - Main orchestrator

### Key Concepts

- **Context Invariance**: Circuit works across different sentence structures
- **Variable Invariance**: Circuit works with different names, objects, places
- **Robust Interventions**: Interventions that work across contexts
- **Universal Motifs**: Patterns that appear in multiple circuits

## 📊 Evaluation Metrics

### Robustness Dimensions

1. **Structural Consistency** (0-1): How consistent is the circuit structure?
2. **Functional Consistency** (0-1): How consistent is the circuit behavior?
3. **Context Independence** (0-1): How independent from surface features?
4. **Noise Robustness** (0-1): How tolerant to intervention noise?

### Overall Robustness Score

```
Robustness = 0.3 × Structural + 0.4 × Functional + 0.3 × Independence
```

## 🔍 Discovery Process

### 1. Template Definition
```python
templates = [
    "When {name} went to {place}, {name}",      # Classic pattern
    "{name} and {other} visited {place}. {name}",  # With distractor
    "Yesterday {name} bought {object}. {name}",  # Temporal context
]
```

### 2. Variable Instantiation
```python
variables = {
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "place": ["Paris", "school", "the park"],
    "object": ["a book", "flowers", "tickets"]
}
```

### 3. Context Generation
- Systematically fill templates with different variables
- Create variations (punctuation, case, prefixes)
- Generate 50+ diverse test contexts

### 4. Invariant Pattern Detection
- Test each model component across contexts
- Identify consistently important components
- Validate patterns across contexts

### 5. Circuit Construction
- Build full circuits from patterns
- Test stability across contexts
- Measure robustness score

### 6. Validation
- Test on completely unseen contexts
- Compare against baseline circuits
- Analyze failure modes

## 📈 Results Interpretation

### High Robustness (>0.8)
✅ Circuit likely captures fundamental computation  
✅ Should generalize to new contexts  
✅ Good candidate for universal pattern  

### Medium Robustness (0.6-0.8)
⚠️ Circuit works but has context dependencies  
⚠️ May need refinement or additional components  
⚠️ Useful but not fully robust  

### Low Robustness (<0.6)
❌ Circuit likely overfit to specific contexts  
❌ May not represent true computational pattern  
❌ Needs redesign or different approach  

## 🧪 Example: Robust Induction Circuit

```python
# Run the complete example
python examples/discover_robust_induction_circuit.py
```

This example demonstrates finding induction circuits that work across:
- Different sentence structures
- Different names and objects  
- Different temporal contexts
- Different syntactic patterns

Expected output:
```
🎉 Found 3 robust circuits!
✅ Circuit 1: robustness=0.87, components=5
✅ Circuit 2: robustness=0.82, components=7  
✅ Circuit 3: robustness=0.79, components=4

🔍 Key insight: Robust circuits use layers 4-8
   and are independent of surface-level features
```

## 🔬 Advanced Features

### Circuit Family Comparison
```python
# Compare high vs medium robustness circuits
comparison = discovery_engine.compare_circuit_families(
    high_robustness_circuits,
    medium_robustness_circuits,
    test_contexts
)
```

### Universal Motif Detection
```python
# Find patterns that appear across multiple circuits
motifs = discovery_engine.find_universal_motifs(discovered_circuits)
```

### Failure Pattern Analysis
```python
# Understand when and why circuits fail
analysis = evaluator.analyze_failure_patterns(
    circuit_config, test_contexts, behavioral_test
)
```

### Cross-Context Validation
```python
# Test if circuits work on completely different contexts
validation = discovery_engine.validate_circuit_robustness(
    circuit, novel_contexts, behavioral_tests
)
```

## 📚 Theoretical Foundation

### Circuit Universality Hypothesis
Based on research suggesting that neural networks learn universal computational patterns:

1. **Induction Heads** - Universal pattern copying mechanisms
2. **Name Mover Heads** - Universal information routing
3. **Suppression Heads** - Universal duplicate suppression

### Robustness Criteria
Derived from mechanistic interpretability principles:

1. **Causal Sufficiency** - Circuit captures essential computation
2. **Context Independence** - Works across surface variations  
3. **Intervention Stability** - Effects are consistent and replicable
4. **Compositional Structure** - Circuit has interpretable subcomponents

## 🛠️ Installation

```bash
# Install dependencies
pip install torch transformers numpy networkx matplotlib

# Install pyvene
pip install pyvene

# Clone this repository
git clone <your-repo-url>
cd robust-circuit-discovery

# Install in development mode
pip install -e .
```

## 📖 Citation

If you use this library in your research, please cite:

```bibtex
@software{robust_circuits_2024,
  title={Robust Circuit Discovery: Finding Context and Variable-Invariant Circuits},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/robust-circuit-discovery}
}
```

And the original pyvene paper:
```bibtex
@inproceedings{wu-etal-2024-pyvene,
    title = "pyvene: A Library for Understanding and Improving {P}y{T}orch Models via Interventions",
    author = "Wu, Zhengxuan and Geiger, Atticus and Arora, Aryaman and Huang, Jing and Wang, Zheng and Goodman, Noah and Manning, Christopher and Potts, Christopher",
    booktitle = "Proceedings of NAACL-HLT 2024",
    year = "2024",
}
```

## 🤝 Contributing

We welcome contributions! Areas where help is needed:

1. **New Circuit Types** - Implement discovery for other circuit types
2. **Evaluation Metrics** - Develop better robustness measures
3. **Baseline Circuits** - Create more sophisticated baselines
4. **Visualization** - Improve circuit visualization tools
5. **Documentation** - Expand tutorials and examples

See `CONTRIBUTING.md` for details.

## 🐛 Known Limitations

1. **Computational Complexity** - Full circuit discovery can be slow
2. **Model Dependency** - Currently optimized for transformer models
3. **Evaluation Metrics** - Robustness metrics are still being refined
4. **Scale** - Not yet tested on very large models (>10B parameters)

## 🗺️ Roadmap

### Short Term (Q1 2024)
- [ ] Support for more model architectures (CNNs, RNNs)
- [ ] Improved visualization tools
- [ ] More sophisticated baseline generation
- [ ] Performance optimizations

### Medium Term (Q2-Q3 2024)  
- [ ] Automated behavioral test generation
- [ ] Multi-task circuit discovery
- [ ] Circuit editing and modification tools
- [ ] Integration with other interpretability tools

### Long Term (Q4 2024+)
- [ ] Theoretical analysis of robustness measures
- [ ] Large-scale circuit databases
- [ ] Real-time circuit monitoring
- [ ] Applications to AI safety

## 🙋 FAQ

**Q: How is this different from regular circuit discovery?**  
A: Regular circuit discovery often finds circuits that work on specific examples but fail to generalize. We specifically search for circuits that work across diverse contexts and variables.

**Q: What makes a circuit "robust"?**  
A: A robust circuit maintains its function across different contexts, variables, and even when interventions are noisy. It captures the essential computation, not surface-level patterns.

**Q: How do I know if my circuit is truly robust?**  
A: Test it on completely different contexts than those used for discovery. A robust circuit should maintain high performance even on novel contexts.

**Q: Can I use this with models other than GPT-2?**  
A: Yes! The system works with any PyTorch model, though it's currently optimized for transformer architectures.

**Q: How long does circuit discovery take?**  
A: Depends on the number of contexts and circuit complexity. For small models (~100M parameters), expect 10-30 minutes. For larger models, it can take several hours.

## 📄 License

MIT License - see `LICENSE` file for details.

## 🔗 Related Work

- [pyvene](https://github.com/stanfordnlp/pyvene) - Base intervention framework
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Transformer analysis tools
- [Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) - Original induction head paper
- [IOI Circuit](https://arxiv.org/abs/2211.00593) - Indirect Object Identification circuit

---

**Built with ❤️ for the mechanistic interpretability community**

