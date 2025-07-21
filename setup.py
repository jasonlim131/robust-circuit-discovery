"""
Setup script for Robust Circuit Discovery library.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

requirements = [
    "torch>=1.9.0",
    "transformers>=4.20.0", 
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "networkx>=2.6.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "scikit-learn>=1.0.0",
    "tqdm>=4.60.0",
    "pyvene>=0.1.8",
]

setup(
    name="robust-circuits",
    version="0.1.0",
    author="Circuit Discovery Team",
    author_email="your.email@example.com",
    description="Finding context and variable-invariant circuits in neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/robust-circuit-discovery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "discover-robust-circuits=examples.discover_robust_induction_circuit:run_robust_induction_discovery",
        ],
    },
    include_package_data=True,
    package_data={
        "robust_circuits": ["*.json", "*.txt"],
    },
    keywords=[
        "mechanistic interpretability",
        "neural networks", 
        "circuit discovery",
        "robustness",
        "invariance",
        "transformers",
        "interpretability",
        "AI safety",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/robust-circuit-discovery/issues",
        "Source": "https://github.com/your-username/robust-circuit-discovery",
        "Documentation": "https://github.com/your-username/robust-circuit-discovery/blob/main/README.md",
    },
)