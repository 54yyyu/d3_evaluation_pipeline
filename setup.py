"""
Setup script for the evaluation pipeline package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        "torch>=1.9.0",
        "pytorch-lightning>=1.9.0", 
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.1.0",
        "biopython>=1.78",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0"
    ]

setup(
    name="d3-evaluation-pipeline",
    version="0.1.0",
    description="Comprehensive evaluation pipeline for synthetic DNA sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="D3 Analysis Team",
    author_email="",
    url="https://github.com/your-org/d3-analysis",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0"
        ],
        "motif-analysis": [
            "pymemesuite>=0.1.0",
            "tangermeme>=0.3.0"
        ],
        "attribution": [
            "captum>=0.5.0",
            "logomaker>=0.8"
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0"
        ],
        "all": [
            "pymemesuite>=0.1.0",
            "tangermeme>=0.3.0", 
            "captum>=0.5.0",
            "logomaker>=0.8",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0"
        ]
    },
    
    # Entry points for CLI scripts
    entry_points={
        "console_scripts": [
            "evaluate-single=scripts.run_single_evaluation:main",
            "evaluate-full=scripts.run_full_pipeline:main",
        ]
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "*.md", "*.txt"]
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Keywords
    keywords="bioinformatics, DNA sequences, evaluation, machine learning, synthetic biology",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/your-org/d3-analysis/issues",
        "Source": "https://github.com/your-org/d3-analysis",
        "Documentation": "https://github.com/your-org/d3-analysis/blob/main/README.md",
    },
)