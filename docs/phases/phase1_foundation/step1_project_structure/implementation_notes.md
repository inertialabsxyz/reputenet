# Step 1: Project Structure Setup - Implementation Notes

**Implementation Date:** TBD
**Dependencies:** None (foundational step)
**Estimated Time:** 1-2 days

---

## Implementation Sequence

### Phase 1: Directory Structure Creation

```bash
# Create main source structure
mkdir -p src/reputenet/{agents,tools,utils}
mkdir -p tests/{agents,tools,fixtures}
mkdir -p docs/{api,development,deployment}
mkdir -p scripts

# Create __init__.py files for Python packages
touch src/reputenet/__init__.py
touch src/reputenet/agents/__init__.py
touch src/reputenet/tools/__init__.py
touch src/reputenet/utils/__init__.py
touch tests/__init__.py
touch tests/agents/__init__.py
touch tests/tools/__init__.py
```

### Phase 2: Core Module Stubs

Create placeholder files for main modules:

```python
# src/reputenet/graph.py
"""LangGraph orchestration for ReputeNet agents."""

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    """State shared across all nodes in the ReputeNet graph."""
    # TODO: Implement in Phase 2
    pass

def build_graph() -> StateGraph:
    """Build and configure the ReputeNet agent graph."""
    # TODO: Implement in Phase 2
    pass

if __name__ == "__main__":
    # TODO: Add CLI interface
    print("ReputeNet graph execution - TODO: implement")
```

```python
# src/reputenet/schema.py
"""Data schemas and type definitions for ReputeNet."""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class ReputationInput(BaseModel):
    """Input parameters for reputation analysis."""
    chain_id: int = Field(default=1, description="Blockchain chain ID")
    targets: List[str] = Field(description="Target addresses to analyze")
    lookback_days: int = Field(default=90, description="Days of history to analyze")
    max_txs: int = Field(default=2000, description="Maximum transactions to fetch")

class ReputationOutput(BaseModel):
    """Output reputation score and components."""
    address: str
    reputation_score: int = Field(ge=0, le=100)
    components: Dict[str, float]
    flags: List[str] = Field(default_factory=list)

# TODO: Add additional schema definitions in Phase 2
```

### Phase 3: Configuration Files

#### pyproject.toml (Complete)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reputenet"
version = "0.1.0"
description = "Multi-Agent System for On-Chain Reputation Analysis"
authors = [
    {name = "Movement Labs", email = "engineering@movementlabs.xyz"},
]
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
keywords = ["blockchain", "reputation", "langchain", "ethereum"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

# Dependencies will be added in Step 2
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1.0",
    "mypy>=1.0",
    "pre-commit>=3.0",
]

[project.urls]
Homepage = "https://github.com/movementlabs/reputenet"
Repository = "https://github.com/movementlabs/reputenet"
Documentation = "https://github.com/movementlabs/reputenet/docs"
Issues = "https://github.com/movementlabs/reputenet/issues"

[project.scripts]
reputenet = "reputenet.cli:main"

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
    "/LICENSE",
]

[tool.hatch.build.targets.wheel]
packages = ["src/reputenet"]

# Black configuration
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

# Ruff configuration
[tool.ruff]
line-length = 88
target-version = "py39"
exclude = [
    ".bzr",
    ".direnv",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".nox",
    ".pants.d",
    ".ruff_cache",
    ".svn",
    ".tox",
    ".venv",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "node_modules",
    "venv",
]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # imported but unused
"tests/**/*" = ["B011"]   # assert false

# MyPy configuration
[tool.mypy]
python_version = "3.9"
check_untyped_defs = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_return_any = true
strict_equality = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true

# Pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src/reputenet",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

# Coverage configuration
[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

#### .env.template
```bash
# ReputeNet Configuration Template
# Copy this file to .env and fill in your actual values

# Blockchain RPC Configuration
RPC_URL_MAINNET=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
# Alternative: https://eth-mainnet.alchemyapi.io/v2/YOUR_API_KEY

# API Keys
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
ALCHEMY_API_KEY=YOUR_ALCHEMY_API_KEY

# LLM Configuration
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
# Alternative LLM providers:
# ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
# COHERE_API_KEY=YOUR_COHERE_API_KEY

# Optional: DefiLlama (no key required for basic usage)
DEFILLAMA_API_URL=https://api.llama.fi

# Application Configuration
LOG_LEVEL=INFO
CACHE_DIR=./.cache
MAX_RETRIES=3
REQUEST_TIMEOUT=30

# Development Configuration
ENVIRONMENT=development
DEBUG=false
```

### Phase 4: Supporting Files

#### README.md (Initial)
```markdown
# ReputeNet

Multi-Agent System for On-Chain Reputation Analysis using LangChain/LangGraph.

## Overview

ReputeNet analyzes Ethereum wallet addresses to compute reputation scores based on:
- On-chain activity patterns
- Risk assessment heuristics
- Sybil detection algorithms
- Protocol interaction diversity

## Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/movementlabs/reputenet.git
   cd reputenet
   ```

2. **Install Dependencies**
   ```bash
   # Using UV (recommended)
   uv pip install -e ".[dev]"

   # Or using pip
   pip install -e ".[dev]"
   ```

3. **Configure Environment**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys
   ```

4. **Run Example**
   ```bash
   python -m reputenet --address 0x... --chain-id 1
   ```

## Development

### Setup Development Environment
```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/
```

### Project Structure
```
src/reputenet/          # Main package
├── agents/             # LangChain agents
├── tools/              # External API adapters
├── utils/              # Utility functions
├── graph.py            # LangGraph orchestration
├── schema.py           # Data models
└── config.py           # Configuration management

tests/                  # Test suite
docs/                   # Documentation
```

## Architecture

ReputeNet uses a graph-orchestrated multi-agent system:

1. **DataHarvester** - Fetches on-chain transaction data
2. **AddressProfiler** - Extracts behavioral features
3. **RiskScorer** - Applies risk assessment heuristics
4. **SybilDetector** - Detects sybil behavior patterns
5. **ReputationAggregator** - Computes final reputation score
6. **Reporter** - Generates JSON and Markdown outputs

## License

MIT License - see LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](docs/development/CONTRIBUTING.md) for development guidelines.
```

#### Basic Test Structure
```python
# tests/conftest.py
"""Pytest configuration and fixtures."""

import pytest
from typing import Dict, Any

@pytest.fixture
def sample_address() -> str:
    """Sample Ethereum address for testing."""
    return "0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"

@pytest.fixture
def sample_reputation_input() -> Dict[str, Any]:
    """Sample input for reputation analysis."""
    return {
        "chain_id": 1,
        "targets": ["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"],
        "lookback_days": 90,
        "max_txs": 1000
    }

# TODO: Add more fixtures as needed
```

```python
# tests/test_schema.py
"""Test data schema validation."""

import pytest
from reputenet.schema import ReputationInput, ReputationOutput

def test_reputation_input_validation():
    """Test ReputationInput validates correctly."""
    input_data = {
        "chain_id": 1,
        "targets": ["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"],
        "lookback_days": 90,
        "max_txs": 1000
    }

    result = ReputationInput(**input_data)
    assert result.chain_id == 1
    assert len(result.targets) == 1
    assert result.lookback_days == 90

def test_reputation_output_validation():
    """Test ReputationOutput validates correctly."""
    output_data = {
        "address": "0x742c4af20a2e0c8e82be16ab44d9421b1b78e569",
        "reputation_score": 75,
        "components": {
            "activity_quality": 0.8,
            "risk": 0.1,
            "sybil": 0.2
        },
        "flags": ["high_activity"]
    }

    result = ReputationOutput(**output_data)
    assert result.reputation_score == 75
    assert "activity_quality" in result.components
```

---

## Validation Steps

### Pre-Implementation Checklist
- [ ] Stakeholder input on design questions resolved
- [ ] Package management approach confirmed (UV recommended)
- [ ] Module organization strategy agreed upon
- [ ] Configuration architecture defined

### Post-Implementation Verification
```bash
# 1. Verify package structure
python -c "import reputenet; print('Package imports successfully')"

# 2. Check code formatting
black --check src/ tests/
ruff check src/ tests/

# 3. Run basic tests
pytest tests/test_schema.py -v

# 4. Verify configuration loading
python -c "
from reputenet.config import load_config
config = load_config()
print('Configuration loads successfully')
"

# 5. Test CLI entry point
reputenet --help
```

---

## Common Issues and Solutions

### Import Path Issues
If imports don't work correctly:
```bash
# Ensure package is installed in development mode
pip install -e .

# Check PYTHONPATH includes project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Pre-commit Hook Failures
```bash
# Fix formatting issues
black src/ tests/
ruff check --fix src/ tests/

# Update pre-commit hooks
pre-commit autoupdate
```

### Configuration Loading Errors
```bash
# Ensure .env file exists and is properly formatted
cp .env.template .env
# Edit .env with actual values
```

---

## Success Metrics

Step 1 is complete when:
1. ✅ All directories created following src/ layout
2. ✅ Package imports work without errors
3. ✅ Basic tests pass
4. ✅ Code formatting and linting configured
5. ✅ Environment configuration template exists
6. ✅ Documentation reflects actual structure
7. ✅ Pre-commit hooks function properly

**Next Steps:** Proceed to Step 2 (Dependency Management) with completed project structure.