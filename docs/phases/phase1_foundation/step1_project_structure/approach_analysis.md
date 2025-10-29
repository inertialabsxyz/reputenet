# Step 1: Project Structure Setup - Approach Analysis

**Objective:** Create proper Python project structure with packaging
**Estimated Duration:** 1-2 days

---

## Approach Options

### Option 1: Flat Package Structure
```
reputenet/
├── reputenet/
│   ├── __init__.py
│   ├── graph.py
│   ├── agents/
│   ├── tools/
│   └── schema.py
├── tests/
├── pyproject.toml
└── README.md
```

**Pros:**
- Simple and direct
- Easy to understand for small teams
- Minimal import complexity

**Cons:**
- Can become cluttered as project grows
- Less clear separation between source and other files
- Harder to distinguish package code from scripts

### Option 2: src/ Layout (Recommended)
```
reputenet/
├── src/
│   └── reputenet/
│       ├── __init__.py
│       ├── graph.py
│       ├── agents/
│       ├── tools/
│       ├── schema.py
│       └── di.py
├── tests/
├── docs/
├── pyproject.toml
├── .env.template
└── README.md
```

**Pros:**
- Clear separation of source code from other files
- Prevents accidental imports during development
- Better testing isolation
- Industry standard for modern Python projects

**Cons:**
- Slightly more complex import paths
- Additional directory nesting

### Option 3: Monorepo with Multiple Packages
```
reputenet/
├── packages/
│   ├── reputenet-core/
│   ├── reputenet-agents/
│   └── reputenet-tools/
├── apps/
│   └── reputenet-api/
└── shared/
```

**Pros:**
- Maximum modularity
- Enables independent versioning
- Clear component boundaries

**Cons:**
- Overkill for initial implementation
- Complex dependency management
- Significant overhead for team of 1-2 developers

---

## Recommended Approach: src/ Layout

**Rationale:**
- **Industry Standard:** Widely adopted in modern Python projects
- **Clean Separation:** Source code is clearly isolated
- **Testing Benefits:** Prevents import of source during testing
- **Scalability:** Structure scales well as project grows
- **Tooling Support:** Works well with modern Python tooling

### Detailed Structure

```
reputenet/
├── src/
│   └── reputenet/
│       ├── __init__.py              # Package initialization
│       ├── graph.py                 # LangGraph orchestration
│       ├── cli.py                   # Command-line interface
│       ├── config.py                # Configuration management
│       ├── di.py                    # Dependency injection
│       ├── schema.py                # Data schemas and types
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py              # Base agent class
│       │   ├── profiler.py          # AddressProfiler agent
│       │   ├── risk.py              # RiskScorer agent
│       │   ├── sybil.py             # SybilDetector agent
│       │   ├── aggregator.py        # ReputationAggregator agent
│       │   └── reporter.py          # Reporter agent
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py              # Base tool interface
│       │   ├── eth_provider.py      # Ethereum RPC provider
│       │   ├── etherscan.py         # Etherscan API client
│       │   ├── defillama.py         # DefiLlama API client
│       │   └── labels.py            # Address labeling service
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # Structured logging setup
│           ├── cache.py             # Caching utilities
│           └── validation.py        # Data validation helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration
│   ├── test_graph.py                # Integration tests
│   ├── agents/
│   │   ├── test_profiler.py
│   │   ├── test_risk.py
│   │   └── ...
│   ├── tools/
│   │   ├── test_eth_provider.py
│   │   └── ...
│   └── fixtures/
│       ├── addresses.json           # Test address data
│       └── transactions.json        # Mock transaction data
├── docs/
│   ├── api/                         # API documentation
│   ├── development/                 # Development guides
│   └── deployment/                  # Deployment instructions
├── scripts/
│   ├── setup_dev.py                 # Development environment setup
│   └── run_example.py               # Example usage script
├── .env.template                    # Environment variable template
├── .gitignore                       # Git ignore rules
├── .pre-commit-config.yaml          # Pre-commit hooks
├── pyproject.toml                   # Project configuration
├── README.md                        # Project overview
└── LICENSE                          # Project license
```

---

## Implementation Strategy

### Phase 1a: Core Structure
1. Create `src/reputenet/` directory structure
2. Add `__init__.py` files for package recognition
3. Create basic `pyproject.toml` with project metadata

### Phase 1b: Module Stubs
1. Create empty Python files for main modules
2. Add basic docstrings and TODOs
3. Ensure all imports work correctly

### Phase 1c: Support Files
1. Create `.env.template` with required variables
2. Set up `.gitignore` for Python projects
3. Create basic `README.md` with setup instructions

### Phase 1d: Testing Structure
1. Create `tests/` directory with `conftest.py`
2. Add basic test files that can be expanded later
3. Create test fixtures directory

---

## File Templates

### `src/reputenet/__init__.py`
```python
"""ReputeNet: Multi-Agent System for On-Chain Reputation Analysis."""

__version__ = "0.1.0"
__author__ = "Movement Labs"

from .graph import ReputeNetGraph
from .schema import ReputationInput, ReputationOutput

__all__ = ["ReputeNetGraph", "ReputationInput", "ReputationOutput"]
```

### `pyproject.toml` (Initial)
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
]

dependencies = [
    # Will be filled in Step 2
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio",
    "black",
    "ruff",
    "mypy",
    "pre-commit",
]

[project.urls]
Homepage = "https://github.com/movementlabs/reputenet"
Repository = "https://github.com/movementlabs/reputenet"
Documentation = "https://github.com/movementlabs/reputenet/docs"

[project.scripts]
reputenet = "reputenet.cli:main"

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
]

[tool.hatch.build.targets.wheel]
packages = ["src/reputenet"]
```

---

## Success Criteria

**Step 1 is complete when:**
1. Directory structure is created following src/ layout
2. All package `__init__.py` files exist
3. Basic `pyproject.toml` contains project metadata
4. `.env.template` lists required environment variables
5. `README.md` provides project overview and setup placeholder
6. All directories can be imported without errors
7. Structure follows Python packaging best practices

---

## Next Step Dependencies

This step provides the foundation for:
- **Step 2 (Dependencies):** `pyproject.toml` ready for dependency additions
- **Step 3 (Configuration):** Structure for `config.py` and environment handling
- **Step 4 (Infrastructure):** Testing and tooling configuration locations