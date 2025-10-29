# Step 2: Dependency Management - Implementation Notes

**Implementation Context:** UV package manager, range pinning, mock-first approach
**Estimated Duration:** 1-2 hours
**Dependencies:** Step 1 (Project Structure) complete

---

## Implementation Strategy

### 1. Complete pyproject.toml Configuration (30 minutes)

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reputenet"
version = "0.1.0"
description = "Multi-Agent System for On-Chain Reputation Analysis (Prototype)"
authors = [
    {name = "Movement Labs", email = "engineering@movementlabs.xyz"},
]
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
keywords = ["blockchain", "reputation", "langchain", "ethereum"]

dependencies = [
    # Core LangChain ecosystem
    "langchain>=0.2.0,<0.3.0",
    "langgraph>=0.2.0,<0.3.0",
    "langchain-openai>=0.1.0,<0.2.0",

    # Data handling and validation
    "pydantic>=2.7.0,<3.0.0",
    "pandas>=2.0.0,<3.0.0",
    "numpy>=1.24.0,<2.0.0",

    # Blockchain and Web3 (with mock support)
    "web3>=6.11.0,<7.0.0",
    "eth-account>=0.9.0,<1.0.0",
    "eth-utils>=2.2.0,<3.0.0",

    # HTTP and API clients
    "httpx>=0.27.0,<0.28.0",
    "requests>=2.31.0,<3.0.0",
    "aiohttp>=3.9.0,<4.0.0",

    # Infrastructure
    "python-dotenv>=1.0.0,<2.0.0",
    "structlog>=24.1.0,<25.0.0",
    "diskcache>=5.6.0,<6.0.0",
    "click>=8.1.0,<9.0.0",

    # Mock data generation
    "faker>=20.0.0,<21.0.0",
    "factory-boy>=3.3.0,<4.0.0",
    "responses>=0.24.0,<0.25.0",

    # Service framework
    "fastapi>=0.104.0,<0.105.0",
    "uvicorn[standard]>=0.24.0,<0.25.0",

    # Data analysis
    "scipy>=1.11.0,<2.0.0",
    "networkx>=3.2.0,<4.0.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.21.0,<0.22.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-mock>=3.12.0,<4.0.0",
    "pytest-xdist>=3.3.0,<4.0.0",

    # Code quality
    "black>=23.9.0,<24.0.0",
    "ruff>=0.1.0,<0.2.0",
    "mypy>=1.6.0,<2.0.0",
    "pre-commit>=3.5.0,<4.0.0",

    # Development tools
    "ipython>=8.15.0,<9.0.0",
    "jupyter>=1.0.0,<2.0.0",
    "rich>=13.6.0,<14.0.0",

    # Documentation
    "mkdocs>=1.5.0,<2.0.0",
    "mkdocs-material>=9.4.0,<10.0.0",
]

production = [
    "gunicorn>=21.2.0,<22.0.0",
    "prometheus-client>=0.19.0,<1.0.0",
    "sentry-sdk>=1.35.0,<2.0.0",
]

performance = [
    "orjson>=3.9.0,<4.0.0",
    "uvloop>=0.19.0,<1.0.0",
    "cython>=3.0.0,<4.0.0",
]

[project.scripts]
reputenet = "reputenet.cli:main"
reputenet-api = "reputenet.api:run"

[project.urls]
Homepage = "https://github.com/movementlabs/reputenet"
Repository = "https://github.com/movementlabs/reputenet"
Documentation = "https://reputenet.readthedocs.io"
Issues = "https://github.com/movementlabs/reputenet/issues"

# Development tool configurations
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 88
target-version = "py39"
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "N",   # pep8-naming
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # imported but unused
"tests/**/*" = ["B011"]   # assert false

[tool.mypy]
python_version = "3.9"
check_untyped_defs = true
disallow_any_generics = true
disallow_incomplete_defs = true
disallow_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "web3.*",
    "eth_account.*",
    "faker.*",
    "factory_boy.*",
    "responses.*",
    "networkx.*",
]
ignore_missing_imports = true

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
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "mock: marks tests that use mock data",
]

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/test_*",
    "*/conftest.py",
]

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

### 2. Environment Setup Script (15 minutes)

```bash
#!/bin/bash
# scripts/setup_dev.sh - Development environment setup

set -e

echo "🚀 Setting up ReputeNet development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "✅ UV package manager available"

# Create virtual environment and install dependencies
echo "📦 Installing dependencies with UV..."
uv pip install -e ".[dev]"

echo "✅ Dependencies installed successfully"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

echo "✅ Pre-commit hooks configured"

# Create .env from template if it doesn't exist
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.template .env
    echo "✅ .env file created - please edit with your configuration"
else
    echo "✅ .env file already exists"
fi

# Run basic tests to verify setup
echo "🧪 Running basic tests to verify setup..."
python -c "import reputenet; print('✅ Package imports successfully')"

# Check code formatting
echo "🎨 Checking code formatting..."
black --check src/ tests/ || echo "ℹ️  Run 'black src/ tests/' to format code"
ruff check src/ tests/ || echo "ℹ️  Run 'ruff check --fix src/ tests/' to fix issues"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your configuration"
echo "  2. Run 'pytest' to run tests"
echo "  3. Run 'reputenet --help' to see CLI options"
echo "  4. Run 'reputenet-api' to start the API server"
```

### 3. Dependency Installation Validation (15 minutes)

```python
# scripts/validate_dependencies.py
"""Validate all dependencies are correctly installed and compatible."""

import sys
import importlib
import subprocess
from typing import List, Tuple, Dict

def check_core_dependencies() -> List[Tuple[str, bool, str]]:
    """Check core runtime dependencies."""

    core_deps = [
        ("langchain", "LangChain core framework"),
        ("langgraph", "LangGraph orchestration"),
        ("pydantic", "Data validation"),
        ("web3", "Ethereum integration"),
        ("fastapi", "API framework"),
        ("structlog", "Structured logging"),
        ("httpx", "HTTP client"),
        ("faker", "Mock data generation"),
        ("diskcache", "Caching layer"),
    ]

    results = []
    for dep, description in core_deps:
        try:
            importlib.import_module(dep)
            results.append((dep, True, description))
        except ImportError as e:
            results.append((dep, False, f"Import error: {e}"))

    return results

def check_dev_dependencies() -> List[Tuple[str, bool, str]]:
    """Check development dependencies."""

    dev_deps = [
        ("pytest", "Testing framework"),
        ("black", "Code formatting"),
        ("ruff", "Linting"),
        ("mypy", "Type checking"),
        ("pre_commit", "Pre-commit hooks"),
    ]

    results = []
    for dep, description in dev_deps:
        try:
            if dep == "pre_commit":
                # Special case for pre-commit (hyphenated package name)
                importlib.import_module("pre_commit")
            else:
                importlib.import_module(dep)
            results.append((dep, True, description))
        except ImportError as e:
            results.append((dep, False, f"Import error: {e}"))

    return results

def check_version_compatibility() -> Dict[str, str]:
    """Check Python and key package versions."""

    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

    # Check key package versions
    try:
        import langchain
        versions["langchain"] = langchain.__version__
    except:
        versions["langchain"] = "Not available"

    try:
        import pydantic
        versions["pydantic"] = pydantic.__version__
    except:
        versions["pydantic"] = "Not available"

    try:
        import web3
        versions["web3"] = web3.__version__
    except:
        versions["web3"] = "Not available"

    return versions

def run_basic_functionality_tests() -> List[Tuple[str, bool, str]]:
    """Run basic functionality tests."""

    tests = []

    # Test package import
    try:
        import reputenet
        tests.append(("package_import", True, "ReputeNet package imports successfully"))
    except Exception as e:
        tests.append(("package_import", False, f"Package import failed: {e}"))

    # Test configuration loading
    try:
        from reputenet.config import load_config
        config = load_config()
        tests.append(("config_loading", True, "Configuration loads successfully"))
    except Exception as e:
        tests.append(("config_loading", False, f"Configuration loading failed: {e}"))

    # Test mock data generation
    try:
        from reputenet.mock_data.generators import MockAddressGenerator
        generator = MockAddressGenerator()
        profile = generator.generate_profile("normal_user")
        tests.append(("mock_data", True, "Mock data generation works"))
    except Exception as e:
        tests.append(("mock_data", False, f"Mock data generation failed: {e}"))

    return tests

def main():
    """Main validation function."""

    print("🔍 Validating ReputeNet Dependencies")
    print("=" * 50)

    # Check core dependencies
    print("\n📦 Core Dependencies:")
    core_results = check_core_dependencies()
    for dep, success, description in core_results:
        status = "✅" if success else "❌"
        print(f"  {status} {dep:<15} - {description}")

    # Check dev dependencies
    print("\n🛠️  Development Dependencies:")
    dev_results = check_dev_dependencies()
    for dep, success, description in dev_results:
        status = "✅" if success else "❌"
        print(f"  {status} {dep:<15} - {description}")

    # Check versions
    print("\n📋 Version Information:")
    versions = check_version_compatibility()
    for package, version in versions.items():
        print(f"  📌 {package:<15} - {version}")

    # Run functionality tests
    print("\n🧪 Functionality Tests:")
    func_results = run_basic_functionality_tests()
    for test, success, description in func_results:
        status = "✅" if success else "❌"
        print(f"  {status} {test:<15} - {description}")

    # Summary
    core_failed = sum(1 for _, success, _ in core_results if not success)
    dev_failed = sum(1 for _, success, _ in dev_results if not success)
    func_failed = sum(1 for _, success, _ in func_results if not success)

    total_failed = core_failed + dev_failed + func_failed

    print(f"\n📊 Summary:")
    print(f"  Core dependencies: {len(core_results) - core_failed}/{len(core_results)} ✅")
    print(f"  Dev dependencies:  {len(dev_results) - dev_failed}/{len(dev_results)} ✅")
    print(f"  Functionality:     {len(func_results) - func_failed}/{len(func_results)} ✅")

    if total_failed == 0:
        print("\n🎉 All dependencies validated successfully!")
        return 0
    else:
        print(f"\n❌ {total_failed} issues found. Please resolve before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### 4. Installation Commands (20 minutes)

```bash
# Development setup commands

# 1. Install UV package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies with UV
uv pip install -e ".[dev]"

# 4. Set up pre-commit hooks
pre-commit install

# 5. Create environment configuration
cp .env.template .env
# Edit .env with appropriate values

# 6. Validate installation
python scripts/validate_dependencies.py

# 7. Run initial tests
pytest tests/ -v

# 8. Check code quality
black --check src/ tests/
ruff check src/ tests/
mypy src/

# 9. Test CLI interface
reputenet --help

# 10. Test API interface
reputenet-api &
curl http://localhost:8000/health
```

### 5. Troubleshooting Guide

```markdown
## Common Installation Issues

### Issue: UV not found
**Solution:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Issue: Python version incompatibility
**Solution:**
```bash
# Install Python 3.9+ using pyenv
pyenv install 3.11.6
pyenv local 3.11.6
```

### Issue: Pre-commit hooks failing
**Solution:**
```bash
pre-commit clean
pre-commit install
pre-commit run --all-files
```

### Issue: Import errors after installation
**Solution:**
```bash
# Reinstall in development mode
pip uninstall reputenet
uv pip install -e ".[dev]"
```

### Issue: Permission errors on macOS/Linux
**Solution:**
```bash
# Use user installation
uv pip install --user -e ".[dev]"
```

### Issue: Windows-specific installation problems
**Solution:**
```bash
# Use PowerShell and ensure execution policy allows scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then run installation commands
```
```

---

## Success Criteria

**Step 2 is complete when:**

1. ✅ **All dependencies install without conflicts** using UV
2. ✅ **Development tools work correctly** (black, ruff, pytest, mypy)
3. ✅ **Pre-commit hooks function properly**
4. ✅ **Package imports successfully** in development mode
5. ✅ **Configuration loading works** with .env file
6. ✅ **Basic CLI and API interfaces respond**
7. ✅ **Validation script passes** all dependency checks

**Next Dependencies:**
- Provides foundation for Step 3 (Configuration System)
- Enables Step 4 (Basic Infrastructure) tool setup
- Supports all future development workflow