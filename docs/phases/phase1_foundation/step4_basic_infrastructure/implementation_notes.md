# Step 4: Basic Project Infrastructure - Implementation Notes

**Context:** Professional development tooling setup with balanced enforcement
**Approach:** Comprehensive toolchain optimized for prototype development velocity

---

## Implementation Strategy

### Toolchain Architecture
Based on design decisions, implementing:
- **Black + Ruff** for formatting and linting with minimal configuration
- **mypy** for gradual type checking with prototype-friendly settings
- **pytest** with comprehensive testing infrastructure
- **pre-commit** with advisory warnings, not blocking commits
- **GitHub Actions** for CI/CD with quality gates

### Configuration Management
All tool configuration centralized in `pyproject.toml` for consistency and maintainability.

---

## Core Tool Configuration

### pyproject.toml - Tool Configuration Section

#### Development Dependencies
```toml
# Add to existing pyproject.toml [project.optional-dependencies]
dev = [
    # Code quality
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",

    # Testing
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.11.0",
    "pytest-asyncio>=0.21.0",
    "coverage[toml]>=7.3.0",

    # Development workflow
    "pre-commit>=3.4.0",
    "commitizen>=3.8.0",

    # Type stubs
    "types-requests>=2.31.0",
    "types-PyYAML>=6.0.0",
]
```

#### Tool Configurations
```toml
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
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

# Enable specific rule sets
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
    "N",    # pep8-naming
]

ignore = [
    "E501",   # Line too long (handled by Black)
    "E203",   # Whitespace before ':' (Black compatibility)
    "B008",   # Do not perform function calls in argument defaults
    "N812",   # Lowercase imported as non-lowercase (for sklearn, etc.)
]

# Exclude files and directories
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

[tool.ruff.mccabe]
max-complexity = 10

[tool.ruff.isort]
known-first-party = ["reputenet"]
force-single-line = false
lines-after-imports = 2

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

# Gradual typing approach - not too strict for prototype
disallow_untyped_defs = false
disallow_incomplete_defs = false
disallow_untyped_calls = false
check_untyped_defs = true

# Module discovery
mypy_path = "src"
namespace_packages = true
explicit_package_bases = true

# Error output
show_error_codes = true
show_column_numbers = true
pretty = true

[[tool.mypy.overrides]]
module = [
    "langchain.*",
    "langgraph.*",
    "web3.*",
    "eth_account.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Test discovery and execution
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=html:htmlcov",
    "--cov-report=term-missing",
    "--cov-report=xml",
    "--cov-fail-under=70",  # Start with reasonable coverage threshold
    "-ra",  # Show all test results
]

# Test markers
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "mock: marks tests that use mock data",
    "real_api: marks tests that require real API access",
]

# Async test support
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*",
    "*/migrations/*",
    "*/scripts/*",
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

[tool.coverage.html]
directory = "htmlcov"

[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
tag_format = "v$major.$minor.$patch"
version_files = [
    "pyproject.toml:version",
    "src/reputenet/__init__.py:__version__",
]
```

---

## Pre-commit Configuration

### .pre-commit-config.yaml
```yaml
# Pre-commit hooks configuration
repos:
  # Code formatting
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        name: Black code formatter
        description: "Format Python code with Black"
        args: [--preview]

  # Import sorting
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        name: isort import sorter
        description: "Sort Python imports"
        args: ["--profile", "black"]

  # Linting and code quality
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.3
    hooks:
      - id: ruff
        name: Ruff linter
        description: "Run Ruff for linting"
        args: [--fix, --exit-non-zero-on-fix]

  # Type checking (optional, can be slow)
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        name: mypy type checker
        description: "Run mypy for type checking"
        additional_dependencies: [
          "types-requests",
          "types-PyYAML",
          "pydantic",
        ]
        args: [--config-file=pyproject.toml]

  # Security checks
  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        name: Bandit security linter
        description: "Run Bandit for security issues"
        args: ["-r", "src", "-f", "json"]
        exclude: "tests/"

  # General file checks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
        name: Trim trailing whitespace
      - id: end-of-file-fixer
        name: Fix end of files
      - id: check-yaml
        name: Check YAML syntax
      - id: check-toml
        name: Check TOML syntax
      - id: check-added-large-files
        name: Check for large files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
        name: Check for merge conflicts
      - id: debug-statements
        name: Check for debug statements

  # Local custom hooks
  - repo: local
    hooks:
      - id: pytest-fast
        name: Fast unit tests
        entry: pytest tests/unit --maxfail=3 -x -q
        language: system
        pass_filenames: false
        types: [python]
        stages: [commit]

      - id: config-validation
        name: Configuration validation
        entry: python -m config.validation
        language: system
        pass_filenames: false
        files: "^(config/|\.env)"
```

---

## Testing Infrastructure

### Test Directory Structure
```
tests/
├── conftest.py                 # Global test configuration
├── unit/                       # Unit tests
│   ├── conftest.py            # Unit test fixtures
│   ├── test_config.py         # Configuration tests
│   ├── agents/                # Agent unit tests
│   │   ├── test_data_harvester.py
│   │   ├── test_address_profiler.py
│   │   └── test_risk_scorer.py
│   └── tools/                 # Tool unit tests
├── integration/               # Integration tests
│   ├── conftest.py           # Integration test fixtures
│   ├── test_agent_pipeline.py
│   └── test_api_endpoints.py
├── fixtures/                  # Test data and mocks
│   ├── mock_blockchain_data.py
│   ├── sample_addresses.json
│   └── mock_responses/
└── performance/              # Performance tests
    └── test_pipeline_performance.py
```

### Global Test Configuration

#### tests/conftest.py
```python
"""Global test configuration and fixtures."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock

from config import Settings, load_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_settings() -> Settings:
    """Get test environment configuration."""
    return load_settings("testing")


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Mock LLM for testing agent interactions."""
    llm = AsyncMock()
    llm.ainvoke.return_value = "Mock LLM response"
    llm.name = "MockLLM"
    return llm


@pytest.fixture
def mock_web3() -> Mock:
    """Mock Web3 provider for blockchain interactions."""
    web3 = Mock()

    # Mock common Web3 methods
    web3.eth.get_block.return_value = {
        "number": 18500000,
        "hash": "0x1234567890abcdef",
        "timestamp": 1699000000,
        "transactions": []
    }

    web3.eth.get_transaction.return_value = {
        "hash": "0xabcdef1234567890",
        "from": "0x1111111111111111111111111111111111111111",
        "to": "0x2222222222222222222222222222222222222222",
        "value": 1000000000000000000,  # 1 ETH in wei
        "gas": 21000,
        "gasPrice": 20000000000,  # 20 gwei
    }

    web3.isConnected.return_value = True
    return web3


@pytest.fixture
def mock_database(test_settings):
    """Mock database for testing."""
    # Database setup and teardown logic
    pass


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.mock = pytest.mark.mock
```

#### tests/unit/conftest.py
```python
"""Unit test specific fixtures."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from reputenet.agents import DataHarvester, AddressProfiler, RiskScorer


@pytest.fixture
def mock_data_harvester():
    """Mock DataHarvester agent."""
    harvester = AsyncMock(spec=DataHarvester)
    harvester.collect_address_data.return_value = {
        "address": "0x1234567890123456789012345678901234567890",
        "transactions": [],
        "balance": "1000000000000000000",
        "contract_interactions": []
    }
    return harvester


@pytest.fixture
def mock_address_profiler():
    """Mock AddressProfiler agent."""
    profiler = AsyncMock(spec=AddressProfiler)
    profiler.analyze_behavior.return_value = {
        "sophistication_score": 0.7,
        "behavior_patterns": ["defi_user", "governance_participant"],
        "risk_indicators": ["high_value_transactions"]
    }
    return profiler


@pytest.fixture
def mock_blockchain_data():
    """Sample blockchain data for testing."""
    return {
        "block": {
            "number": 18500000,
            "timestamp": 1699000000,
            "transactions": ["0xabc123", "0xdef456"]
        },
        "transactions": [
            {
                "hash": "0xabc123",
                "from": "0x1111111111111111111111111111111111111111",
                "to": "0x2222222222222222222222222222222222222222",
                "value": "1000000000000000000",
                "gas": 21000,
                "gasPrice": "20000000000"
            }
        ]
    }


@pytest.fixture
def sample_address():
    """Sample Ethereum address for testing."""
    return "0x1234567890123456789012345678901234567890"


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    with patch.dict('os.environ', {
        'REPUTENET_APP_ENVIRONMENT': 'testing',
        'REPUTENET_DATABASE_URL': 'sqlite:///:memory:',
        'REPUTENET_API_MODE': 'mock',
        'REPUTENET_LOGGING_LEVEL': 'WARNING'
    }):
        yield
```

### Sample Unit Tests

#### tests/unit/test_config.py
```python
"""Configuration system unit tests."""

import os
import pytest
from unittest.mock import patch

from config import Settings, load_settings, validate_configuration


class TestConfigurationLoading:
    """Test configuration loading and validation."""

    def test_development_config_defaults(self):
        """Test development configuration has correct defaults."""
        settings = load_settings("development")

        assert settings.app.debug is True
        assert settings.app.environment == "development"
        assert settings.api.mode == "mock"
        assert settings.logging.level == "DEBUG"

    def test_production_config_security(self):
        """Test production configuration has secure defaults."""
        settings = load_settings("production")

        assert settings.app.debug is False
        assert settings.app.environment == "production"
        assert settings.api.mode == "real"
        assert settings.logging.format == "json"

    def test_environment_variable_override(self):
        """Test environment variables override defaults."""
        with patch.dict(os.environ, {
            'REPUTENET_APP_DEBUG': 'true',
            'REPUTENET_API_MODE': 'real'
        }):
            settings = load_settings("development")
            assert settings.app.debug is True
            assert settings.api.mode == "real"

    def test_configuration_validation(self):
        """Test configuration validation catches issues."""
        settings = load_settings("production")
        issues = validate_configuration(settings)

        # Should have issues with default secret key
        assert any("secret key" in issue.lower() for issue in issues)

    @pytest.mark.parametrize("environment", ["development", "testing", "production", "mock"])
    def test_all_environments_load(self, environment):
        """Test all environment configurations load successfully."""
        settings = load_settings(environment)
        assert settings.app.environment == environment


class TestConfigurationModels:
    """Test configuration model validation."""

    def test_database_config_validation(self):
        """Test database configuration validation."""
        from config.base import DatabaseConfig

        # Valid configuration
        config = DatabaseConfig(url="postgresql://user:pass@localhost/db")
        assert config.url == "postgresql://user:pass@localhost/db"

        # Invalid pool size
        with pytest.raises(ValueError):
            DatabaseConfig(pool_size=0)

    def test_api_config_validation(self):
        """Test API configuration validation."""
        from config.base import APIConfig

        # Valid configuration
        config = APIConfig(mode="mock", rate_limit=100)
        assert config.mode == "mock"
        assert config.rate_limit == 100

        # Invalid mode
        with pytest.raises(ValueError):
            APIConfig(mode="invalid_mode")

    def test_logging_config_validation(self):
        """Test logging configuration validation."""
        from config.base import LoggingConfig

        # Valid configuration
        config = LoggingConfig(level="INFO", format="json")
        assert config.level == "INFO"
        assert config.format == "json"

        # Invalid level
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")
```

---

## CI/CD Pipeline Implementation

### GitHub Actions Workflow

#### .github/workflows/ci.yml
```yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  PYTHON_VERSION: "3.9"
  UV_VERSION: "0.1.44"

jobs:
  quality-checks:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install UV
        run: |
          pip install uv==${{ env.UV_VERSION }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/uv
          key: ${{ runner.os }}-uv-${{ hashFiles('pyproject.toml') }}
          restore-keys: |
            ${{ runner.os }}-uv-

      - name: Install dependencies
        run: |
          uv install --dev

      - name: Check formatting
        run: |
          black --check --diff src tests

      - name: Check import sorting
        run: |
          isort --check-only --diff src tests

      - name: Lint code
        run: |
          ruff check src tests

      - name: Type checking
        run: |
          mypy src

      - name: Security check
        run: |
          bandit -r src -f json

  test:
    name: Test Suite
    runs-on: ubuntu-latest
    timeout-minutes: 15
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install UV
        run: |
          pip install uv==${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv install --dev

      - name: Run unit tests
        run: |
          pytest tests/unit -v --cov=src --cov-report=xml

      - name: Run integration tests
        run: |
          pytest tests/integration -v

      - name: Upload coverage to Codecov
        if: matrix.python-version == '3.9'
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    timeout-minutes: 20
    if: github.event_name == 'pull_request'

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install UV
        run: |
          pip install uv==${{ env.UV_VERSION }}

      - name: Install dependencies
        run: |
          uv install --dev

      - name: Run performance tests
        run: |
          pytest tests/performance -v --benchmark-only

  dependency-security:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 5

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install UV
        run: |
          pip install uv==${{ env.UV_VERSION }}

      - name: Security audit
        run: |
          uv pip install safety
          safety check --json

  build-validation:
    name: Build Validation
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install UV
        run: |
          pip install uv==${{ env.UV_VERSION }}

      - name: Build package
        run: |
          uv build

      - name: Validate package
        run: |
          pip install dist/*.whl
          python -c "import reputenet; print(reputenet.__version__)"
```

---

## Development Scripts

### Quality Check Script

#### scripts/quality_check.sh
```bash
#!/bin/bash
# Quality check script for local development

set -e

echo "🔍 Running ReputeNet code quality checks..."
echo "============================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    print_warning "Not in a virtual environment. Activate venv first:"
    echo "source .venv/bin/activate"
    exit 1
fi

# 1. Format code
print_status "1. Formatting code with Black..."
if black src tests; then
    print_success "Code formatting completed"
else
    print_error "Code formatting failed"
    exit 1
fi

# 2. Sort imports
print_status "2. Sorting imports with isort..."
if isort src tests; then
    print_success "Import sorting completed"
else
    print_error "Import sorting failed"
    exit 1
fi

# 3. Lint code
print_status "3. Linting code with Ruff..."
if ruff check src tests --fix; then
    print_success "Linting completed"
else
    print_error "Linting failed"
    exit 1
fi

# 4. Type checking
print_status "4. Type checking with mypy..."
if mypy src; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found (not blocking)"
fi

# 5. Security check
print_status "5. Security check with Bandit..."
if bandit -r src -f json -o bandit-report.json; then
    print_success "Security check passed"
else
    print_warning "Security issues found - check bandit-report.json"
fi

# 6. Run tests
print_status "6. Running unit tests..."
if pytest tests/unit -v --cov=src --cov-report=term; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# 7. Run integration tests
print_status "7. Running integration tests..."
if pytest tests/integration -v; then
    print_success "Integration tests passed"
else
    print_warning "Integration tests failed (may require setup)"
fi

# 8. Configuration validation
print_status "8. Validating configuration..."
if python -c "from config import get_settings, validate_configuration; settings = get_settings(); issues = validate_configuration(settings); exit(len(issues))"; then
    print_success "Configuration validation passed"
else
    print_warning "Configuration validation issues found"
fi

echo ""
print_success "All quality checks completed! 🎉"
echo "Ready to commit and push to repository."
```

### Development Setup Script

#### scripts/setup_dev.sh
```bash
#!/bin/bash
# Development environment setup script

set -e

echo "🚀 Setting up ReputeNet development environment..."
echo "================================================"

# Check Python version
python_version=$(python3 --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.9"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install UV package manager
echo "⚡ Installing UV package manager..."
pip install uv

# Install dependencies
echo "📚 Installing project dependencies..."
uv install --dev

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Setup environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f ".env" ]; then
    python scripts/setup_env.py --environment development
    echo "📝 Created .env file for development"
else
    echo "📝 .env file already exists"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p data/mock
mkdir -p reports

# Validate setup
echo "✅ Validating development setup..."
python scripts/quality_check.sh --validate || echo "⚠️  Setup validation had warnings (may be expected)"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run quality checks:"
echo "  ./scripts/quality_check.sh"
echo ""
echo "To start development:"
echo "  pytest tests/unit  # Run unit tests"
echo "  black src tests    # Format code"
echo "  ruff check src tests # Lint code"
```

This implementation provides a robust, professional development infrastructure that balances comprehensive tooling with development velocity, perfect for the prototype's needs while being ready for production scaling.