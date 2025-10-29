# Step 4: Basic Project Infrastructure - Design Questions

**Context:** Development tooling setup for professional prototype development
**Decision Point:** Balancing comprehensive tooling with development velocity

---

## Critical Design Questions

### 1. Development Tooling Comprehensiveness
**Question:** How comprehensive should the development tooling be for a prototype?

**Context from Decisions:**
- Prototype targeting business and technical stakeholders
- Single developer initially, potential team growth
- Professional presentation requirements
- Development velocity important for prototype timeline

**Options:**
- **Comprehensive Tooling** ⭐ - Full suite (Black, Ruff, mypy, pytest, pre-commit, CI/CD)
- **Essential Tooling** - Basic formatting, linting, and testing
- **Minimal Tooling** - Just testing framework, manual quality checks
- **IDE-Dependent** - Rely on editor/IDE for quality checks

**Decision Needed:** Level of tooling sophistication for professional prototype?

### 2. Code Quality Enforcement Strategy
**Question:** How strict should automated code quality enforcement be?

**Options:**
- **Strict Enforcement** - Pre-commit hooks block commits that fail checks
- **Advisory Warnings** ⭐ - Tools warn but don't block development
- **CI-Only Enforcement** - Quality checks only in continuous integration
- **Manual Enforcement** - Developer responsibility, no automation

**Context:** Need balance between quality and rapid prototyping velocity

**Decision Needed:** Enforcement approach that maintains quality without blocking development?

### 3. Testing Infrastructure Sophistication
**Question:** How sophisticated should the testing infrastructure be?

**Options:**
- **Comprehensive Testing** ⭐ - Unit, integration, coverage, performance tests
- **Basic Unit Testing** - Simple pytest setup with basic fixtures
- **Manual Testing** - No automated tests, manual validation only
- **End-to-End Only** - Focus on system-level testing

**Context:** Prototype needs reliability but also rapid iteration

**Decision Needed:** Testing approach that ensures quality while supporting development speed?

### 4. CI/CD Pipeline Complexity
**Question:** How complex should the continuous integration pipeline be?

**Options:**
- **Full CI/CD Pipeline** ⭐ - Testing, quality checks, automated deployment
- **Basic CI** - Just testing and linting on pull requests
- **Manual Integration** - No automated pipeline, manual testing
- **Future Implementation** - Skip CI/CD for prototype phase

**Context:** Professional prototype targeting stakeholders, future production needs

**Decision Needed:** CI/CD sophistication appropriate for prototype phase?

---

## Secondary Design Questions

### 5. Tool Configuration Management
**Question:** How should development tool configuration be managed?

**Options:**
- **Centralized Configuration** ⭐ - All tools configured in pyproject.toml
- **Individual Config Files** - Separate configuration file for each tool
- **IDE Configuration** - Tool settings managed in IDE/editor
- **Environment Variables** - Configuration via environment variables

### 6. Code Style Strictness
**Question:** How strict should code formatting and style rules be?

**Options:**
- **Opinionated Formatting** ⭐ - Black with minimal configuration, strict consistency
- **Configurable Style** - Customizable formatting rules for team preferences
- **Loose Standards** - Basic style guidelines, manual enforcement
- **No Enforcement** - Developer choice, no standardization

### 7. Type Checking Approach
**Question:** How aggressive should static type checking be?

**Options:**
- **Strict Type Checking** - mypy in strict mode with comprehensive coverage
- **Gradual Typing** ⭐ - Progressive type hints, moderate mypy settings
- **Basic Type Hints** - Type hints for public APIs only
- **No Type Checking** - Skip static type analysis

### 8. Pre-commit Hook Strategy
**Question:** What should be included in pre-commit hooks?

**Options:**
- **Comprehensive Hooks** ⭐ - Format, lint, type check, test critical paths
- **Format Only** - Just code formatting and import sorting
- **Lint Only** - Code quality checks without formatting
- **No Pre-commit** - All checks in CI pipeline only

---

## Recommended Decisions

### ✅ High Confidence Recommendations

1. **Comprehensive Tooling with Balanced Enforcement** ⭐
   - **Rationale:** Professional prototype requires professional tooling
   - **Implementation:** Full toolchain with configurable strictness for rapid prototyping

2. **Advisory Quality Enforcement** ⭐
   - **Rationale:** Maintain velocity while encouraging quality practices
   - **Implementation:** Pre-commit warnings, CI enforcement, bypass options

3. **Comprehensive Testing Infrastructure** ⭐
   - **Rationale:** Prototype reliability important for stakeholder confidence
   - **Implementation:** pytest with fixtures, coverage, and integration tests

4. **Full CI/CD Pipeline** ⭐
   - **Rationale:** Professional appearance and future production readiness
   - **Implementation:** GitHub Actions with testing, quality checks, and deployment prep

---

## Impact on Implementation

### Development Toolchain
**Core Tools:**
- **Black:** Code formatting with minimal configuration
- **Ruff:** Fast, comprehensive linting replacing multiple tools
- **mypy:** Gradual type checking with reasonable strictness
- **pytest:** Feature-rich testing with coverage and fixtures
- **pre-commit:** Automated quality checks with bypass options

**Tool Configuration Strategy:**
```toml
# pyproject.toml - Centralized tool configuration
[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # Line too long (handled by Black)

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Gradual typing

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=src --cov-report=html --cov-report=term"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/venv/*"]
```

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        args: [--preview]

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.292
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit --maxfail=1 -q
        language: system
        pass_filenames: false
        always_run: true
```

### CI/CD Pipeline Structure
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install uv
          uv install --dev

      - name: Format check
        run: black --check src tests

      - name: Lint check
        run: ruff check src tests

      - name: Type check
        run: mypy src

      - name: Test
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Testing Infrastructure
```python
# tests/conftest.py
import pytest
import tempfile
from pathlib import Path
from config import Settings, load_settings

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def test_settings():
    """Get test configuration."""
    return load_settings("testing")

@pytest.fixture
def mock_database(test_settings):
    """Create test database."""
    # Database setup logic
    yield
    # Cleanup logic

# tests/unit/conftest.py
import pytest
from unittest.mock import AsyncMock, Mock

@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = AsyncMock()
    llm.ainvoke.return_value = "Mock response"
    return llm

@pytest.fixture
def mock_web3():
    """Mock Web3 provider for testing."""
    web3 = Mock()
    web3.eth.get_block.return_value = {"number": 12345}
    return web3
```

### Quality Gates and Enforcement
```bash
# scripts/quality_check.sh
#!/bin/bash
set -e

echo "Running code quality checks..."

echo "1. Format check..."
black --check src tests

echo "2. Import sorting..."
isort --check-only src tests

echo "3. Linting..."
ruff check src tests

echo "4. Type checking..."
mypy src

echo "5. Testing..."
pytest --cov=src --cov-report=term --cov-fail-under=80

echo "All quality checks passed! ✅"
```

---

## Development Workflow Integration

### Developer Onboarding Process
1. **Clone repository**
2. **Run setup script:** `scripts/setup_dev.sh`
3. **Install pre-commit hooks:** `pre-commit install`
4. **Validate setup:** `scripts/quality_check.sh`
5. **Start development**

### Daily Development Flow
```
Write code
    ↓
Pre-commit hooks (format, lint, basic tests)
    ↓
Commit (with semantic commit message)
    ↓
Push to feature branch
    ↓
CI pipeline runs full checks
    ↓
Create pull request
    ↓
Code review + automated checks
    ↓
Merge to main
```

### Bypass Mechanisms
```bash
# Skip pre-commit for urgent fixes
git commit -m "urgent fix" --no-verify

# Skip specific checks
SKIP=mypy git commit -m "feature: work in progress"

# Run checks manually
pre-commit run --all-files
```

---

## Next Steps

1. **Configure core tooling** in pyproject.toml with balanced settings
2. **Set up pre-commit hooks** with warning mode for development velocity
3. **Create comprehensive test structure** with fixtures and coverage
4. **Implement GitHub Actions CI/CD** with quality gates
5. **Document development workflow** with setup automation
6. **Create quality check scripts** for local development validation