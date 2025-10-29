# Phase 1: Foundation Setup - Implementation Runbook

**Target:** Mock-first prototype for deployed service
**Context:** Single developer, no API keys, containerized deployment
**Duration:** 1-2 days

---

## Prerequisites

- Python 3.9+ installed
- Docker installed (for deployment testing)
- Git repository initialized
- UV package manager installed (`pip install uv`)

---

## Step-by-Step Implementation

### Step 1: Project Structure Setup (30 minutes)

```bash
# 1. Create directory structure
mkdir -p src/reputenet/{agents,tools,utils}
mkdir -p tests/{agents,tools,fixtures}
mkdir -p docs/{api,development,deployment}
mkdir -p scripts
mkdir -p mock_data

# 2. Create __init__.py files
touch src/reputenet/__init__.py
touch src/reputenet/agents/__init__.py
touch src/reputenet/tools/__init__.py
touch src/reputenet/utils/__init__.py
touch tests/__init__.py
touch tests/agents/__init__.py
touch tests/tools/__init__.py
```

### Step 2: Create Core Configuration Files (45 minutes)

#### 2.1 pyproject.toml
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

dependencies = [
    # Core LangChain
    "langchain>=0.2.0,<0.3.0",
    "langgraph>=0.2.0,<0.3.0",
    "langchain-openai>=0.1.0,<0.2.0",

    # Data & Validation
    "pydantic>=2.7.0,<3.0.0",
    "web3==6.11.3",
    "httpx>=0.27.0,<0.28.0",

    # Infrastructure
    "python-dotenv>=1.0.0,<2.0.0",
    "structlog>=24.1.0,<25.0.0",
    "diskcache>=5.6.0,<6.0.0",

    # Mock Data & Testing
    "faker>=20.0.0,<21.0.0",
    "responses>=0.24.0,<0.25.0",

    # Service Framework
    "fastapi>=0.104.0,<0.105.0",
    "uvicorn[standard]>=0.24.0,<0.25.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
    "factory-boy>=3.3.0",
    "freezegun>=1.2.0",
]

[project.scripts]
reputenet = "reputenet.cli:main"
reputenet-api = "reputenet.api:run"

# Tool configurations
[tool.black]
line-length = 88
target-version = ['py39']

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP"]
ignore = ["E501", "B008"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--cov=src/reputenet", "--cov-report=term-missing"]
```

#### 2.2 .env.template
```bash
# ReputeNet Configuration Template (Prototype)
# Copy to .env and customize as needed

# Operating Mode
MOCK_MODE=true
ENVIRONMENT=development

# Mock Data Configuration
MOCK_DATA_DIR=./mock_data
CACHE_DIR=./.cache

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# LangChain Configuration (for mock LLM)
OPENAI_API_KEY=mock-key-for-testing

# Future Real API Configuration (when available)
# RPC_URL_MAINNET=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
# ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
# ALCHEMY_API_KEY=YOUR_ALCHEMY_API_KEY
```

#### 2.3 Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install UV for fast dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN uv pip install --system .

# Copy source code
COPY src/ ./src/
COPY mock_data/ ./mock_data/

# Verify installation
RUN python -c "import reputenet; print('✅ ReputeNet installed successfully')"

# Expose API port
EXPOSE 8000

# Default command
CMD ["reputenet-api"]
```

#### 2.4 docker-compose.yml
```yaml
version: '3.8'

services:
  reputenet:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MOCK_MODE=true
      - LOG_LEVEL=INFO
    volumes:
      - ./.cache:/app/.cache
      - ./mock_data:/app/mock_data
```

### Step 3: Core Module Stubs (30 minutes)

#### 3.1 src/reputenet/__init__.py
```python
"""ReputeNet: Multi-Agent System for On-Chain Reputation Analysis."""

__version__ = "0.1.0"
__author__ = "Movement Labs"

from .schema import ReputationInput, ReputationOutput

__all__ = ["ReputationInput", "ReputationOutput"]
```

#### 3.2 src/reputenet/schema.py
```python
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
    reputation_score: int = Field(ge=0, le=100, description="Overall reputation score")
    components: Dict[str, float] = Field(description="Component scores")
    flags: List[str] = Field(default_factory=list, description="Risk flags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


class GraphState(BaseModel):
    """State shared across all nodes in the ReputeNet graph."""

    params: ReputationInput
    raw: Dict[str, Any] = Field(default_factory=dict)
    features: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    sybil: Dict[str, Any] = Field(default_factory=dict)
    reputation: Dict[str, Any] = Field(default_factory=dict)
    reports: Dict[str, Any] = Field(default_factory=dict)
```

#### 3.3 src/reputenet/config.py
```python
"""Configuration management for ReputeNet."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Application configuration."""

    # Operating mode
    mock_mode: bool = Field(default=True)
    environment: str = Field(default="development")

    # Directories
    mock_data_dir: str = Field(default="./mock_data")
    cache_dir: str = Field(default="./.cache")

    # Service configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # API keys (for future use)
    openai_api_key: Optional[str] = Field(default=None)
    rpc_url_mainnet: Optional[str] = Field(default=None)
    etherscan_api_key: Optional[str] = Field(default=None)
    alchemy_api_key: Optional[str] = Field(default=None)


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        mock_mode=os.getenv("MOCK_MODE", "true").lower() == "true",
        environment=os.getenv("ENVIRONMENT", "development"),
        mock_data_dir=os.getenv("MOCK_DATA_DIR", "./mock_data"),
        cache_dir=os.getenv("CACHE_DIR", "./.cache"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        rpc_url_mainnet=os.getenv("RPC_URL_MAINNET"),
        etherscan_api_key=os.getenv("ETHERSCAN_API_KEY"),
        alchemy_api_key=os.getenv("ALCHEMY_API_KEY"),
    )
```

#### 3.4 src/reputenet/cli.py
```python
"""Command-line interface for ReputeNet."""

import sys
from typing import List
from .config import load_config
from .schema import ReputationInput


def main():
    """Main CLI entry point."""
    config = load_config()

    if len(sys.argv) < 2:
        print("Usage: reputenet <address> [--chain-id <id>]")
        print("Example: reputenet 0x742c4af20a2e0c8e82be16ab44d9421b1b78e569")
        return

    address = sys.argv[1]

    # Basic input validation
    if not address.startswith("0x") or len(address) != 42:
        print("Error: Invalid Ethereum address format")
        return

    # Create input
    input_data = ReputationInput(targets=[address])

    print(f"🔍 Analyzing reputation for {address}")
    print(f"📊 Mode: {'Mock' if config.mock_mode else 'Live'}")
    print("⚠️  Full implementation coming in Phase 2-3")


if __name__ == "__main__":
    main()
```

#### 3.5 src/reputenet/api.py
```python
"""FastAPI service for ReputeNet."""

from fastapi import FastAPI, HTTPException
from .config import load_config
from .schema import ReputationInput, ReputationOutput


def create_app() -> FastAPI:
    """Create FastAPI application."""
    config = load_config()

    app = FastAPI(
        title="ReputeNet API",
        description="Multi-Agent System for On-Chain Reputation Analysis",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "mode": "mock" if config.mock_mode else "live",
            "version": "0.1.0"
        }

    @app.post("/analyze", response_model=ReputationOutput)
    async def analyze_reputation(input_data: ReputationInput):
        """Analyze reputation for given addresses."""
        if not input_data.targets:
            raise HTTPException(status_code=400, detail="No target addresses provided")

        # Mock response for now
        address = input_data.targets[0]
        return ReputationOutput(
            address=address,
            reputation_score=75,
            components={
                "activity_quality": 0.8,
                "risk": 0.1,
                "sybil": 0.2
            },
            flags=["mock_data"],
            metadata={"prototype": True}
        )

    return app


def run():
    """Run the API server."""
    import uvicorn
    config = load_config()

    uvicorn.run(
        "reputenet.api:create_app",
        factory=True,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run()
```

### Step 4: Basic Testing and Validation (30 minutes)

#### 4.1 tests/conftest.py
```python
"""Pytest configuration and fixtures."""

import pytest
from reputenet.config import Config
from reputenet.schema import ReputationInput


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return Config(
        mock_mode=True,
        environment="test",
        log_level="DEBUG"
    )


@pytest.fixture
def sample_address():
    """Sample Ethereum address for testing."""
    return "0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"


@pytest.fixture
def sample_input(sample_address):
    """Sample reputation input."""
    return ReputationInput(targets=[sample_address])
```

#### 4.2 tests/test_schema.py
```python
"""Test data schema validation."""

import pytest
from reputenet.schema import ReputationInput, ReputationOutput


def test_reputation_input_validation(sample_address):
    """Test ReputationInput validates correctly."""
    input_data = ReputationInput(targets=[sample_address])
    assert input_data.chain_id == 1
    assert len(input_data.targets) == 1
    assert input_data.lookback_days == 90


def test_reputation_output_validation(sample_address):
    """Test ReputationOutput validates correctly."""
    output = ReputationOutput(
        address=sample_address,
        reputation_score=75,
        components={"test": 0.5}
    )
    assert output.reputation_score == 75
    assert output.address == sample_address
```

### Step 5: Installation and Verification (15 minutes)

```bash
# 1. Copy environment template
cp .env.template .env

# 2. Install package in development mode
uv pip install -e ".[dev]"

# 3. Install pre-commit hooks
pre-commit install

# 4. Run basic tests
pytest tests/ -v

# 5. Test CLI
reputenet 0x742c4af20a2e0c8e82be16ab44d9421b1b78e569

# 6. Test API (in another terminal)
reputenet-api &
curl http://localhost:8000/health

# 7. Test container build
docker build -t reputenet .
docker run -p 8000:8000 reputenet

# 8. Verify code formatting
black --check src/ tests/
ruff check src/ tests/
```

---

## Success Validation

### ✅ Phase 1 Complete Checklist

1. **Project Structure** ✅
   - [ ] src/reputenet/ structure exists
   - [ ] All __init__.py files created
   - [ ] Mock data directory prepared

2. **Dependencies** ✅
   - [ ] UV installs all packages without conflicts
   - [ ] All imports work correctly
   - [ ] Development tools function

3. **Configuration** ✅
   - [ ] .env template exists and loads
   - [ ] Config validation works
   - [ ] Mock mode enabled by default

4. **Basic Interfaces** ✅
   - [ ] CLI command responds
   - [ ] API server starts and responds
   - [ ] Health check endpoint works

5. **Testing Infrastructure** ✅
   - [ ] pytest runs successfully
   - [ ] Basic tests pass
   - [ ] Code formatting works

6. **Containerization** ✅
   - [ ] Docker builds successfully
   - [ ] Container runs and responds
   - [ ] API accessible from container

---

## Next Steps

**Phase 1 ✅ COMPLETE** → Ready for **Phase 2: Core Infrastructure**

1. **Mock Data Generation** - Create realistic blockchain mock data
2. **LangGraph Implementation** - Set up agent orchestration
3. **Tool Adapters** - Build mock API interfaces
4. **Dependency Injection** - Create service container

**Estimated Phase 2 Duration:** 2-3 days