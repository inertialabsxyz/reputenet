# Step 2: Dependency Management - Approach Analysis

**Objective:** Install and configure all required dependencies for mock-first prototype
**Context:** No external API keys, deployed service target, single developer

---

## Core Dependencies (Required)

### LangChain Ecosystem
```toml
langchain = ">=0.2.0"
langgraph = ">=0.2.0"
langchain-openai = ">=0.1.0"  # For mock LLM calls
```

### Data Handling
```toml
pydantic = ">=2.7.0"         # Type validation and schemas
web3 = ">=6.0.0"             # Ethereum address validation and mock data
httpx = ">=0.27.0"           # HTTP client for mock APIs
```

### Infrastructure
```toml
python-dotenv = ">=1.0.0"    # Environment configuration
structlog = ">=24.1.0"       # Structured logging
diskcache = ">=5.6.0"        # Caching layer for mock responses
```

### Development Tools
```toml
pytest = ">=7.0.0"
pytest-asyncio = ">=0.21.0"
pytest-cov = ">=4.0.0"
black = ">=23.0.0"
ruff = ">=0.1.0"
mypy = ">=1.0.0"
pre-commit = ">=3.0.0"
```

### Mock and Testing
```toml
responses = ">=0.24.0"       # Mock HTTP responses
freezegun = ">=1.2.0"        # Mock time for testing
factory-boy = ">=3.3.0"     # Generate mock blockchain data
```

### Service Deployment
```toml
fastapi = ">=0.104.0"        # Web API framework
uvicorn = ">=0.24.0"         # ASGI server
gunicorn = ">=21.2.0"        # Production WSGI server
```

---

## Mock-First Strategy

### Approach: Realistic Mock Infrastructure
**Rationale:** Since no API keys are available, create comprehensive mocks that simulate real blockchain data patterns.

### Mock Data Libraries
```toml
faker = ">=20.0.0"           # Generate realistic fake data
mimesis = ">=11.0.0"         # Alternative fake data generator
```

### Mock Ethereum Data
- Use actual Ethereum address formats
- Generate realistic transaction patterns
- Create mock ERC-20/NFT interactions
- Simulate real protocol interactions (Uniswap, AAVE, etc.)

---

## Dependency Strategy

### Option 1: Minimal Dependencies (Recommended for Prototype)
**Pros:**
- Faster installation and container builds
- Fewer version conflicts
- Simpler maintenance

**Cons:**
- May need to implement more functionality manually
- Less feature-rich out of the box

### Option 2: Feature-Rich Dependencies
**Pros:**
- More built-in functionality
- Faster development

**Cons:**
- Larger container size
- More complex dependency resolution
- Potential version conflicts

**Decision:** Use minimal dependencies for prototype, can expand later

---

## Version Pinning Strategy

### Approach: Pin Major Versions, Allow Minor Updates
```toml
# Pin major versions for stability
langchain = ">=0.2.0,<0.3.0"
pydantic = ">=2.7.0,<3.0.0"

# Pin exact versions for critical dependencies
web3 = "6.11.3"  # Specific version for Ethereum compatibility
```

**Rationale:**
- Stability for prototype demonstration
- Allow minor updates for security patches
- Pin exact versions for blockchain-specific libraries

---

## pyproject.toml Structure

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

    # Mock Data
    "faker>=20.0.0,<21.0.0",
    "responses>=0.24.0,<0.25.0",

    # Service
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

production = [
    "gunicorn>=21.2.0",
]

[project.scripts]
reputenet = "reputenet.cli:main"
reputenet-api = "reputenet.api:run"
```

---

## Installation Verification

### Local Development
```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Verify installation
python -c "import reputenet; print('✅ Package imports')"
python -c "import langchain; print('✅ LangChain available')"
python -c "import web3; print('✅ Web3 available')"
```

### Container Build
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv pip install --system .

COPY src/ ./src/
RUN python -c "import reputenet; print('✅ Container build success')"
```

---

## Mock API Configuration

### Environment Template Update
```bash
# Mock API Configuration (no real keys needed)
MOCK_MODE=true
MOCK_DATA_DIR=./mock_data

# Future real API keys (when available)
# RPC_URL_MAINNET=
# ETHERSCAN_API_KEY=
# OPENAI_API_KEY=

# Service Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## Success Criteria

**Step 2 Complete When:**
1. ✅ All dependencies install without conflicts
2. ✅ Mock data libraries are functional
3. ✅ Basic imports work correctly
4. ✅ Container builds successfully
5. ✅ UV installation works reproducibly
6. ✅ Development tools (black, ruff, pytest) function

**Next Step:** Configuration system with mock API support