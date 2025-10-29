# Step 3: Tool Adapters with Mock Implementation - Implementation Notes

**Context:** Production-ready tool adapters with comprehensive mock backends
**Approach:** Full adapter pattern with realistic mock implementations and seamless real API migration

---

## Implementation Strategy

### Tool Adapter Framework
Based on design decisions, implementing:
- **Abstract interfaces** with mock and real implementations using adapter pattern
- **Production-grade rate limiting** with token bucket algorithm and configurable limits
- **Comprehensive error simulation** covering real-world failure scenarios
- **Integrated mock data system** ensuring consistency across tools
- **Multi-layer caching** for performance optimization
- **Configuration-driven selection** between mock and real implementations

### Architecture Overview
```python
# Core tool framework
tools/
├── base/
│   ├── adapter.py           # Abstract base tool classes
│   ├── rate_limiter.py      # Rate limiting framework
│   ├── cache.py            # Caching infrastructure
│   ├── errors.py           # Error definitions
│   └── middleware.py       # Request/response middleware
├── interfaces/
│   ├── eth_provider.py     # Ethereum RPC interface
│   ├── etherscan.py        # Etherscan API interface
│   ├── defillama.py        # DeFiLlama interface
│   └── label_registry.py   # Address labeling interface
├── implementations/mock/    # Comprehensive mock implementations
├── implementations/real/    # Production API implementations
└── factory.py              # Configuration-driven tool creation
```

---

## Core Framework Implementation

### Base Adapter Classes

#### tools/base/adapter.py
```python
"""Base tool adapter framework with rate limiting and caching."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Type, TypeVar
from dataclasses import dataclass
from datetime import datetime, timedelta

from pydantic import BaseModel
from schemas.base import BaseReputeNetModel
from .rate_limiter import RateLimiter
from .cache import CacheManager
from .errors import ToolError, RateLimitError, TimeoutError

T = TypeVar('T', bound=BaseModel)

@dataclass
class ToolConfig:
    """Configuration for tool adapters."""
    mode: str = "mock"  # "mock" or "real"
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 5
    timeout_seconds: float = 30.0
    cache_ttl_seconds: int = 300
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_metrics: bool = True

class ToolRequest(BaseReputeNetModel):
    """Base request model for all tools."""
    tool_name: str
    method: str
    parameters: Dict[str, Any] = {}
    timeout: Optional[float] = None
    cache_key: Optional[str] = None
    priority: int = 0  # Higher priority = processed first

class ToolResponse(BaseReputeNetModel):
    """Base response model for all tools."""
    tool_name: str
    method: str
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    cached: bool = False
    response_time_ms: float = 0.0
    rate_limited: bool = False

class ToolMetrics(BaseModel):
    """Tool performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limit_hits: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None

class BaseTool(ABC):
    """Abstract base class for all tool adapters."""

    def __init__(self, config: ToolConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        self.rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_rpm,
            burst_size=config.rate_limit_burst
        )
        self.cache = CacheManager(default_ttl=config.cache_ttl_seconds)
        self.metrics = ToolMetrics()

        # Request queue for rate limiting
        self.request_queue = asyncio.PriorityQueue()
        self.queue_processor_task = None

    async def execute(self, request: ToolRequest) -> ToolResponse:
        """Execute tool request with rate limiting, caching, and error handling."""
        start_time = time.time()

        try:
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.last_request_time = datetime.utcnow()

            # Check cache first
            if request.cache_key:
                cached_result = await self.cache.get(request.cache_key)
                if cached_result:
                    self.metrics.cache_hits += 1
                    response = ToolResponse(
                        tool_name=request.tool_name,
                        method=request.method,
                        success=True,
                        data=cached_result,
                        cached=True,
                        response_time_ms=(time.time() - start_time) * 1000
                    )
                    return response

            self.metrics.cache_misses += 1

            # Apply rate limiting
            if not await self.rate_limiter.acquire():
                self.metrics.rate_limit_hits += 1
                raise RateLimitError(f"Rate limit exceeded for {self.config.rate_limit_rpm} RPM")

            # Execute the actual request
            result = await self._execute_with_retry(request)

            # Cache successful results
            if request.cache_key and result.success:
                await self.cache.set(request.cache_key, result.data)

            # Update metrics
            self.metrics.successful_requests += 1
            response_time = (time.time() - start_time) * 1000
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.successful_requests - 1) + response_time) /
                self.metrics.successful_requests
            )

            result.response_time_ms = response_time
            return result

        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.error(f"Tool execution failed: {e}")

            return ToolResponse(
                tool_name=request.tool_name,
                method=request.method,
                success=False,
                error_message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )

    async def _execute_with_retry(self, request: ToolRequest) -> ToolResponse:
        """Execute request with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Set timeout
                timeout = request.timeout or self.config.timeout_seconds

                # Execute the actual implementation
                result = await asyncio.wait_for(
                    self._execute_impl(request),
                    timeout=timeout
                )

                return result

            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"Request timeout after {timeout}s")
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries and self._is_retryable_error(e):
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    break

        raise last_exception

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable."""
        retryable_errors = (
            TimeoutError,
            ConnectionError,
            # Add other retryable error types
        )
        return isinstance(error, retryable_errors)

    @abstractmethod
    async def _execute_impl(self, request: ToolRequest) -> ToolResponse:
        """Implement actual tool execution logic."""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics."""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.successful_requests / self.metrics.total_requests

        cache_hit_rate = 0.0
        total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = self.metrics.cache_hits / total_cache_requests

        return {
            "total_requests": self.metrics.total_requests,
            "success_rate": success_rate,
            "cache_hit_rate": cache_hit_rate,
            "average_response_time_ms": self.metrics.average_response_time,
            "rate_limit_hits": self.metrics.rate_limit_hits,
            "last_request": self.metrics.last_request_time
        }
```

### Rate Limiting Framework

#### tools/base/rate_limiter.py
```python
"""Token bucket rate limiting implementation."""

import asyncio
import time
from typing import Dict

class RateLimiter:
    """Token bucket rate limiter with burst capability."""

    def __init__(self, requests_per_minute: int, burst_size: int):
        self.rpm = requests_per_minute
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token, return True if successful."""
        async with self.lock:
            self._refill_tokens()

            if self.tokens > 0:
                self.tokens -= 1
                return True

            return False

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on rate (RPM to tokens per second)
        tokens_to_add = elapsed * (self.rpm / 60.0)
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = now

    def get_wait_time(self) -> float:
        """Get estimated wait time for next token."""
        if self.tokens > 0:
            return 0.0

        # Time needed to get one token
        return 60.0 / self.rpm

class GlobalRateLimiter:
    """Global rate limiter managing multiple tools."""

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}

    def add_tool(self, tool_name: str, rpm: int, burst: int):
        """Add rate limiter for specific tool."""
        self.limiters[tool_name] = RateLimiter(rpm, burst)

    async def acquire(self, tool_name: str) -> bool:
        """Acquire token for specific tool."""
        if tool_name not in self.limiters:
            return True  # No limit configured

        return await self.limiters[tool_name].acquire()

    def get_wait_time(self, tool_name: str) -> float:
        """Get wait time for specific tool."""
        if tool_name not in self.limiters:
            return 0.0

        return self.limiters[tool_name].get_wait_time()
```

---

## Mock Implementation Framework

### Mock Tool Base Class

#### tools/implementations/mock/base_mock.py
```python
"""Base mock tool implementation with realistic behavior."""

import random
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from tools.base.adapter import BaseTool, ToolRequest, ToolResponse, ToolConfig
from tools.base.errors import ToolError, ServiceUnavailableError, InvalidRequestError
from mock_data.generators.base import MockDataValidator

class ErrorScenario:
    """Mock error scenario configuration."""
    def __init__(self, error_type: str, probability: float, message: str):
        self.error_type = error_type
        self.probability = probability
        self.message = message

class MockToolBase(BaseTool):
    """Base class for mock tool implementations."""

    def __init__(self, config: ToolConfig, mock_data_source: Any = None):
        super().__init__(config)
        self.mock_data_source = mock_data_source
        self.validator = MockDataValidator()

        # Error scenarios for realistic testing
        self.error_scenarios = [
            ErrorScenario("rate_limit", 0.02, "Rate limit exceeded"),
            ErrorScenario("timeout", 0.01, "Request timeout"),
            ErrorScenario("service_unavailable", 0.005, "Service temporarily unavailable"),
            ErrorScenario("invalid_request", 0.01, "Invalid request parameters")
        ]

    async def _execute_impl(self, request: ToolRequest) -> ToolResponse:
        """Execute mock request with realistic behavior."""

        # Simulate network latency
        await self._simulate_latency()

        # Check for error injection
        await self._check_error_scenarios(request)

        # Execute mock-specific logic
        data = await self._execute_mock_logic(request)

        # Validate response data
        if self.config.enable_metrics:
            validation_result = self.validator.validate_dataset({"data": data})
            if not validation_result.is_valid:
                self.logger.warning(f"Mock data validation issues: {validation_result.issues}")

        return ToolResponse(
            tool_name=request.tool_name,
            method=request.method,
            success=True,
            data=data
        )

    async def _simulate_latency(self):
        """Simulate realistic API latency."""
        # Simulate network latency with some variance
        base_latency = 0.1  # 100ms base
        variance = random.uniform(0.05, 0.3)  # 50-300ms additional
        await asyncio.sleep(base_latency + variance)

    async def _check_error_scenarios(self, request: ToolRequest):
        """Check if any error scenarios should be triggered."""
        for scenario in self.error_scenarios:
            if random.random() < scenario.probability:
                if scenario.error_type == "rate_limit":
                    raise RateLimitError(scenario.message)
                elif scenario.error_type == "timeout":
                    # Simulate timeout by sleeping longer than expected
                    await asyncio.sleep(request.timeout or 60)
                elif scenario.error_type == "service_unavailable":
                    raise ServiceUnavailableError(scenario.message)
                elif scenario.error_type == "invalid_request":
                    raise InvalidRequestError(scenario.message)

    async def _execute_mock_logic(self, request: ToolRequest) -> Any:
        """Override this method in specific mock implementations."""
        raise NotImplementedError("Mock implementations must override _execute_mock_logic")

    def _get_consistent_mock_data(self, key: str, data_type: str) -> Any:
        """Get consistent mock data for a given key."""
        if self.mock_data_source:
            return self.mock_data_source.get_data(key, data_type)
        return None
```

### Ethereum Provider Mock

#### tools/implementations/mock/eth_provider_mock.py
```python
"""Mock Ethereum provider with realistic blockchain data."""

import random
from typing import Dict, Any, Optional, List

from tools.interfaces.eth_provider import EthProviderTool
from tools.implementations.mock.base_mock import MockToolBase
from tools.base.adapter import ToolRequest, ToolResponse

class MockEthProvider(MockToolBase, EthProviderTool):
    """Mock Ethereum provider using generated blockchain data."""

    async def _execute_mock_logic(self, request: ToolRequest) -> Any:
        """Execute Ethereum RPC mock logic."""
        method = request.method
        params = request.parameters

        if method == "eth_getBalance":
            return await self._get_balance(params.get("address"))
        elif method == "eth_getTransactionCount":
            return await self._get_transaction_count(params.get("address"))
        elif method == "eth_getBlockByNumber":
            return await self._get_block_by_number(params.get("block_number"))
        elif method == "eth_getTransactionByHash":
            return await self._get_transaction_by_hash(params.get("tx_hash"))
        else:
            raise InvalidRequestError(f"Unknown method: {method}")

    async def _get_balance(self, address: str) -> str:
        """Get mock balance for address."""
        # Use consistent mock data based on address
        mock_address_data = self._get_consistent_mock_data(address, "address")

        if mock_address_data:
            return mock_address_data.get("balance", "0")

        # Generate realistic balance if no mock data
        balance_eth = random.lognormal(-2, 2)  # Log-normal distribution
        balance_wei = max(0, int(balance_eth * 10**18))
        return hex(balance_wei)

    async def _get_transaction_count(self, address: str) -> str:
        """Get mock transaction count (nonce) for address."""
        mock_address_data = self._get_consistent_mock_data(address, "address")

        if mock_address_data:
            return hex(mock_address_data.get("transaction_count", 0))

        # Generate realistic transaction count
        tx_count = random.poisson(50)  # Poisson distribution
        return hex(tx_count)

    async def _get_block_by_number(self, block_number: int) -> Dict[str, Any]:
        """Get mock block data."""
        mock_block_data = self._get_consistent_mock_data(str(block_number), "block")

        if mock_block_data:
            return mock_block_data

        # Generate realistic block data
        return {
            "number": hex(block_number),
            "hash": "0x" + random.randbytes(32).hex(),
            "parentHash": "0x" + random.randbytes(32).hex(),
            "timestamp": hex(int(datetime.utcnow().timestamp())),
            "gasLimit": hex(30000000),
            "gasUsed": hex(random.randint(1000000, 29000000)),
            "transactions": [
                "0x" + random.randbytes(32).hex()
                for _ in range(random.randint(50, 300))
            ]
        }

    async def _get_transaction_by_hash(self, tx_hash: str) -> Dict[str, Any]:
        """Get mock transaction data."""
        mock_tx_data = self._get_consistent_mock_data(tx_hash, "transaction")

        if mock_tx_data:
            return mock_tx_data

        # Generate realistic transaction data
        return {
            "hash": tx_hash,
            "blockNumber": hex(random.randint(18000000, 18500000)),
            "transactionIndex": hex(random.randint(0, 200)),
            "from": "0x" + random.randbytes(20).hex(),
            "to": "0x" + random.randbytes(20).hex(),
            "value": hex(random.randint(0, 10**18)),
            "gas": hex(random.randint(21000, 500000)),
            "gasPrice": hex(random.randint(10**9, 100*10**9)),  # 1-100 gwei
            "input": "0x" + random.randbytes(random.randint(0, 100)).hex(),
            "status": hex(1 if random.random() > 0.05 else 0)  # 95% success rate
        }
```

---

## Tool Factory Implementation

#### tools/factory.py
```python
"""Tool factory for configuration-driven tool selection."""

from typing import Dict, Type, Any
from tools.base.adapter import BaseTool, ToolConfig
from tools.interfaces.eth_provider import EthProviderTool
from tools.implementations.mock.eth_provider_mock import MockEthProvider
from tools.implementations.real.eth_provider_real import RealEthProvider

class ToolFactory:
    """Factory for creating tool instances based on configuration."""

    def __init__(self):
        self._mock_tools: Dict[str, Type[BaseTool]] = {
            "eth_provider": MockEthProvider,
            # Add other mock tools
        }

        self._real_tools: Dict[str, Type[BaseTool]] = {
            "eth_provider": RealEthProvider,
            # Add other real tools
        }

    def create_tool(self, tool_name: str, config: ToolConfig, **kwargs) -> BaseTool:
        """Create tool instance based on configuration."""

        if config.mode == "mock":
            tool_class = self._mock_tools.get(tool_name)
        elif config.mode == "real":
            tool_class = self._real_tools.get(tool_name)
        else:
            raise ValueError(f"Unknown tool mode: {config.mode}")

        if not tool_class:
            raise ValueError(f"Unknown tool: {tool_name}")

        return tool_class(config, **kwargs)

    def create_tool_suite(self, suite_config: Dict[str, ToolConfig], **kwargs) -> Dict[str, BaseTool]:
        """Create a complete suite of tools."""
        tools = {}

        for tool_name, config in suite_config.items():
            tools[tool_name] = self.create_tool(tool_name, config, **kwargs)

        return tools

    def get_available_tools(self, mode: str = "mock") -> List[str]:
        """Get list of available tools for given mode."""
        if mode == "mock":
            return list(self._mock_tools.keys())
        elif mode == "real":
            return list(self._real_tools.keys())
        else:
            return []

# Global tool factory instance
tool_factory = ToolFactory()

# Configuration helper
def create_development_tools(mock_data_source: Any = None) -> Dict[str, BaseTool]:
    """Create tool suite for development environment."""

    # Development configuration with relaxed rate limits
    dev_config = ToolConfig(
        mode="mock",
        rate_limit_rpm=300,  # Higher limit for development
        cache_ttl_seconds=60,  # Shorter cache for development
        enable_retry=True,
        max_retries=2,  # Fewer retries for faster feedback
        enable_metrics=True
    )

    suite_config = {
        "eth_provider": dev_config,
        "etherscan": dev_config,
        "defillama": dev_config,
        "label_registry": dev_config
    }

    return tool_factory.create_tool_suite(suite_config, mock_data_source=mock_data_source)

def create_production_tools(credentials: Dict[str, str]) -> Dict[str, BaseTool]:
    """Create tool suite for production environment."""

    # Production configuration with realistic rate limits
    prod_config = ToolConfig(
        mode="real",
        rate_limit_rpm=60,  # Conservative production limits
        cache_ttl_seconds=300,  # Longer cache for production
        enable_retry=True,
        max_retries=3,
        timeout_seconds=30.0,
        enable_metrics=True
    )

    suite_config = {
        "eth_provider": prod_config,
        "etherscan": prod_config,
        "defillama": prod_config,
        "label_registry": prod_config
    }

    return tool_factory.create_tool_suite(suite_config, credentials=credentials)
```

This implementation provides a comprehensive, production-ready tool adapter framework with sophisticated mock implementations that maintain consistency with the generated mock data while providing realistic error scenarios and performance characteristics.