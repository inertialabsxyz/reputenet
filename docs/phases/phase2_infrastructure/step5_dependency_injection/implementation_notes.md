# Step 5: Dependency Injection and Service Layer - Implementation Notes

**Context:** Professional dependency injection container with comprehensive service lifecycle management
**Approach:** Full-featured DI with scoping, configuration binding, and testing integration

---

## Implementation Strategy

### Service Layer Architecture
Based on design decisions, implementing:
- **Full-featured DI container** with registration patterns, scoping, and lifecycle management
- **Automatic configuration binding** with environment-specific component selection
- **Complete service lifecycle** with initialization, startup, health checks, and graceful shutdown
- **Multiple service scopes** (singleton, transient, request-scoped) with performance optimization
- **Comprehensive testing integration** with test-specific containers and mock injection

### File Structure
```
services/
├── __init__.py
├── container.py              # Core DI container implementation
├── registry.py               # Service registration and discovery
├── factory.py                # Service factory implementations
├── lifecycle.py              # Service lifecycle management
├── scopes.py                 # Service scoping strategies
├── configuration.py          # Configuration binding integration
├── agents/
│   ├── __init__.py
│   ├── agent_factory.py      # Agent creation with DI
│   ├── agent_registry.py     # Agent service registration
│   └── agent_lifecycle.py    # Agent-specific lifecycle
├── tools/
│   ├── __init__.py
│   ├── tool_factory.py       # Tool creation with DI
│   ├── tool_registry.py      # Tool service registration
│   └── tool_lifecycle.py     # Tool-specific lifecycle
├── infrastructure/
│   ├── __init__.py
│   ├── database_service.py   # Database connection service
│   ├── cache_service.py      # Caching service implementation
│   ├── metrics_service.py    # Metrics collection service
│   └── logging_service.py    # Centralized logging service
└── testing/
    ├── __init__.py
    ├── test_container.py      # Test-specific DI container
    ├── mock_services.py       # Mock service implementations
    └── fixtures.py            # Test fixtures and utilities
```

---

## Core DI Container Implementation

### Service Container

#### services/container.py
```python
"""Core dependency injection container with full feature set."""

import asyncio
import inspect
import logging
from typing import (
    Type, TypeVar, Dict, Any, Optional, Callable, List, Union,
    get_type_hints, get_origin, get_args
)
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .scopes import ServiceScope, ScopeManager
from .lifecycle import ServiceLifecycle, LifecycleManager
from .configuration import ConfigurationBinder

T = TypeVar('T')

class RegistrationType(Enum):
    """Types of service registration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"
    INSTANCE = "instance"
    FACTORY = "factory"

@dataclass
class ServiceRegistration:
    """Service registration information."""
    interface: Type
    implementation: Optional[Type] = None
    factory: Optional[Callable] = None
    instance: Optional[Any] = None
    scope: ServiceScope = ServiceScope.SINGLETON
    registration_type: RegistrationType = RegistrationType.SINGLETON
    dependencies: List[Type] = None
    configuration_type: Optional[Type] = None

class ServiceContainer:
    """Comprehensive dependency injection container."""

    def __init__(self, config_binder: Optional[ConfigurationBinder] = None):
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._scope_manager = ScopeManager()
        self._lifecycle_manager = LifecycleManager()
        self._config_binder = config_binder
        self._logger = logging.getLogger(__name__)
        self._building = set()  # Circular dependency detection

    def register_singleton(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        configuration_type: Optional[Type] = None
    ) -> 'ServiceContainer':
        """Register a singleton service."""
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            scope=ServiceScope.SINGLETON,
            registration_type=RegistrationType.SINGLETON,
            configuration_type=configuration_type
        )

    def register_transient(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        configuration_type: Optional[Type] = None
    ) -> 'ServiceContainer':
        """Register a transient service (new instance each time)."""
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            scope=ServiceScope.TRANSIENT,
            registration_type=RegistrationType.TRANSIENT,
            configuration_type=configuration_type
        )

    def register_scoped(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        scope_name: str = "request",
        configuration_type: Optional[Type] = None
    ) -> 'ServiceContainer':
        """Register a scoped service (one instance per scope)."""
        return self._register_service(
            interface=interface,
            implementation=implementation,
            factory=factory,
            scope=ServiceScope.SCOPED,
            registration_type=RegistrationType.SCOPED,
            configuration_type=configuration_type
        )

    def register_instance(
        self,
        interface: Type[T],
        instance: T
    ) -> 'ServiceContainer':
        """Register a pre-created instance."""
        registration = ServiceRegistration(
            interface=interface,
            instance=instance,
            scope=ServiceScope.SINGLETON,
            registration_type=RegistrationType.INSTANCE
        )

        self._registrations[interface] = registration
        self._instances[interface] = instance

        # Register with lifecycle manager if applicable
        if isinstance(instance, ServiceLifecycle):
            self._lifecycle_manager.register_service(instance)

        return self

    def register_factory(
        self,
        interface: Type[T],
        factory: Callable[[], T],
        scope: ServiceScope = ServiceScope.TRANSIENT
    ) -> 'ServiceContainer':
        """Register a factory function."""
        return self._register_service(
            interface=interface,
            factory=factory,
            scope=scope,
            registration_type=RegistrationType.FACTORY
        )

    def _register_service(
        self,
        interface: Type[T],
        implementation: Optional[Type[T]] = None,
        factory: Optional[Callable[[], T]] = None,
        scope: ServiceScope = ServiceScope.SINGLETON,
        registration_type: RegistrationType = RegistrationType.SINGLETON,
        configuration_type: Optional[Type] = None
    ) -> 'ServiceContainer':
        """Internal service registration method."""

        # Determine implementation if not provided
        if implementation is None and factory is None:
            implementation = interface

        # Analyze dependencies
        dependencies = []
        if implementation:
            dependencies = self._analyze_dependencies(implementation)

        registration = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            factory=factory,
            scope=scope,
            registration_type=registration_type,
            dependencies=dependencies,
            configuration_type=configuration_type
        )

        self._registrations[interface] = registration
        self._logger.debug(f"Registered service: {interface.__name__} as {registration_type.value}")

        return self

    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance."""
        return asyncio.run(self.resolve_async(interface))

    async def resolve_async(self, interface: Type[T]) -> T:
        """Asynchronously resolve a service instance."""

        # Check for circular dependency
        if interface in self._building:
            raise ValueError(f"Circular dependency detected: {interface.__name__}")

        # Check if already instantiated (singleton)
        if interface in self._instances:
            return self._instances[interface]

        # Find registration
        registration = self._registrations.get(interface)
        if not registration:
            raise ValueError(f"Service not registered: {interface.__name__}")

        try:
            self._building.add(interface)

            # Create instance based on registration type
            if registration.registration_type == RegistrationType.INSTANCE:
                instance = registration.instance

            elif registration.registration_type == RegistrationType.FACTORY:
                instance = registration.factory()

            else:
                # Create instance using implementation
                instance = await self._create_instance(registration)

            # Manage instance based on scope
            if registration.scope == ServiceScope.SINGLETON:
                self._instances[interface] = instance

            # Register with lifecycle manager
            if isinstance(instance, ServiceLifecycle):
                self._lifecycle_manager.register_service(instance)

            return instance

        finally:
            self._building.discard(interface)

    async def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance with dependency injection."""

        implementation = registration.implementation
        if not implementation:
            raise ValueError(f"No implementation found for {registration.interface.__name__}")

        # Get constructor parameters
        constructor = implementation.__init__
        sig = inspect.signature(constructor)
        parameters = sig.parameters

        # Prepare constructor arguments
        kwargs = {}
        for param_name, param in parameters.items():
            if param_name == 'self':
                continue

            param_type = param.annotation
            if param_type == inspect.Parameter.empty:
                continue

            # Handle configuration binding
            if self._config_binder and registration.configuration_type:
                config_instance = self._config_binder.bind_configuration(registration.configuration_type)
                if param_type == registration.configuration_type:
                    kwargs[param_name] = config_instance
                    continue

            # Resolve dependency
            try:
                dependency = await self.resolve_async(param_type)
                kwargs[param_name] = dependency
            except ValueError as e:
                self._logger.warning(f"Could not resolve dependency {param_type.__name__} for {param_name}: {e}")
                # Use default value if available
                if param.default != inspect.Parameter.empty:
                    kwargs[param_name] = param.default

        # Create instance
        instance = implementation(**kwargs)

        self._logger.debug(f"Created instance: {implementation.__name__}")
        return instance

    def _analyze_dependencies(self, implementation: Type) -> List[Type]:
        """Analyze constructor dependencies for a service."""
        dependencies = []

        constructor = implementation.__init__
        sig = inspect.signature(constructor)

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            if param.annotation != inspect.Parameter.empty:
                dependencies.append(param.annotation)

        return dependencies

    async def start_services(self) -> None:
        """Start all registered services."""
        await self._lifecycle_manager.start_all()

    async def stop_services(self) -> None:
        """Stop all registered services."""
        await self._lifecycle_manager.stop_all()

    def get_registrations(self) -> Dict[Type, ServiceRegistration]:
        """Get all service registrations (for debugging)."""
        return self._registrations.copy()

    def is_registered(self, interface: Type) -> bool:
        """Check if a service is registered."""
        return interface in self._registrations
```

### Configuration Integration

#### services/configuration.py
```python
"""Configuration binding for dependency injection."""

from typing import Type, Dict, Any, Optional
from dataclasses import fields, is_dataclass

from config import Settings, get_settings

class ConfigurationBinder:
    """Bind configuration objects to service constructors."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._config_cache: Dict[Type, Any] = {}

    def bind_configuration(self, config_type: Type) -> Any:
        """Create configuration instance for service."""

        # Check cache first
        if config_type in self._config_cache:
            return self._config_cache[config_type]

        # Create configuration instance
        if is_dataclass(config_type):
            config_instance = self._create_dataclass_config(config_type)
        else:
            # Try to create using constructor
            config_instance = self._create_constructor_config(config_type)

        # Cache and return
        self._config_cache[config_type] = config_instance
        return config_instance

    def _create_dataclass_config(self, config_type: Type) -> Any:
        """Create dataclass configuration from settings."""
        field_values = {}

        for field in fields(config_type):
            field_name = field.name
            field_value = self._get_config_value(field_name, field.type)

            if field_value is not None:
                field_values[field_name] = field_value
            elif field.default is not None:
                field_values[field_name] = field.default

        return config_type(**field_values)

    def _create_constructor_config(self, config_type: Type) -> Any:
        """Create configuration using constructor."""
        # This would need more sophisticated logic based on constructor signature
        return config_type()

    def _get_config_value(self, field_name: str, field_type: Type) -> Any:
        """Get configuration value from settings."""

        # Map configuration fields to settings
        field_mapping = {
            "database_url": self.settings.database.url,
            "cache_ttl": self.settings.api.timeout,
            "rate_limit_rpm": self.settings.api.rate_limit,
            "log_level": self.settings.logging.level,
            # Add more mappings as needed
        }

        return field_mapping.get(field_name)
```

---

## Service Factory Implementation

### Agent Factory with DI

#### services/agents/agent_factory.py
```python
"""Agent factory with dependency injection."""

from typing import Dict, Any, Type

from reputenet.agents import (
    DataHarvesterAgent,
    AddressProfilerAgent,
    RiskScorerAgent,
    SybilDetectorAgent,
    ReputationAggregatorAgent,
    ReporterAgent
)
from services.container import ServiceContainer

class AgentFactory:
    """Factory for creating agents with dependency injection."""

    def __init__(self, container: ServiceContainer):
        self.container = container

    async def create_data_harvester(self) -> DataHarvesterAgent:
        """Create DataHarvester agent with all dependencies."""
        return await self.container.resolve_async(DataHarvesterAgent)

    async def create_address_profiler(self) -> AddressProfilerAgent:
        """Create AddressProfiler agent with all dependencies."""
        return await self.container.resolve_async(AddressProfilerAgent)

    async def create_risk_scorer(self) -> RiskScorerAgent:
        """Create RiskScorer agent with all dependencies."""
        return await self.container.resolve_async(RiskScorerAgent)

    async def create_sybil_detector(self) -> SybilDetectorAgent:
        """Create SybilDetector agent with all dependencies."""
        return await self.container.resolve_async(SybilDetectorAgent)

    async def create_reputation_aggregator(self) -> ReputationAggregatorAgent:
        """Create ReputationAggregator agent with all dependencies."""
        return await self.container.resolve_async(ReputationAggregatorAgent)

    async def create_reporter(self) -> ReporterAgent:
        """Create Reporter agent with all dependencies."""
        return await self.container.resolve_async(ReporterAgent)

    async def create_all_agents(self) -> Dict[str, Any]:
        """Create all agents for the workflow."""
        return {
            "data_harvester": await self.create_data_harvester(),
            "address_profiler": await self.create_address_profiler(),
            "risk_scorer": await self.create_risk_scorer(),
            "sybil_detector": await self.create_sybil_detector(),
            "reputation_aggregator": await self.create_reputation_aggregator(),
            "reporter": await self.create_reporter()
        }

class AgentRegistry:
    """Registry for agent service registration."""

    @staticmethod
    def register_agents(container: ServiceContainer, config: Settings) -> None:
        """Register all agents with the DI container."""

        # Register each agent as transient (new instance per workflow)
        container.register_transient(DataHarvesterAgent)
        container.register_transient(AddressProfilerAgent)
        container.register_transient(RiskScorerAgent)
        container.register_transient(SybilDetectorAgent)
        container.register_transient(ReputationAggregatorAgent)
        container.register_transient(ReporterAgent)

        # Register the agent factory as singleton
        container.register_singleton(AgentFactory, lambda: AgentFactory(container))
```

---

## Infrastructure Services

### Database Service with Lifecycle

#### services/infrastructure/database_service.py
```python
"""Database service with connection pooling and lifecycle management."""

import asyncio
import logging
from typing import Optional, Any, Dict
from dataclasses import dataclass

import asyncpg
from services.lifecycle import ServiceLifecycle

@dataclass
class DatabaseConfig:
    """Database service configuration."""
    url: str
    pool_size: int = 10
    timeout: float = 30.0
    command_timeout: float = 60.0
    max_retries: int = 3

class DatabaseService(ServiceLifecycle):
    """Database service with connection pooling."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self) -> None:
        """Initialize the database service."""
        self.logger.info("Initializing database service")

        # Validate configuration
        if not self.config.url:
            raise ValueError("Database URL is required")

    async def start(self) -> None:
        """Start the database service and create connection pool."""
        self.logger.info(f"Starting database service with pool size {self.config.pool_size}")

        try:
            self.pool = await asyncpg.create_pool(
                self.config.url,
                min_size=1,
                max_size=self.config.pool_size,
                command_timeout=self.config.command_timeout,
                timeout=self.config.timeout
            )

            self.logger.info("Database connection pool created successfully")

        except Exception as e:
            self.logger.error(f"Failed to create database pool: {e}")
            raise

    async def stop(self) -> None:
        """Stop the database service and close connection pool."""
        if self.pool:
            self.logger.info("Closing database connection pool")
            await self.pool.close()
            self.pool = None

    async def health_check(self) -> bool:
        """Check database connectivity."""
        if not self.pool:
            return False

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            self.logger.warning(f"Database health check failed: {e}")
            return False

    async def execute(self, query: str, *args) -> Any:
        """Execute a database query."""
        if not self.pool:
            raise RuntimeError("Database service not started")

        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list:
        """Fetch query results."""
        if not self.pool:
            raise RuntimeError("Database service not started")

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch single row."""
        if not self.pool:
            raise RuntimeError("Database service not started")

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
```

---

## Testing Integration

### Test Container and Fixtures

#### services/testing/test_container.py
```python
"""Test-specific dependency injection container."""

from typing import Dict, Any
from unittest.mock import Mock, AsyncMock

from services.container import ServiceContainer
from services.configuration import ConfigurationBinder
from tools.interfaces.eth_provider import EthProviderTool
from tools.interfaces.etherscan import EtherscanTool
from services.infrastructure.database_service import DatabaseService
from services.infrastructure.cache_service import CacheService

class TestServiceContainer(ServiceContainer):
    """Service container configured for testing."""

    def __init__(self, config_binder: ConfigurationBinder = None):
        super().__init__(config_binder)
        self._register_test_services()

    def _register_test_services(self):
        """Register mock services for testing."""

        # Mock external tool dependencies
        self.register_singleton(EthProviderTool, MockEthProviderTool)
        self.register_singleton(EtherscanTool, MockEtherscanTool)

        # Mock infrastructure services
        self.register_singleton(DatabaseService, MockDatabaseService)
        self.register_singleton(CacheService, MockCacheService)

        # Register test utilities
        self.register_singleton(TestDataGenerator, TestDataGenerator)

class MockEthProviderTool:
    """Mock Ethereum provider for testing."""

    def __init__(self):
        self.get_balance = AsyncMock(return_value="1000000000000000000")  # 1 ETH
        self.get_transaction_count = AsyncMock(return_value=42)
        self.get_block = AsyncMock(return_value={"number": 18500000})

class MockEtherscanTool:
    """Mock Etherscan tool for testing."""

    def __init__(self):
        self.get_transactions = AsyncMock(return_value=[])
        self.get_contract_abi = AsyncMock(return_value={})

class MockDatabaseService:
    """Mock database service for testing."""

    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.execute = AsyncMock()
        self.fetch = AsyncMock(return_value=[])
        self.fetchrow = AsyncMock(return_value=None)

    async def initialize(self):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def health_check(self):
        return True

class MockCacheService:
    """Mock cache service for testing."""

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: int = 300):
        self.cache[key] = value

    async def delete(self, key: str):
        self.cache.pop(key, None)

    async def clear(self):
        self.cache.clear()

class TestDataGenerator:
    """Generate test data for unit tests."""

    def create_sample_address_data(self) -> Dict[str, Any]:
        """Create sample address data for testing."""
        return {
            "address": "0x1234567890123456789012345678901234567890",
            "balance": "1000000000000000000",
            "transaction_count": 42,
            "labels": ["test_address"]
        }

    def create_sample_transaction_data(self) -> Dict[str, Any]:
        """Create sample transaction data for testing."""
        return {
            "hash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "from": "0x1111111111111111111111111111111111111111",
            "to": "0x2222222222222222222222222222222222222222",
            "value": "1000000000000000000",
            "gas": "21000",
            "gasPrice": "20000000000"
        }

# Test fixtures
import pytest

@pytest.fixture
def test_container():
    """Create test container with mock services."""
    return TestServiceContainer()

@pytest.fixture
def test_data_generator():
    """Create test data generator."""
    return TestDataGenerator()

@pytest.fixture
async def address_profiler_agent(test_container):
    """Create AddressProfiler agent with test dependencies."""
    from reputenet.agents import AddressProfilerAgent
    return await test_container.resolve_async(AddressProfilerAgent)
```

---

## Application Startup Integration

#### services/application.py
```python
"""Application startup with dependency injection."""

import asyncio
import logging
from typing import Dict, Any

from config import get_settings, Settings
from services.container import ServiceContainer
from services.configuration import ConfigurationBinder
from services.agents.agent_registry import AgentRegistry
from services.tools.tool_registry import ToolRegistry
from services.infrastructure import (
    DatabaseService,
    CacheService,
    MetricsService,
    LoggingService
)

class Application:
    """Main application with dependency injection."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)

        # Create configuration binder
        self.config_binder = ConfigurationBinder(self.settings)

        # Create DI container
        self.container = ServiceContainer(self.config_binder)

        # Register all services
        self._register_services()

    def _register_services(self):
        """Register all application services."""

        # Register infrastructure services
        self._register_infrastructure_services()

        # Register tools based on configuration
        ToolRegistry.register_tools(self.container, self.settings)

        # Register agents
        AgentRegistry.register_agents(self.container, self.settings)

    def _register_infrastructure_services(self):
        """Register core infrastructure services."""

        # Database service
        self.container.register_singleton(
            DatabaseService,
            configuration_type=DatabaseConfig
        )

        # Cache service
        self.container.register_singleton(
            CacheService,
            configuration_type=CacheConfig
        )

        # Metrics service
        self.container.register_singleton(MetricsService)

        # Logging service
        self.container.register_singleton(LoggingService)

    async def start(self):
        """Start the application."""
        self.logger.info("Starting ReputeNet application")

        try:
            # Start all services
            await self.container.start_services()

            self.logger.info("Application started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start application: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the application."""
        self.logger.info("Stopping ReputeNet application")

        try:
            # Stop all services
            await self.container.stop_services()

            self.logger.info("Application stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during application shutdown: {e}")

    def get_container(self) -> ServiceContainer:
        """Get the DI container."""
        return self.container

# Global application instance
app = Application()

async def main():
    """Main application entry point."""
    try:
        await app.start()

        # Keep application running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Received shutdown signal")
    finally:
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

This comprehensive dependency injection implementation provides enterprise-grade service management with automatic configuration binding, sophisticated lifecycle management, and comprehensive testing support that enables rapid development while maintaining production readiness.