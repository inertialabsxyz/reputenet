# Step 5: Dependency Injection and Service Layer - Design Questions

**Context:** Service layer architecture with dependency injection for component management
**Decision Point:** DI complexity vs development velocity and testing requirements

---

## Critical Design Questions

### 1. DI Container Sophistication Level
**Question:** How sophisticated should the dependency injection container be?

**Options:**
- **Full-Featured DI Container** ⭐ - Complete DI with scoping, lifecycle, circular dependency detection
- **Simple Service Registry** - Basic service registration and lookup
- **Factory Pattern Only** - Just factory methods without central container
- **Manual Dependency Management** - No DI framework, manual component wiring

**Context:** Multi-agent system with complex dependencies, need testing flexibility

**Decision Needed:** DI sophistication that balances professional patterns with development speed?

### 2. Service Lifecycle Management
**Question:** How comprehensive should service lifecycle management be?

**Options:**
- **Complete Lifecycle** ⭐ - Initialization, startup, health checks, graceful shutdown
- **Basic Startup/Shutdown** - Simple start and stop methods
- **No Lifecycle Management** - Services manage their own lifecycle
- **Lazy Initialization Only** - Create services on first use, no explicit lifecycle

**Context:** Services need proper initialization and cleanup for production readiness

**Decision Needed:** Lifecycle management approach that ensures reliable service operation?

### 3. Configuration Integration Strategy
**Question:** How should configuration be integrated with dependency injection?

**Options:**
- **Automatic Configuration Binding** ⭐ - DI container automatically injects configuration
- **Manual Configuration Passing** - Pass configuration explicitly to services
- **Configuration Service** - Centralized configuration service injected into other services
- **Environment-Based Factories** - Different factories for different environments

**Context:** Need to switch between mock and real implementations based on configuration

**Decision Needed:** Configuration integration that supports environment-specific component selection?

### 4. Service Scoping Strategy
**Question:** What service scoping strategies should be supported?

**Options:**
- **Multiple Scopes** ⭐ - Singleton, transient, request-scoped services
- **Singleton Only** - All services are singletons
- **Transient Only** - New instance every time
- **No Scoping** - Manual instance management

**Context:** Different services have different lifecycle requirements

**Decision Needed:** Scoping approach that matches service requirements and performance needs?

---

## Secondary Design Questions

### 5. Interface vs Implementation Registration
**Question:** Should services be registered by interface or implementation?

**Options:**
- **Interface-Based Registration** ⭐ - Register services by abstract interface
- **Implementation Registration** - Register concrete implementations directly
- **Hybrid Approach** - Mix of interface and implementation registration
- **Name-Based Registration** - Register services by string names

### 6. Circular Dependency Handling
**Question:** How should circular dependencies be handled?

**Options:**
- **Automatic Detection and Resolution** ⭐ - Container detects and resolves circular dependencies
- **Proxy-Based Resolution** - Use proxies to break circular dependencies
- **Manual Breaking** - Developer responsible for breaking cycles
- **Prohibition** - Disallow circular dependencies entirely

### 7. Testing Integration Complexity
**Question:** How sophisticated should testing integration be?

**Options:**
- **Test-Specific DI Container** ⭐ - Separate container configuration for tests
- **Mock Registration** - Easy registration of mocks in main container
- **Test Fixtures** - Pre-configured test service setups
- **Manual Test Setup** - Manual component setup for each test

---

## Recommended Decisions

### ✅ High Confidence Recommendations

1. **Full-Featured DI Container with Professional Patterns** ⭐
   - **Rationale:** Multi-agent system complexity requires sophisticated dependency management
   - **Implementation:** Custom DI container with scoping, lifecycle, and configuration binding

2. **Complete Service Lifecycle Management** ⭐
   - **Rationale:** Production readiness requires proper service initialization and cleanup
   - **Implementation:** Startup, health checks, graceful shutdown for all services

3. **Automatic Configuration Binding** ⭐
   - **Rationale:** Seamless environment switching requires configuration-driven component selection
   - **Implementation:** DI container automatically injects appropriate configuration into services

4. **Multiple Service Scopes with Smart Defaults** ⭐
   - **Rationale:** Different services have different lifecycle and performance requirements
   - **Implementation:** Singleton for infrastructure, transient for agents, request-scoped for workflow

---

## Impact on Implementation

### DI Container Architecture
```python
# Service registration patterns
class ServiceContainer:
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register singleton service."""

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register transient service."""

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register factory-created service."""

    def register_instance(self, interface: Type[T], instance: T) -> None:
        """Register pre-created instance."""

# Configuration-driven registration
def register_tools(container: ServiceContainer, config: ToolConfig):
    if config.mode == "mock":
        container.register_singleton(EthProviderTool, MockEthProvider)
        container.register_singleton(EtherscanTool, MockEtherscan)
    else:
        container.register_singleton(EthProviderTool, RealEthProvider)
        container.register_singleton(EtherscanTool, RealEtherscan)

# Agent with dependency injection
class AddressProfilerAgent:
    def __init__(
        self,
        eth_provider: EthProviderTool,
        etherscan: EtherscanTool,
        cache_service: CacheService,
        metrics_service: MetricsService
    ):
        self.eth_provider = eth_provider
        self.etherscan = etherscan
        self.cache = cache_service
        self.metrics = metrics_service
```

### Service Lifecycle Framework
```python
class ServiceLifecycle:
    async def initialize(self) -> None:
        """Initialize service (load config, validate settings)."""

    async def start(self) -> None:
        """Start service (connect to external resources)."""

    async def health_check(self) -> HealthStatus:
        """Check service health."""

    async def stop(self) -> None:
        """Gracefully stop service."""

class ServiceManager:
    def __init__(self, container: ServiceContainer):
        self.container = container
        self.services: List[ServiceLifecycle] = []

    async def start_all_services(self) -> None:
        """Start all services in dependency order."""
        for service in self.services:
            await service.initialize()
            await service.start()

    async def stop_all_services(self) -> None:
        """Stop all services in reverse dependency order."""
        for service in reversed(self.services):
            await service.stop()
```

### Configuration Integration
```python
# Automatic configuration injection
@dataclass
class DatabaseServiceConfig:
    url: str
    pool_size: int = 10
    timeout: float = 30.0

class DatabaseService(ServiceLifecycle):
    def __init__(self, config: DatabaseServiceConfig):
        self.config = config
        self.pool = None

    async def initialize(self) -> None:
        # Validate configuration
        if not self.config.url:
            raise ValueError("Database URL is required")

    async def start(self) -> None:
        # Create connection pool
        self.pool = create_pool(
            self.config.url,
            pool_size=self.config.pool_size,
            timeout=self.config.timeout
        )

# Registration with configuration binding
def register_database_service(container: ServiceContainer, config: Settings):
    db_config = DatabaseServiceConfig(
        url=config.database.url,
        pool_size=config.database.pool_size,
        timeout=config.database.timeout
    )

    container.register_singleton(
        DatabaseService,
        lambda: DatabaseService(db_config)
    )
```

### Testing Integration
```python
# Test-specific container configuration
class TestServiceContainer(ServiceContainer):
    def __init__(self):
        super().__init__()
        self._register_test_services()

    def _register_test_services(self):
        # Register mocks for all external dependencies
        self.register_singleton(EthProviderTool, MockEthProvider)
        self.register_singleton(DatabaseService, InMemoryDatabase)
        self.register_singleton(CacheService, MockCacheService)

# Test fixture for agent testing
@pytest.fixture
def address_profiler_agent(test_container: TestServiceContainer):
    """Create AddressProfiler with injected test dependencies."""
    return test_container.resolve(AddressProfilerAgent)

# Integration test with partial mocks
def create_integration_test_container():
    container = ServiceContainer()

    # Use real database but mock external APIs
    container.register_singleton(DatabaseService, RealDatabaseService)
    container.register_singleton(EthProviderTool, MockEthProvider)

    return container
```

### Service Factory Pattern
```python
class AgentFactory:
    def __init__(self, container: ServiceContainer):
        self.container = container

    def create_data_harvester(self) -> DataHarvesterAgent:
        """Create DataHarvester with all dependencies."""
        return DataHarvesterAgent(
            eth_provider=self.container.resolve(EthProviderTool),
            etherscan=self.container.resolve(EtherscanTool),
            cache_service=self.container.resolve(CacheService),
            metrics_service=self.container.resolve(MetricsService)
        )

    def create_address_profiler(self) -> AddressProfilerAgent:
        """Create AddressProfiler with all dependencies."""
        return AddressProfilerAgent(
            eth_provider=self.container.resolve(EthProviderTool),
            etherscan=self.container.resolve(EtherscanTool),
            cache_service=self.container.resolve(CacheService),
            metrics_service=self.container.resolve(MetricsService)
        )

    def create_all_agents(self) -> Dict[str, Any]:
        """Create all agents for workflow."""
        return {
            "data_harvester": self.create_data_harvester(),
            "address_profiler": self.create_address_profiler(),
            "risk_scorer": self.create_risk_scorer(),
            "sybil_detector": self.create_sybil_detector(),
            "reputation_aggregator": self.create_reputation_aggregator(),
            "reporter": self.create_reporter()
        }
```

---

## Next Steps

1. **Implement core DI container** with registration and resolution
2. **Create service lifecycle framework** with startup/shutdown management
3. **Build configuration integration** for automatic dependency binding
4. **Implement service scoping** with singleton, transient, and request scopes
5. **Create test utilities** for easy mock injection and test containers
6. **Document DI patterns** and service creation guidelines