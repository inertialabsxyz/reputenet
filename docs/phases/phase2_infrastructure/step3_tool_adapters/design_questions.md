# Step 3: Tool Adapters with Mock Implementation - Design Questions

**Context:** External API tool adapter design for mock-first development with production migration path
**Decision Point:** Balancing realistic mock behavior with development velocity and production readiness

---

## Critical Design Questions

### 1. Mock vs Real Implementation Strategy
**Question:** How should mock and real implementations be architected for seamless migration?

**Options:**
- **Full Adapter Pattern** ⭐ - Abstract interfaces with mock and real implementations
- **Mock-First with Stubs** - Mock implementations with real API stubs
- **Configuration Switch** - Single implementation with mock/real mode switch
- **Separate Codebases** - Completely separate mock and real implementations

**Context:** Need realistic development environment with clear production migration path

**Decision Needed:** Architecture that supports both realistic prototyping and production deployment?

### 2. Rate Limiting and Quota Management
**Question:** How comprehensive should rate limiting be for a prototype?

**Options:**
- **Production-Grade Limits** ⭐ - Realistic rate limiting matching real APIs
- **Relaxed Development Limits** - Higher limits for faster development
- **No Rate Limiting** - Skip rate limiting for prototype simplicity
- **Configurable Limits** - Different limits for different environments

**Context:** Need to demonstrate production readiness while maintaining development velocity

**Decision Needed:** Rate limiting approach that balances realism with development needs?

### 3. Error Simulation Complexity
**Question:** How sophisticated should error simulation be in mock implementations?

**Options:**
- **Comprehensive Error Scenarios** ⭐ - Full spectrum of real-world errors
- **Basic Error Types** - Simple error injection for common cases
- **Happy Path Only** - Focus on successful responses for development speed
- **Scenario-Based Errors** - Configurable error scenarios for testing

**Context:** Error handling is critical for production readiness demonstration

**Decision Needed:** Error simulation approach that ensures robust error handling?

### 4. Mock Data Consistency Strategy
**Question:** How should mock tool responses stay consistent with generated mock data?

**Options:**
- **Integrated Mock Data** ⭐ - Tools use same mock data generation system
- **Static Fixtures** - Pre-generated response fixtures
- **Independent Mocks** - Each tool generates its own mock data
- **Hybrid Approach** - Mix of integrated and static data

**Context:** Consistency across tools essential for realistic agent testing

**Decision Needed:** Approach that maintains data consistency while supporting tool independence?

---

## Secondary Design Questions

### 5. Caching Strategy Sophistication
**Question:** How sophisticated should the caching layer be?

**Options:**
- **Multi-Layer Caching** ⭐ - In-memory, Redis, and persistent caching
- **Simple In-Memory** - Basic in-memory caching only
- **No Caching** - Skip caching for prototype simplicity
- **Tool-Specific Caching** - Different caching strategies per tool

### 6. Performance Monitoring Integration
**Question:** How should tool performance be monitored and reported?

**Options:**
- **Comprehensive Metrics** ⭐ - Response times, error rates, cache hit rates
- **Basic Logging** - Simple request/response logging
- **No Monitoring** - Skip monitoring for prototype
- **Agent-Level Only** - Monitor only at agent level, not tool level

### 7. Real API Integration Preparation
**Question:** How much real API integration code should be implemented now?

**Options:**
- **Complete Real Implementations** ⭐ - Full real API code for future use
- **Interface Definitions Only** - Just interfaces, implement real APIs later
- **Stub Implementations** - Basic real API stubs for testing
- **Documentation Only** - Document real API requirements

---

## Recommended Decisions

### ✅ High Confidence Recommendations

1. **Full Adapter Pattern with Abstract Interfaces** ⭐
   - **Rationale:** Clean separation enables seamless mock-to-real migration
   - **Implementation:** Abstract base classes with mock and real implementations

2. **Production-Grade Rate Limiting** ⭐
   - **Rationale:** Demonstrates production readiness and realistic behavior
   - **Implementation:** Configurable rate limits matching real API constraints

3. **Comprehensive Error Scenario Simulation** ⭐
   - **Rationale:** Robust error handling essential for stakeholder confidence
   - **Implementation:** Error scenario library covering real-world failure modes

4. **Integrated Mock Data with Cross-Tool Consistency** ⭐
   - **Rationale:** Consistency crucial for realistic multi-agent testing
   - **Implementation:** Shared mock data system with tool-specific adaptations

---

## Impact on Implementation

### Tool Architecture
```python
# Abstract interface pattern
class BaseTool(ABC):
    @abstractmethod
    async def execute(self, request: ToolRequest) -> ToolResponse:
        pass

class EthProviderTool(BaseTool):
    # Interface definition

class MockEthProvider(EthProviderTool):
    # Mock implementation using generated data

class RealEthProvider(EthProviderTool):
    # Real implementation for production

# Configuration-driven selection
def create_tool(tool_type: str, config: ToolConfig) -> BaseTool:
    if config.mode == "mock":
        return MockToolFactory.create(tool_type)
    else:
        return RealToolFactory.create(tool_type)
```

### Rate Limiting Framework
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_size: int):
        self.rpm = requests_per_minute
        self.burst = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()

    async def acquire(self) -> bool:
        self._refill_tokens()
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False

# Tool-specific rate limits
RATE_LIMITS = {
    "etherscan": RateLimiter(requests_per_minute=300, burst_size=5),
    "defillama": RateLimiter(requests_per_minute=120, burst_size=3),
    "ethereum_rpc": RateLimiter(requests_per_minute=600, burst_size=10)
}
```

### Error Simulation System
```python
class ErrorScenario:
    def __init__(self, error_type: str, probability: float, message: str):
        self.error_type = error_type
        self.probability = probability
        self.message = message

ERROR_SCENARIOS = {
    "rate_limited": ErrorScenario("RateLimitError", 0.05, "Rate limit exceeded"),
    "timeout": ErrorScenario("TimeoutError", 0.02, "Request timeout"),
    "service_unavailable": ErrorScenario("ServiceError", 0.01, "Service temporarily unavailable"),
    "invalid_response": ErrorScenario("ParseError", 0.01, "Invalid response format")
}

class MockToolWithErrors:
    def __init__(self, error_scenarios: List[ErrorScenario]):
        self.error_scenarios = error_scenarios

    async def execute(self, request: ToolRequest) -> ToolResponse:
        # Check for error injection
        for scenario in self.error_scenarios:
            if random.random() < scenario.probability:
                raise ToolError(scenario.error_type, scenario.message)

        # Normal execution
        return await self._normal_execution(request)
```

---

## Next Steps

1. **Define abstract tool interfaces** for all external APIs
2. **Implement comprehensive mock backends** with realistic behavior
3. **Build rate limiting and caching infrastructure** with production patterns
4. **Create error simulation framework** for robust testing
5. **Implement tool factory pattern** for configuration-driven selection
6. **Prepare real API implementations** for future migration