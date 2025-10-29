# Step 2: Data Schemas and State Management - Design Questions

**Context:** Type-safe data models and LangGraph state management design
**Decision Point:** Balancing type safety, performance, and development velocity

---

## Critical Design Questions

### 1. Schema Validation Strictness
**Question:** How strict should schema validation be throughout the system?

**Context from Decisions:**
- Multi-agent system requires reliable data flow
- Prototype needs development velocity
- Business stakeholders expect professional quality
- Mock data needs to validate consistently

**Options:**
- **Strict Validation** ⭐ - Full validation everywhere, fail fast on invalid data
- **Selective Validation** - Validate at boundaries, trust internal data
- **Development Mode** - Strict in production, relaxed in development
- **Lazy Validation** - Validate only when data is accessed

**Decision Needed:** Validation approach that ensures reliability without impacting development speed?

### 2. LangGraph State Complexity
**Question:** How complex should the central LangGraph state model be?

**Options:**
- **Monolithic State** - Single large state object containing all data
- **Modular State** ⭐ - Separate state components for different concerns
- **Agent-Specific State** - Each agent maintains its own state portion
- **Minimal State** - Only essential data in central state

**Context:** Need to support 6 agents with complex data dependencies

**Decision Needed:** State architecture that supports agent complexity while maintaining performance?

### 3. Schema Evolution Strategy
**Question:** How should schema changes be handled as the system evolves?

**Options:**
- **Versioned Schemas** ⭐ - Explicit schema versions with migration support
- **Backward Compatible** - Always maintain backward compatibility
- **Breaking Changes Allowed** - Accept breaking changes during prototype phase
- **Immutable Schemas** - Never change schemas, only add new ones

**Context:** Prototype will evolve rapidly, but need to support data persistence

**Decision Needed:** Schema evolution approach for prototype development?

### 4. Performance vs Type Safety Trade-off
**Question:** How should we balance type safety with runtime performance?

**Options:**
- **Type Safety Priority** ⭐ - Comprehensive validation, optimize where needed
- **Performance Priority** - Minimal validation for speed
- **Configurable Trade-off** - Different validation levels for different contexts
- **Lazy Optimization** - Start with safety, optimize based on profiling

**Context:** Agents need to process potentially large datasets quickly

**Decision Needed:** Approach that maintains professional quality while meeting performance needs?

---

## Secondary Design Questions

### 5. State Persistence Strategy
**Question:** How should agent state be persisted and restored?

**Options:**
- **Full State Serialization** ⭐ - Serialize complete state for checkpoints
- **Event Sourcing** - Store state changes as events
- **Selective Persistence** - Only persist critical state portions
- **No Persistence** - Ephemeral state for prototype

### 6. Validation Error Handling
**Question:** How should schema validation errors be handled and reported?

**Options:**
- **Fail Fast** ⭐ - Stop processing on validation errors
- **Collect and Report** - Collect all errors before failing
- **Graceful Degradation** - Continue with warnings for non-critical errors
- **User-Friendly Messages** - Transform technical errors to user-friendly messages

### 7. Schema Documentation Generation
**Question:** How should schema documentation be generated and maintained?

**Options:**
- **Auto-Generated Docs** ⭐ - Generate from Pydantic models
- **Manual Documentation** - Hand-written schema documentation
- **Hybrid Approach** - Generated base + manual examples
- **Code-Only Documentation** - Documentation through code comments

### 8. Cross-Schema Validation
**Question:** How should validation work across related schemas?

**Options:**
- **Reference Validation** ⭐ - Validate foreign key relationships
- **Eventual Consistency** - Allow temporary inconsistencies
- **No Cross-Validation** - Each schema validates independently
- **Graph Validation** - Validate entire object graphs

---

## Recommended Decisions

### ✅ High Confidence Recommendations

1. **Strict Validation with Performance Optimization** ⭐
   - **Rationale:** Professional prototype requires reliability, optimize bottlenecks
   - **Implementation:** Full validation with caching and fast paths

2. **Modular LangGraph State Architecture** ⭐
   - **Rationale:** Supports agent complexity while maintaining organization
   - **Implementation:** Separate state components with typed interfaces

3. **Versioned Schema Evolution** ⭐
   - **Rationale:** Supports rapid prototyping while enabling data persistence
   - **Implementation:** Schema versions with automatic migration support

4. **Type Safety Priority with Lazy Optimization** ⭐
   - **Rationale:** Start with correctness, optimize based on actual performance needs
   - **Implementation:** Comprehensive validation with performance monitoring

---

## Impact on Implementation

### Schema Architecture
**Core Components:**
- **Base Models:** Foundation classes with common validation patterns
- **Blockchain Schemas:** Complete models for all blockchain data types
- **Agent State Models:** LangGraph state components for workflow management
- **Analysis Schemas:** Models for risk assessment and reputation analysis
- **Validation Framework:** Centralized validation with performance optimization

**Validation Strategy:**
```python
# Validation levels
class ValidationLevel(Enum):
    STRICT = "strict"        # Full validation, all rules
    STANDARD = "standard"    # Common validation rules
    FAST = "fast"           # Minimal validation for performance
    DISABLED = "disabled"   # No validation (testing only)

# Context-aware validation
def validate_model(data: Dict, level: ValidationLevel = ValidationLevel.STANDARD):
    if level == ValidationLevel.DISABLED:
        return data
    elif level == ValidationLevel.FAST:
        return fast_validate(data)
    elif level == ValidationLevel.STANDARD:
        return standard_validate(data)
    else:  # STRICT
        return strict_validate(data)
```

### LangGraph State Design
```python
# Modular state components
class AgentState(BaseModel):
    """Base state for all agents."""
    agent_id: str
    status: AgentStatus
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class DataHarvesterState(AgentState):
    """State for data collection."""
    target_address: str
    collected_data: Optional[AddressData] = None
    collection_errors: List[str] = []

class AddressProfilerState(AgentState):
    """State for address profiling."""
    address_data: Optional[AddressData] = None
    behavior_patterns: List[BehaviorPattern] = []
    sophistication_score: Optional[float] = None

class ReputeNetGraphState(BaseModel):
    """Central LangGraph state."""
    # Core workflow data
    analysis_request: AnalysisRequest
    current_agent: str
    workflow_status: WorkflowStatus

    # Agent states
    data_harvester: DataHarvesterState
    address_profiler: AddressProfilerState
    risk_scorer: RiskScorerState
    sybil_detector: SybilDetectorState
    reputation_aggregator: ReputationAggregatorState
    reporter: ReporterState

    # Shared data
    address_data: Optional[AddressData] = None
    risk_assessment: Optional[RiskAssessment] = None
    reputation_score: Optional[ReputationScore] = None
    final_report: Optional[AnalysisReport] = None

    # Workflow metadata
    started_at: datetime
    updated_at: datetime
    execution_time: float = 0.0
    errors: List[WorkflowError] = []
```

### Schema Organization
```python
# Base schema patterns
class BaseBlockchainModel(BaseModel):
    """Base for all blockchain-related models."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
        use_enum_values = True
        extra = "forbid"  # Strict - no unknown fields

class BaseAgentModel(BaseModel):
    """Base for all agent-related models."""
    version: str = "1.0"
    agent_type: str
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        validate_assignment = True
        use_enum_values = True

# Schema versioning
class SchemaVersion(BaseModel):
    major: int
    minor: int
    patch: int

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible(self, other: "SchemaVersion") -> bool:
        return self.major == other.major
```

### Validation Framework
```python
# Performance-optimized validation
class ValidationCache:
    """Cache validation results for performance."""
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size

    def get_cached_validation(self, data_hash: str) -> Optional[Any]:
        return self.cache.get(data_hash)

    def cache_validation(self, data_hash: str, result: Any):
        if len(self.cache) >= self.max_size:
            # Remove oldest entries
            oldest_keys = list(self.cache.keys())[:self.max_size // 2]
            for key in oldest_keys:
                del self.cache[key]
        self.cache[data_hash] = result

# Fast validation for common patterns
def fast_validate_address(address: str) -> bool:
    """Fast address validation."""
    return len(address) == 42 and address.startswith("0x")

def fast_validate_transaction_hash(tx_hash: str) -> bool:
    """Fast transaction hash validation."""
    return len(tx_hash) == 66 and tx_hash.startswith("0x")

# Context-aware validation
class ValidationContext:
    """Validation context for different scenarios."""
    def __init__(
        self,
        level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cache: bool = True,
        strict_types: bool = True
    ):
        self.level = level
        self.enable_cache = enable_cache
        self.strict_types = strict_types
        self.cache = ValidationCache() if enable_cache else None

# Error handling
class SchemaValidationError(Exception):
    """Schema validation error with context."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.timestamp = datetime.utcnow()

class ValidationErrorCollector:
    """Collect multiple validation errors."""
    def __init__(self):
        self.errors: List[SchemaValidationError] = []

    def add_error(self, error: SchemaValidationError):
        self.errors.append(error)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def get_summary(self) -> str:
        return f"{len(self.errors)} validation errors"
```

---

## State Transition Validation

### Agent State Transitions
```python
# Valid state transitions
VALID_TRANSITIONS = {
    AgentStatus.PENDING: [AgentStatus.RUNNING, AgentStatus.FAILED],
    AgentStatus.RUNNING: [AgentStatus.COMPLETED, AgentStatus.FAILED],
    AgentStatus.COMPLETED: [AgentStatus.PENDING],  # For reruns
    AgentStatus.FAILED: [AgentStatus.PENDING, AgentStatus.RUNNING]  # For retries
}

def validate_state_transition(
    from_state: AgentStatus,
    to_state: AgentStatus
) -> bool:
    """Validate agent state transition."""
    return to_state in VALID_TRANSITIONS.get(from_state, [])

# State update validation
def update_graph_state(
    current_state: ReputeNetGraphState,
    updates: Dict[str, Any],
    validate: bool = True
) -> ReputeNetGraphState:
    """Update graph state with validation."""
    if validate:
        # Validate state transition
        current_agent = updates.get("current_agent", current_state.current_agent)
        new_status = updates.get("workflow_status", current_state.workflow_status)

        if not validate_workflow_transition(current_state.workflow_status, new_status):
            raise ValueError(f"Invalid workflow transition: {current_state.workflow_status} -> {new_status}")

    # Create new state (immutable update)
    return current_state.copy(update=updates)
```

---

## Next Steps

1. **Implement base schema framework** with validation and caching
2. **Create modular LangGraph state architecture** with typed components
3. **Design blockchain data models** with comprehensive validation
4. **Implement schema versioning system** for evolution support
5. **Create performance monitoring** for validation bottlenecks
6. **Build comprehensive test suite** using mock data validation