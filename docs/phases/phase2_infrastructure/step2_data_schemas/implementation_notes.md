# Step 2: Data Schemas and State Management - Implementation Notes

**Context:** Comprehensive type-safe schemas with optimized LangGraph state management
**Approach:** Modular Pydantic models with performance-optimized validation

---

## Implementation Strategy

### Schema Architecture
Based on design decisions, implementing:
- **Modular Pydantic models** with strict validation and performance optimization
- **LangGraph state components** with typed interfaces and immutable updates
- **Versioned schema evolution** supporting rapid prototype development
- **Context-aware validation** with different strictness levels
- **Performance caching** for validation and serialization

### File Structure
```
schemas/
├── __init__.py
├── base.py                    # Base models and validation framework
├── version.py                 # Schema versioning system
├── blockchain/
│   ├── __init__.py
│   ├── addresses.py           # Address and account models
│   ├── transactions.py        # Transaction and block models
│   ├── tokens.py             # ERC-20 and NFT models
│   ├── protocols.py          # DeFi protocol models
│   └── networks.py           # Blockchain network models
├── agents/
│   ├── __init__.py
│   ├── state.py              # LangGraph central state
│   ├── inputs.py             # Agent input models
│   ├── outputs.py            # Agent output models
│   ├── workflows.py          # Workflow status models
│   └── errors.py             # Agent error models
├── analysis/
│   ├── __init__.py
│   ├── risk.py               # Risk assessment models
│   ├── reputation.py         # Reputation scoring models
│   ├── behavior.py           # Behavioral analysis models
│   ├── patterns.py           # Pattern recognition models
│   └── reports.py            # Report and summary models
├── tools/
│   ├── __init__.py
│   ├── external_apis.py      # External API response models
│   ├── rate_limiting.py      # Rate limiting models
│   └── caching.py           # Cache models
└── validation/
    ├── __init__.py
    ├── framework.py          # Validation framework
    ├── cache.py             # Validation caching
    ├── rules.py             # Custom validation rules
    └── errors.py            # Error handling
```

---

## Core Schema Framework

### Base Models and Validation

#### schemas/base.py
```python
"""Base schema models and validation framework."""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.config import BaseConfig


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"        # Full validation, all rules, performance secondary
    STANDARD = "standard"    # Balanced validation for normal operations
    FAST = "fast"           # Minimal validation for performance-critical paths
    DISABLED = "disabled"   # No validation (testing/development only)


class BaseReputeNetModel(BaseModel):
    """Base model for all ReputeNet schemas."""

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    schema_version: str = "1.0.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config(BaseConfig):
        # Validation settings
        validate_assignment = True
        use_enum_values = True
        allow_population_by_field_name = True
        extra = "forbid"  # Strict - no unknown fields

        # Performance settings
        allow_reuse = True
        validate_all = True

        # JSON settings
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator('updated_at', always=True)
    def set_updated_at(cls, v, values):
        """Automatically set updated_at when model is modified."""
        return v or datetime.utcnow()

    def model_hash(self) -> str:
        """Generate hash for model instance (for caching)."""
        model_dict = self.dict(exclude={'created_at', 'updated_at', 'metadata'})
        model_json = json.dumps(model_dict, sort_keys=True, default=str)
        return hashlib.sha256(model_json.encode()).hexdigest()

    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata field."""
        self.metadata[key] = value
        self.updated_at = datetime.utcnow()

    @classmethod
    def get_schema_info(cls) -> Dict[str, Any]:
        """Get schema information for documentation."""
        return {
            "model_name": cls.__name__,
            "schema_version": "1.0.0",
            "fields": list(cls.__fields__.keys()),
            "required_fields": [
                name for name, field in cls.__fields__.items()
                if field.required
            ]
        }


class BaseBlockchainModel(BaseReputeNetModel):
    """Base model for blockchain-related data."""

    # Blockchain-specific metadata
    network: str = Field(default="ethereum", description="Blockchain network")
    block_number: Optional[int] = Field(None, ge=0, description="Block number")
    transaction_hash: Optional[str] = Field(None, regex=r"^0x[a-fA-F0-9]{64}$")

    @validator('transaction_hash')
    def validate_transaction_hash(cls, v):
        """Validate transaction hash format."""
        if v and (not v.startswith('0x') or len(v) != 66):
            raise ValueError('Invalid transaction hash format')
        return v

    class Config(BaseReputeNetModel.Config):
        # Blockchain-specific settings
        schema_extra = {
            "example": {
                "network": "ethereum",
                "block_number": 18500000,
                "transaction_hash": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
            }
        }


class BaseAgentModel(BaseReputeNetModel):
    """Base model for agent-related data."""

    # Agent metadata
    agent_type: str = Field(..., description="Type of agent")
    agent_version: str = Field(default="1.0.0", description="Agent version")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")

    # Performance tracking
    memory_usage: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    cpu_time: Optional[float] = Field(None, ge=0.0, description="CPU time in seconds")

    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Validate confidence score range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

    def add_performance_metrics(
        self,
        processing_time: float,
        memory_usage: int,
        cpu_time: float
    ) -> None:
        """Add performance metrics to the model."""
        self.processing_time = processing_time
        self.memory_usage = memory_usage
        self.cpu_time = cpu_time
        self.updated_at = datetime.utcnow()


class BaseAnalysisModel(BaseReputeNetModel):
    """Base model for analysis results."""

    # Analysis metadata
    analysis_type: str = Field(..., description="Type of analysis")
    analysis_version: str = Field(default="1.0.0", description="Analysis algorithm version")
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Data quality
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    data_completeness: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_reliability: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Evidence and reasoning
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    reasoning: Optional[str] = Field(None, description="Analysis reasoning")
    limitations: List[str] = Field(default_factory=list, description="Analysis limitations")

    def add_evidence(self, evidence_item: str) -> None:
        """Add evidence to the analysis."""
        self.evidence.append(evidence_item)
        self.updated_at = datetime.utcnow()

    def add_limitation(self, limitation: str) -> None:
        """Add limitation to the analysis."""
        self.limitations.append(limitation)
        self.updated_at = datetime.utcnow()


# Type variables for generic models
T = TypeVar('T', bound=BaseReputeNetModel)
```

### Schema Versioning System

#### schemas/version.py
```python
"""Schema versioning and migration system."""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel


class MigrationType(Enum):
    """Types of schema migrations."""
    FIELD_ADDED = "field_added"
    FIELD_REMOVED = "field_removed"
    FIELD_RENAMED = "field_renamed"
    FIELD_TYPE_CHANGED = "field_type_changed"
    MODEL_RESTRUCTURED = "model_restructured"


@dataclass
class SchemaMigration:
    """Schema migration definition."""
    from_version: str
    to_version: str
    migration_type: MigrationType
    description: str
    migration_function: Callable[[Dict[str, Any]], Dict[str, Any]]
    rollback_function: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


class SchemaVersion:
    """Schema version management."""

    def __init__(self, major: int, minor: int, patch: int):
        self.major = major
        self.minor = minor
        self.patch = patch

    @property
    def version_string(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.version_string

    def __eq__(self, other: 'SchemaVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: 'SchemaVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: 'SchemaVersion') -> bool:
        return self < other or self == other

    def is_compatible(self, other: 'SchemaVersion') -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major

    def is_backward_compatible(self, other: 'SchemaVersion') -> bool:
        """Check if this version is backward compatible with other."""
        if self.major != other.major:
            return False
        return self >= other


class SchemaRegistry:
    """Registry for schema versions and migrations."""

    def __init__(self):
        self.migrations: List[SchemaMigration] = []
        self.current_version = SchemaVersion(1, 0, 0)
        self.logger = logging.getLogger(__name__)

    def register_migration(self, migration: SchemaMigration) -> None:
        """Register a schema migration."""
        self.migrations.append(migration)
        self.logger.info(f"Registered migration: {migration.from_version} -> {migration.to_version}")

    def get_migration_path(self, from_version: str, to_version: str) -> List[SchemaMigration]:
        """Get migration path between versions."""
        # Simple implementation - in practice, would implement graph traversal
        relevant_migrations = [
            m for m in self.migrations
            if m.from_version == from_version and m.to_version == to_version
        ]
        return relevant_migrations

    def migrate_data(self, data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate data between schema versions."""
        migration_path = self.get_migration_path(from_version, to_version)

        if not migration_path:
            if from_version == to_version:
                return data
            raise ValueError(f"No migration path from {from_version} to {to_version}")

        migrated_data = data.copy()
        for migration in migration_path:
            try:
                migrated_data = migration.migration_function(migrated_data)
                self.logger.info(f"Applied migration: {migration.description}")
            except Exception as e:
                self.logger.error(f"Migration failed: {migration.description} - {e}")
                raise

        return migrated_data

    def validate_version_compatibility(self, data_version: str, schema_version: str) -> bool:
        """Validate if data version is compatible with schema version."""
        data_ver = SchemaVersion(*map(int, data_version.split('.')))
        schema_ver = SchemaVersion(*map(int, schema_version.split('.')))
        return schema_ver.is_backward_compatible(data_ver)


# Global schema registry
schema_registry = SchemaRegistry()


# Migration examples
def migrate_address_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate address schema from v1.0 to v1.1."""
    # Example: Add new 'labels' field if missing
    if 'labels' not in data:
        data['labels'] = []
    return data


def migrate_transaction_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate transaction schema from v1.0 to v1.1."""
    # Example: Rename 'gas_price' to 'gas_price_wei'
    if 'gas_price' in data and 'gas_price_wei' not in data:
        data['gas_price_wei'] = data.pop('gas_price')
    return data


# Register migrations
schema_registry.register_migration(SchemaMigration(
    from_version="1.0.0",
    to_version="1.1.0",
    migration_type=MigrationType.FIELD_ADDED,
    description="Add labels field to address schema",
    migration_function=migrate_address_v1_to_v2
))

schema_registry.register_migration(SchemaMigration(
    from_version="1.0.0",
    to_version="1.1.0",
    migration_type=MigrationType.FIELD_RENAMED,
    description="Rename gas_price to gas_price_wei in transaction schema",
    migration_function=migrate_transaction_v1_to_v2
))
```

---

## LangGraph State Management

### Central State Model

#### schemas/agents/state.py
```python
"""LangGraph central state management models."""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field
from ..base import BaseReputeNetModel, BaseAgentModel


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Individual agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowError(BaseModel):
    """Workflow error information."""
    agent_name: str
    error_type: str
    error_message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stack_trace: Optional[str] = None
    recoverable: bool = False


class AnalysisRequest(BaseReputeNetModel):
    """Request for address analysis."""
    target_address: str = Field(..., regex=r"^0x[a-fA-F0-9]{40}$")
    analysis_depth: str = Field(default="standard", description="Analysis depth level")
    include_patterns: List[str] = Field(default_factory=list)
    exclude_patterns: List[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=300, ge=30, le=3600)

    # Analysis configuration
    enable_risk_analysis: bool = Field(default=True)
    enable_sybil_detection: bool = Field(default=True)
    enable_reputation_scoring: bool = Field(default=True)
    enable_behavior_analysis: bool = Field(default=True)

    # Data collection settings
    transaction_limit: int = Field(default=1000, ge=10, le=10000)
    historical_days: int = Field(default=365, ge=1, le=1095)
    include_token_transfers: bool = Field(default=True)
    include_nft_activity: bool = Field(default=True)


class AgentExecutionState(BaseAgentModel):
    """Base state for individual agent execution."""
    status: AgentStatus = Field(default=AgentStatus.PENDING)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)

    def start_execution(self) -> None:
        """Mark agent execution as started."""
        self.status = AgentStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def complete_execution(self) -> None:
        """Mark agent execution as completed."""
        self.status = AgentStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def fail_execution(self, error_message: str) -> None:
        """Mark agent execution as failed."""
        self.status = AgentStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def can_retry(self) -> bool:
        """Check if agent can be retried."""
        return self.status == AgentStatus.FAILED and self.retry_count < self.max_retries

    def retry_execution(self) -> None:
        """Retry agent execution."""
        if not self.can_retry():
            raise ValueError("Agent cannot be retried")
        self.retry_count += 1
        self.status = AgentStatus.PENDING
        self.error_message = None
        self.started_at = None
        self.completed_at = None
        self.updated_at = datetime.utcnow()


class DataHarvesterState(AgentExecutionState):
    """State for DataHarvester agent."""
    agent_type: str = Field(default="data_harvester", const=True)

    # Input
    target_address: Optional[str] = None

    # Output
    collected_data: Optional[Dict[str, Any]] = None
    transaction_count: int = Field(default=0, ge=0)
    data_sources: List[str] = Field(default_factory=list)
    collection_errors: List[str] = Field(default_factory=list)

    # Metrics
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class AddressProfilerState(AgentExecutionState):
    """State for AddressProfiler agent."""
    agent_type: str = Field(default="address_profiler", const=True)

    # Input dependencies
    requires_data_harvester: bool = Field(default=True, const=True)

    # Output
    behavior_patterns: List[str] = Field(default_factory=list)
    sophistication_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    activity_patterns: Dict[str, Any] = Field(default_factory=dict)
    interaction_patterns: Dict[str, Any] = Field(default_factory=dict)

    # Feature analysis
    foundation_features: Dict[str, float] = Field(default_factory=dict)
    sophisticated_features: Dict[str, float] = Field(default_factory=dict)
    expert_features: Dict[str, float] = Field(default_factory=dict)


class RiskScorerState(AgentExecutionState):
    """State for RiskScorer agent."""
    agent_type: str = Field(default="risk_scorer", const=True)

    # Input dependencies
    requires_address_profiler: bool = Field(default=True, const=True)

    # Output
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    risk_level: Optional[str] = Field(None)
    risk_factors: List[str] = Field(default_factory=list)
    threat_indicators: Dict[str, float] = Field(default_factory=dict)

    # Risk categories
    security_risks: Dict[str, float] = Field(default_factory=dict)
    compliance_risks: Dict[str, float] = Field(default_factory=dict)
    operational_risks: Dict[str, float] = Field(default_factory=dict)


class SybilDetectorState(AgentExecutionState):
    """State for SybilDetector agent."""
    agent_type: str = Field(default="sybil_detector", const=True)

    # Input dependencies
    requires_address_profiler: bool = Field(default=True, const=True)

    # Output
    sybil_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    cluster_analysis: Dict[str, Any] = Field(default_factory=dict)
    related_addresses: List[str] = Field(default_factory=list)
    coordination_patterns: List[str] = Field(default_factory=list)

    # Detection results
    cluster_id: Optional[str] = None
    cluster_size: int = Field(default=1, ge=1)
    coordination_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class ReputationAggregatorState(AgentExecutionState):
    """State for ReputationAggregator agent."""
    agent_type: str = Field(default="reputation_aggregator", const=True)

    # Input dependencies
    requires_risk_scorer: bool = Field(default=True, const=True)
    requires_sybil_detector: bool = Field(default=True, const=True)

    # Output
    reputation_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reputation_components: Dict[str, float] = Field(default_factory=dict)
    confidence_intervals: Dict[str, tuple] = Field(default_factory=dict)

    # Aggregation metadata
    aggregation_method: str = Field(default="weighted_average")
    component_weights: Dict[str, float] = Field(default_factory=dict)
    data_quality_impact: Optional[float] = Field(None, ge=0.0, le=1.0)


class ReporterState(AgentExecutionState):
    """State for Reporter agent."""
    agent_type: str = Field(default="reporter", const=True)

    # Input dependencies
    requires_reputation_aggregator: bool = Field(default=True, const=True)

    # Output
    final_report: Optional[Dict[str, Any]] = None
    executive_summary: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None

    # Report configuration
    report_format: str = Field(default="comprehensive")
    include_visualizations: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)

    # Report metadata
    report_version: str = Field(default="1.0.0")
    generated_at: Optional[datetime] = None


class ReputeNetGraphState(BaseReputeNetModel):
    """Central LangGraph state for complete workflow."""

    # Workflow control
    analysis_request: AnalysisRequest
    workflow_status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    current_agent: Optional[str] = None
    next_agent: Optional[str] = None

    # Agent states
    data_harvester: DataHarvesterState = Field(default_factory=DataHarvesterState)
    address_profiler: AddressProfilerState = Field(default_factory=AddressProfilerState)
    risk_scorer: RiskScorerState = Field(default_factory=RiskScorerState)
    sybil_detector: SybilDetectorState = Field(default_factory=SybilDetectorState)
    reputation_aggregator: ReputationAggregatorState = Field(default_factory=ReputationAggregatorState)
    reporter: ReporterState = Field(default_factory=ReporterState)

    # Shared data (immutable references)
    shared_data: Dict[str, Any] = Field(default_factory=dict)

    # Workflow metadata
    workflow_id: str = Field(..., description="Unique workflow identifier")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_execution_time: float = Field(default=0.0, ge=0.0)

    # Error handling
    errors: List[WorkflowError] = Field(default_factory=list)
    retry_attempts: int = Field(default=0, ge=0)
    max_retry_attempts: int = Field(default=3, ge=0)

    # Performance tracking
    memory_usage_mb: Optional[float] = Field(None, ge=0.0)
    cpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0)

    def get_agent_state(self, agent_name: str) -> AgentExecutionState:
        """Get state for specific agent."""
        agent_states = {
            "data_harvester": self.data_harvester,
            "address_profiler": self.address_profiler,
            "risk_scorer": self.risk_scorer,
            "sybil_detector": self.sybil_detector,
            "reputation_aggregator": self.reputation_aggregator,
            "reporter": self.reporter
        }

        if agent_name not in agent_states:
            raise ValueError(f"Unknown agent: {agent_name}")

        return agent_states[agent_name]

    def update_agent_state(self, agent_name: str, updates: Dict[str, Any]) -> None:
        """Update specific agent state."""
        agent_state = self.get_agent_state(agent_name)
        for key, value in updates.items():
            if hasattr(agent_state, key):
                setattr(agent_state, key, value)
        agent_state.updated_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def start_workflow(self) -> None:
        """Start workflow execution."""
        self.workflow_status = WorkflowStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def complete_workflow(self) -> None:
        """Complete workflow execution."""
        self.workflow_status = WorkflowStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.total_execution_time = (self.completed_at - self.started_at).total_seconds()
        self.updated_at = datetime.utcnow()

    def fail_workflow(self, error: WorkflowError) -> None:
        """Fail workflow execution."""
        self.workflow_status = WorkflowStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.errors.append(error)
        self.updated_at = datetime.utcnow()

    def can_retry_workflow(self) -> bool:
        """Check if workflow can be retried."""
        return (
            self.workflow_status == WorkflowStatus.FAILED and
            self.retry_attempts < self.max_retry_attempts
        )

    def get_workflow_progress(self) -> Dict[str, Any]:
        """Get workflow progress summary."""
        agent_statuses = {
            "data_harvester": self.data_harvester.status,
            "address_profiler": self.address_profiler.status,
            "risk_scorer": self.risk_scorer.status,
            "sybil_detector": self.sybil_detector.status,
            "reputation_aggregator": self.reputation_aggregator.status,
            "reporter": self.reporter.status
        }

        completed_count = sum(1 for status in agent_statuses.values() if status == AgentStatus.COMPLETED)
        total_count = len(agent_statuses)

        return {
            "workflow_status": self.workflow_status,
            "current_agent": self.current_agent,
            "progress_percent": (completed_count / total_count) * 100,
            "agent_statuses": agent_statuses,
            "execution_time": self.total_execution_time,
            "error_count": len(self.errors)
        }
```

This implementation provides a comprehensive, type-safe schema system with performance optimization, validation caching, and sophisticated LangGraph state management that supports the full agent workflow while maintaining development velocity.