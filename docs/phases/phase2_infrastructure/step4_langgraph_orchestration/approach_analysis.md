# Step 4: LangGraph Orchestration Framework - Approach Analysis

**Objective:** Implement the complete agent workflow using LangGraph
**Context:** Mock-first prototype with 6-node agent pipeline
**Estimated Duration:** 3-4 hours

---

## LangGraph Workflow Design

### Node Architecture (From Spec)
```
DataHarvester → AddressProfiler → RiskScorer → SybilDetector → ReputationAggregator → Reporter
```

### State Flow Pattern
```python
GraphState = {
    params: ReputationInput,      # Input parameters
    raw: Dict[str, Any],          # DataHarvester output
    features: Dict[str, Any],     # AddressProfiler output
    risk: Dict[str, Any],         # RiskScorer output
    sybil: Dict[str, Any],        # SybilDetector output
    reputation: Dict[str, Any],   # ReputationAggregator output
    reports: Dict[str, Any]       # Reporter output
}
```

---

## Implementation Approach

### Option 1: Simple Linear Pipeline (Recommended for Prototype)
```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

def build_reputation_graph() -> StateGraph:
    """Build simple linear pipeline."""

    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("harvest", node_data_harvest)
    workflow.add_node("profile", node_address_profile)
    workflow.add_node("risk", node_risk_score)
    workflow.add_node("sybil", node_sybil_detect)
    workflow.add_node("aggregate", node_reputation_aggregate)
    workflow.add_node("report", node_generate_report)

    # Linear flow
    workflow.set_entry_point("harvest")
    workflow.add_edge("harvest", "profile")
    workflow.add_edge("profile", "risk")
    workflow.add_edge("risk", "sybil")
    workflow.add_edge("sybil", "aggregate")
    workflow.add_edge("aggregate", "report")
    workflow.add_edge("report", END)

    return workflow
```

**Pros:**
- Simple to implement and debug
- Clear execution order
- Easy to understand flow
- Matches specification exactly

**Cons:**
- No parallelization opportunities
- Single failure point per node
- Less flexible for future enhancements

### Option 2: Parallel Processing with Synchronization
```python
def build_parallel_reputation_graph() -> StateGraph:
    """Build pipeline with parallel risk/sybil analysis."""

    workflow = StateGraph(GraphState)

    workflow.add_node("harvest", node_data_harvest)
    workflow.add_node("profile", node_address_profile)
    workflow.add_node("risk", node_risk_score)
    workflow.add_node("sybil", node_sybil_detect)
    workflow.add_node("sync", node_sync_analysis)  # Synchronization point
    workflow.add_node("aggregate", node_reputation_aggregate)
    workflow.add_node("report", node_generate_report)

    # Flow with parallelization
    workflow.set_entry_point("harvest")
    workflow.add_edge("harvest", "profile")
    workflow.add_edge("profile", "risk")
    workflow.add_edge("profile", "sybil")  # Parallel execution
    workflow.add_edge("risk", "sync")
    workflow.add_edge("sybil", "sync")
    workflow.add_edge("sync", "aggregate")
    workflow.add_edge("aggregate", "report")
    workflow.add_edge("report", END)

    return workflow
```

**Pros:**
- Faster execution through parallelization
- More efficient resource usage
- Scalable architecture

**Cons:**
- More complex synchronization logic
- Harder to debug failures
- Overkill for prototype with mock data

### Option 3: Conditional Flow with Error Handling
```python
def build_robust_reputation_graph() -> StateGraph:
    """Build pipeline with error handling and conditional flows."""

    workflow = StateGraph(GraphState)

    # Add nodes with error handling
    workflow.add_node("harvest", node_data_harvest_with_retry)
    workflow.add_node("validate_data", node_validate_harvested_data)
    workflow.add_node("profile", node_address_profile)
    workflow.add_node("risk", node_risk_score)
    workflow.add_node("sybil", node_sybil_detect)
    workflow.add_node("aggregate", node_reputation_aggregate)
    workflow.add_node("report", node_generate_report)
    workflow.add_node("error_handler", node_handle_errors)

    # Conditional edges based on validation
    workflow.set_entry_point("harvest")
    workflow.add_edge("harvest", "validate_data")
    workflow.add_conditional_edges(
        "validate_data",
        lambda state: "continue" if state.get("data_valid") else "error",
        {"continue": "profile", "error": "error_handler"}
    )

    return workflow
```

**Pros:**
- Robust error handling
- Data validation at each step
- Production-ready patterns

**Cons:**
- Significantly more complex
- Longer development time
- May obscure core logic during prototyping

---

## Recommended Approach: Simple Linear Pipeline

For the prototype, implement Option 1 with these enhancements:

### Enhanced Linear Pipeline
```python
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import structlog
from ..schema import GraphState, ReputationInput
from ..tools.mock_provider import MockDataProvider

logger = structlog.get_logger()

class ReputationGraph:
    """LangGraph orchestration for reputation analysis."""

    def __init__(self, data_provider: MockDataProvider):
        self.data_provider = data_provider
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the reputation analysis graph."""
        workflow = StateGraph(GraphState)

        # Add all processing nodes
        workflow.add_node("harvest", self._node_data_harvest)
        workflow.add_node("profile", self._node_address_profile)
        workflow.add_node("risk", self._node_risk_score)
        workflow.add_node("sybil", self._node_sybil_detect)
        workflow.add_node("aggregate", self._node_reputation_aggregate)
        workflow.add_node("report", self._node_generate_report)

        # Linear execution flow
        workflow.set_entry_point("harvest")
        workflow.add_edge("harvest", "profile")
        workflow.add_edge("profile", "risk")
        workflow.add_edge("risk", "sybil")
        workflow.add_edge("sybil", "aggregate")
        workflow.add_edge("aggregate", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    def analyze_reputation(self, input_data: ReputationInput) -> Dict[str, Any]:
        """Execute complete reputation analysis."""
        logger.info("Starting reputation analysis", targets=input_data.targets)

        # Initialize state
        initial_state = GraphState(
            params=input_data,
            raw={},
            features={},
            risk={},
            sybil={},
            reputation={},
            reports={}
        )

        # Execute graph
        final_state = self.graph.invoke(initial_state.dict())

        logger.info("Reputation analysis complete",
                   addresses=len(final_state["reports"]))

        return final_state["reports"]
```

---

## Node Implementation Strategy

### 1. DataHarvester Node (Mock Implementation)
```python
def _node_data_harvest(self, state: GraphState) -> GraphState:
    """Harvest blockchain data for target addresses."""
    logger.info("Harvesting data", targets=state["params"]["targets"])

    raw_data = {}

    for address in state["params"]["targets"]:
        # Use mock data provider
        address_data = self.data_provider.get_address_data(
            address=address,
            lookback_days=state["params"]["lookback_days"],
            max_txs=state["params"]["max_txs"]
        )

        raw_data[address] = {
            "transactions": address_data["transactions"],
            "logs": address_data["logs"],
            "tokens": address_data["tokens"],
            "metadata": {
                "first_seen": address_data["first_seen"],
                "last_seen": address_data["last_seen"],
                "total_txs": len(address_data["transactions"])
            }
        }

    state["raw"] = raw_data
    logger.info("Data harvest complete", addresses=len(raw_data))

    return state
```

### 2. AddressProfiler Node
```python
def _node_address_profile(self, state: GraphState) -> GraphState:
    """Extract behavioral features from raw data."""
    logger.info("Profiling addresses")

    features = {}

    for address, raw_data in state["raw"].items():
        txs = raw_data["transactions"]

        # Calculate profile features
        profile = {
            "account_age_days": self._calculate_account_age(txs),
            "transaction_frequency": len(txs) / state["params"]["lookback_days"],
            "protocol_diversity": self._calculate_protocol_diversity(txs),
            "counterparty_uniqueness": self._calculate_counterparty_uniqueness(txs),
            "value_patterns": self._analyze_value_patterns(txs),
            "gas_patterns": self._analyze_gas_patterns(txs),
            "time_patterns": self._analyze_time_patterns(txs)
        }

        features[address] = profile

    state["features"] = features
    logger.info("Address profiling complete")

    return state

def _calculate_protocol_diversity(self, transactions: List[Dict]) -> float:
    """Calculate protocol interaction diversity."""
    protocols = set()

    for tx in transactions:
        if tx.get("protocol"):
            protocols.add(tx["protocol"])

    # Normalize by common protocol count
    return min(len(protocols) / 5.0, 1.0)  # Max diversity at 5+ protocols
```

### 3. RiskScorer Node
```python
def _node_risk_score(self, state: GraphState) -> GraphState:
    """Apply risk assessment heuristics."""
    logger.info("Scoring risk factors")

    risk_data = {}

    for address, raw_data in state["raw"].items():
        txs = raw_data["transactions"]

        risk_score = 0
        evidence = []

        # Check for mixer interactions
        mixer_count = sum(1 for tx in txs if self._is_mixer_interaction(tx))
        if mixer_count > 0:
            risk_score += 20
            evidence.append(f"mixer_interactions:{mixer_count}")

        # Check for suspicious approvals
        suspicious_approvals = sum(1 for tx in txs if self._is_suspicious_approval(tx))
        if suspicious_approvals > 0:
            risk_score += 15
            evidence.append(f"suspicious_approvals:{suspicious_approvals}")

        # Check transaction failure rate
        failure_rate = sum(1 for tx in txs if tx["status"] == 0) / len(txs)
        if failure_rate > 0.1:  # >10% failure rate
            risk_score += 10
            evidence.append(f"high_failure_rate:{failure_rate:.2f}")

        risk_data[address] = {
            "risk_score": min(risk_score, 100),
            "evidence": evidence,
            "components": {
                "mixer_interactions": mixer_count,
                "suspicious_approvals": suspicious_approvals,
                "failure_rate": failure_rate
            }
        }

    state["risk"] = risk_data
    logger.info("Risk scoring complete")

    return state
```

### 4. Node Error Handling
```python
def _execute_node_with_error_handling(self, node_func, state: GraphState, node_name: str) -> GraphState:
    """Execute node with comprehensive error handling."""
    try:
        start_time = time.time()
        result = node_func(state)
        duration = time.time() - start_time

        logger.info("Node execution complete",
                   node=node_name,
                   duration=duration)

        return result

    except Exception as e:
        logger.error("Node execution failed",
                    node=node_name,
                    error=str(e))

        # For prototype, continue with empty results
        # In production, this would trigger retry logic
        return state
```

---

## Testing and Validation

### Integration Testing
```python
def test_complete_pipeline():
    """Test end-to-end pipeline execution."""
    from ..mock_data import MockDataProvider

    # Setup
    provider = MockDataProvider()
    graph = ReputationGraph(provider)

    # Test input
    test_input = ReputationInput(
        targets=["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"],
        lookback_days=30,
        max_txs=1000
    )

    # Execute
    reports = graph.analyze_reputation(test_input)

    # Validate
    assert len(reports) == 1
    assert "reputation_score" in reports[test_input.targets[0]]
    assert 0 <= reports[test_input.targets[0]]["reputation_score"] <= 100
```

### Performance Testing
```python
def test_pipeline_performance():
    """Test pipeline performance with mock data."""
    provider = MockDataProvider()
    graph = ReputationGraph(provider)

    # Test with multiple addresses
    test_input = ReputationInput(
        targets=[f"0x{i:040x}" for i in range(10)],  # 10 addresses
        lookback_days=90,
        max_txs=2000
    )

    start_time = time.time()
    reports = graph.analyze_reputation(test_input)
    duration = time.time() - start_time

    # Should complete under 30 seconds for 10 addresses with mock data
    assert duration < 30
    assert len(reports) == 10
```

---

## Success Criteria

**Step 4 is complete when:**

1. ✅ **Complete Pipeline** - All 6 nodes execute in correct order
2. ✅ **State Management** - Data flows correctly between nodes
3. ✅ **Mock Integration** - Uses mock data providers effectively
4. ✅ **Error Handling** - Graceful handling of node failures
5. ✅ **Performance** - Acceptable execution time for prototype
6. ✅ **Logging** - Comprehensive structured logging throughout
7. ✅ **Testing** - Integration tests validate end-to-end flow
8. ✅ **Type Safety** - All state transitions are properly typed

**Next Dependencies:**
- Provides orchestration framework for agent implementations (Phase 3)
- Establishes patterns for error handling and retry logic
- Creates foundation for performance optimization and monitoring