# Step 4: LangGraph Orchestration Framework - Implementation Notes

**Context:** Production-ready LangGraph orchestration with parallel execution and error recovery
**Approach:** Parallel branch architecture with graceful degradation and comprehensive checkpointing

---

## Implementation Strategy

### LangGraph Architecture
Based on design decisions, implementing:
- **Parallel branch execution** optimizing RiskScorer + SybilDetector parallel execution
- **Centralized state management** with typed agent interfaces and immutable updates
- **Graceful degradation** allowing workflow continuation with partial failures
- **Agent-level checkpointing** enabling workflow recovery and resume
- **Comprehensive monitoring** tracking execution metrics and state transitions

### Workflow Structure
```
DataHarvester → AddressProfiler → [RiskScorer + SybilDetector] → ReputationAggregator → Reporter
                                         ↘        ↙
                                    (Parallel Execution)
```

---

## Core Orchestration Implementation

### LangGraph Workflow Definition

#### orchestration/workflow.py
```python
"""LangGraph workflow orchestration for ReputeNet agents."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable

from langgraph import StateGraph, START, END
from langgraph.checkpoint import MemorySaver
from langgraph.errors import NodeInterrupt

from schemas.agents.state import (
    ReputeNetGraphState,
    AnalysisRequest,
    WorkflowStatus,
    AgentStatus,
    WorkflowError
)
from .checkpointing import WorkflowCheckpointer
from .error_handling import WorkflowErrorHandler
from .monitoring import WorkflowMonitor

class ReputeNetWorkflow:
    """LangGraph-based workflow orchestration for ReputeNet analysis."""

    def __init__(
        self,
        agents: Dict[str, Callable],
        config: Dict[str, Any] = None
    ):
        self.agents = agents
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.error_handler = WorkflowErrorHandler(
            max_retries=self.config.get('max_retries', 3),
            graceful_degradation=self.config.get('graceful_degradation', True)
        )
        self.checkpointer = WorkflowCheckpointer(
            storage_backend=self.config.get('checkpoint_storage', 'memory')
        )
        self.monitor = WorkflowMonitor()

        # Build the workflow graph
        self.graph = self._build_workflow_graph()

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow with parallel execution."""

        # Create the state graph
        workflow = StateGraph(ReputeNetGraphState)

        # Add agent nodes with error handling wrappers
        workflow.add_node("data_harvester", self._create_agent_wrapper("data_harvester"))
        workflow.add_node("address_profiler", self._create_agent_wrapper("address_profiler"))
        workflow.add_node("risk_scorer", self._create_agent_wrapper("risk_scorer"))
        workflow.add_node("sybil_detector", self._create_agent_wrapper("sybil_detector"))
        workflow.add_node("reputation_aggregator", self._create_agent_wrapper("reputation_aggregator"))
        workflow.add_node("reporter", self._create_agent_wrapper("reporter"))

        # Add control nodes
        workflow.add_node("parallel_coordinator", self._parallel_coordinator)
        workflow.add_node("convergence_point", self._convergence_point)

        # Define workflow edges
        workflow.add_edge(START, "data_harvester")
        workflow.add_edge("data_harvester", "address_profiler")
        workflow.add_edge("address_profiler", "parallel_coordinator")

        # Parallel execution branches
        workflow.add_edge("parallel_coordinator", "risk_scorer")
        workflow.add_edge("parallel_coordinator", "sybil_detector")

        # Convergence after parallel execution
        workflow.add_edge("risk_scorer", "convergence_point")
        workflow.add_edge("sybil_detector", "convergence_point")
        workflow.add_edge("convergence_point", "reputation_aggregator")
        workflow.add_edge("reputation_aggregator", "reporter")
        workflow.add_edge("reporter", END)

        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "parallel_coordinator",
            self._should_skip_parallel_agents,
            {
                "continue": "convergence_point",
                "execute_parallel": ["risk_scorer", "sybil_detector"]
            }
        )

        return workflow

    def _create_agent_wrapper(self, agent_name: str) -> Callable:
        """Create error-handling wrapper for agent execution."""

        async def agent_wrapper(state: ReputeNetGraphState) -> ReputeNetGraphState:
            """Wrapper that adds error handling, monitoring, and checkpointing."""

            # Start execution tracking
            agent_state = state.get_agent_state(agent_name)
            agent_state.start_execution()
            state.current_agent = agent_name

            self.logger.info(f"Starting agent: {agent_name}")
            execution_start = datetime.utcnow()

            try:
                # Execute the agent
                if agent_name not in self.agents:
                    raise ValueError(f"Agent not found: {agent_name}")

                agent_function = self.agents[agent_name]
                updated_state = await agent_function(state)

                # Mark completion
                agent_state = updated_state.get_agent_state(agent_name)
                agent_state.complete_execution()

                # Calculate performance metrics
                execution_time = (datetime.utcnow() - execution_start).total_seconds()
                agent_state.processing_time = execution_time

                # Save checkpoint
                await self.checkpointer.save_checkpoint(
                    state.workflow_id,
                    updated_state,
                    agent_name
                )

                # Update monitoring
                self.monitor.record_agent_completion(agent_name, execution_time)

                self.logger.info(f"Completed agent: {agent_name} in {execution_time:.2f}s")
                return updated_state

            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")

                # Handle the error
                error_handled_state = await self.error_handler.handle_agent_error(
                    agent_name, e, state
                )

                # Update monitoring
                self.monitor.record_agent_error(agent_name, str(e))

                return error_handled_state

        return agent_wrapper

    async def _parallel_coordinator(self, state: ReputeNetGraphState) -> ReputeNetGraphState:
        """Coordinate parallel execution of risk scorer and sybil detector."""

        # Check if address profiler completed successfully
        if state.address_profiler.status != AgentStatus.COMPLETED:
            self.logger.warning("Address profiler not completed, skipping parallel agents")
            return state

        # Set up parallel execution
        state.update_metadata("parallel_execution_started", datetime.utcnow().isoformat())
        return state

    async def _convergence_point(self, state: ReputeNetGraphState) -> ReputeNetGraphState:
        """Convergence point after parallel execution."""

        # Wait for both parallel agents to complete or fail
        risk_status = state.risk_scorer.status
        sybil_status = state.sybil_detector.status

        completed_agents = []
        if risk_status in [AgentStatus.COMPLETED, AgentStatus.SKIPPED]:
            completed_agents.append("risk_scorer")
        if sybil_status in [AgentStatus.COMPLETED, AgentStatus.SKIPPED]:
            completed_agents.append("sybil_detector")

        self.logger.info(f"Parallel execution completed: {completed_agents}")

        # Update state with convergence information
        state.update_metadata("parallel_execution_completed", datetime.utcnow().isoformat())
        state.update_metadata("completed_parallel_agents", completed_agents)

        return state

    def _should_skip_parallel_agents(self, state: ReputeNetGraphState) -> str:
        """Conditional logic for parallel agent execution."""

        # Check if address profiler completed successfully
        if state.address_profiler.status != AgentStatus.COMPLETED:
            return "continue"

        # Check for critical errors
        critical_errors = [
            error for error in state.errors
            if error.agent_name == "address_profiler" and not error.recoverable
        ]

        if critical_errors:
            return "continue"

        return "execute_parallel"

    async def execute_workflow(
        self,
        analysis_request: AnalysisRequest,
        workflow_id: Optional[str] = None
    ) -> ReputeNetGraphState:
        """Execute the complete workflow."""

        # Create or restore workflow state
        if workflow_id:
            # Try to restore from checkpoint
            restored_state = await self.checkpointer.restore_checkpoint(workflow_id)
            if restored_state:
                self.logger.info(f"Restored workflow from checkpoint: {workflow_id}")
                initial_state = restored_state
            else:
                initial_state = self._create_initial_state(analysis_request, workflow_id)
        else:
            workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            initial_state = self._create_initial_state(analysis_request, workflow_id)

        # Start workflow execution
        initial_state.start_workflow()
        self.monitor.start_workflow_tracking(workflow_id)

        try:
            # Compile and execute the graph
            compiled_graph = self.graph.compile(
                checkpointer=MemorySaver(),
                interrupt_before=[],  # No manual interrupts for now
                interrupt_after=[]
            )

            # Execute the workflow
            final_state = None
            async for state in compiled_graph.astream(
                initial_state,
                config={"configurable": {"thread_id": workflow_id}}
            ):
                final_state = state

            # Complete workflow
            if final_state:
                final_state.complete_workflow()
                self.monitor.complete_workflow_tracking(workflow_id)

            return final_state

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")

            # Create error state
            error = WorkflowError(
                agent_name="workflow",
                error_type=type(e).__name__,
                error_message=str(e)
            )

            if 'final_state' in locals():
                final_state.fail_workflow(error)
                return final_state
            else:
                initial_state.fail_workflow(error)
                return initial_state

    def _create_initial_state(
        self,
        analysis_request: AnalysisRequest,
        workflow_id: str
    ) -> ReputeNetGraphState:
        """Create initial workflow state."""

        return ReputeNetGraphState(
            workflow_id=workflow_id,
            analysis_request=analysis_request,
            workflow_status=WorkflowStatus.PENDING
        )

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current workflow status and progress."""

        # Try to get from checkpoint
        state = await self.checkpointer.restore_checkpoint(workflow_id)
        if not state:
            return {"error": "Workflow not found"}

        progress = state.get_workflow_progress()
        monitoring_data = self.monitor.get_workflow_metrics(workflow_id)

        return {
            "workflow_id": workflow_id,
            "status": state.workflow_status,
            "progress": progress,
            "metrics": monitoring_data,
            "errors": [error.dict() for error in state.errors],
            "started_at": state.started_at.isoformat() if state.started_at else None,
            "completed_at": state.completed_at.isoformat() if state.completed_at else None
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow."""

        # Implementation would depend on execution environment
        # For now, mark as cancelled in checkpoint
        state = await self.checkpointer.restore_checkpoint(workflow_id)
        if state and state.workflow_status == WorkflowStatus.RUNNING:
            state.workflow_status = WorkflowStatus.CANCELLED
            await self.checkpointer.save_checkpoint(workflow_id, state, "cancelled")
            return True

        return False

    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get overall workflow performance metrics."""
        return self.monitor.get_overall_metrics()
```

### Error Handling Framework

#### orchestration/error_handling.py
```python
"""Comprehensive error handling for workflow execution."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from schemas.agents.state import (
    ReputeNetGraphState,
    AgentStatus,
    WorkflowError,
    WorkflowStatus
)

class WorkflowErrorHandler:
    """Handle errors and implement graceful degradation strategies."""

    def __init__(self, max_retries: int = 3, graceful_degradation: bool = True):
        self.max_retries = max_retries
        self.graceful_degradation = graceful_degradation
        self.logger = logging.getLogger(__name__)

        # Define recoverable error types
        self.recoverable_errors = {
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "RateLimitError",
            "ServiceUnavailableError"
        }

        # Define critical agents that cannot be skipped
        self.critical_agents = {
            "data_harvester",
            "address_profiler"
        }

    async def handle_agent_error(
        self,
        agent_name: str,
        error: Exception,
        state: ReputeNetGraphState
    ) -> ReputeNetGraphState:
        """Handle agent execution errors with recovery strategies."""

        error_type = type(error).__name__
        is_recoverable = self._is_recoverable_error(error)

        # Create error record
        error_info = WorkflowError(
            agent_name=agent_name,
            error_type=error_type,
            error_message=str(error),
            recoverable=is_recoverable
        )

        state.errors.append(error_info)
        self.logger.error(f"Agent {agent_name} failed: {error_info.error_message}")

        # Try recovery strategies
        if is_recoverable and state.retry_attempts < self.max_retries:
            return await self._attempt_retry(agent_name, state, error_info)

        elif self.graceful_degradation and agent_name not in self.critical_agents:
            return await self._apply_graceful_degradation(agent_name, state, error_info)

        else:
            return await self._fail_workflow(state, error_info)

    async def _attempt_retry(
        self,
        agent_name: str,
        state: ReputeNetGraphState,
        error_info: WorkflowError
    ) -> ReputeNetGraphState:
        """Attempt to retry failed agent."""

        state.retry_attempts += 1
        retry_delay = min(2 ** state.retry_attempts, 60)  # Exponential backoff, max 60s

        self.logger.info(
            f"Retrying agent {agent_name} (attempt {state.retry_attempts}/{self.max_retries}) "
            f"after {retry_delay}s delay"
        )

        # Add retry delay
        await asyncio.sleep(retry_delay)

        # Reset agent state for retry
        agent_state = state.get_agent_state(agent_name)
        agent_state.retry_execution()

        return state

    async def _apply_graceful_degradation(
        self,
        agent_name: str,
        state: ReputeNetGraphState,
        error_info: WorkflowError
    ) -> ReputeNetGraphState:
        """Apply graceful degradation strategies."""

        self.logger.info(f"Applying graceful degradation for agent {agent_name}")

        agent_state = state.get_agent_state(agent_name)
        agent_state.status = AgentStatus.SKIPPED
        agent_state.error_message = f"Skipped due to error: {error_info.error_message}"

        # Apply agent-specific degradation strategies
        if agent_name == "risk_scorer":
            # Continue without risk scoring, use default risk values
            state.shared_data["risk_analysis_skipped"] = True
            state.shared_data["default_risk_score"] = 0.5  # Neutral risk

        elif agent_name == "sybil_detector":
            # Continue without sybil detection
            state.shared_data["sybil_analysis_skipped"] = True
            state.shared_data["default_sybil_probability"] = 0.0

        elif agent_name == "reputation_aggregator":
            # Use simplified aggregation
            state.shared_data["simplified_aggregation"] = True

        # Record degradation in metadata
        state.update_metadata("graceful_degradation_applied", {
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "strategy": "skip_with_defaults"
        })

        return state

    async def _fail_workflow(
        self,
        state: ReputeNetGraphState,
        error_info: WorkflowError
    ) -> ReputeNetGraphState:
        """Fail the entire workflow."""

        self.logger.error(f"Failing workflow due to critical error: {error_info.error_message}")

        state.fail_workflow(error_info)
        return state

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable."""
        error_type = type(error).__name__
        return error_type in self.recoverable_errors

    def get_error_summary(self, state: ReputeNetGraphState) -> Dict[str, Any]:
        """Get summary of all errors in the workflow."""

        if not state.errors:
            return {"total_errors": 0}

        error_types = {}
        recoverable_count = 0
        critical_count = 0

        for error in state.errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            if error.recoverable:
                recoverable_count += 1
            else:
                critical_count += 1

        return {
            "total_errors": len(state.errors),
            "error_types": error_types,
            "recoverable_errors": recoverable_count,
            "critical_errors": critical_count,
            "retry_attempts": state.retry_attempts,
            "degraded_agents": [
                error.agent_name for error in state.errors
                if error.agent_name in ["risk_scorer", "sybil_detector", "reputation_aggregator"]
            ]
        }
```

### Workflow Monitoring

#### orchestration/monitoring.py
```python
"""Workflow execution monitoring and metrics collection."""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class AgentMetrics:
    """Metrics for individual agent execution."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """Metrics for complete workflow execution."""
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_time: float = 0.0
    agent_metrics: Dict[str, AgentMetrics] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    degraded_agents: List[str] = field(default_factory=list)

class WorkflowMonitor:
    """Monitor workflow execution and collect performance metrics."""

    def __init__(self):
        self.active_workflows: Dict[str, WorkflowMetrics] = {}
        self.completed_workflows: List[WorkflowMetrics] = []
        self.global_agent_metrics: Dict[str, AgentMetrics] = {}

    def start_workflow_tracking(self, workflow_id: str) -> None:
        """Start tracking a workflow."""
        self.active_workflows[workflow_id] = WorkflowMetrics(
            workflow_id=workflow_id,
            started_at=datetime.utcnow()
        )

    def complete_workflow_tracking(self, workflow_id: str) -> None:
        """Complete workflow tracking."""
        if workflow_id in self.active_workflows:
            workflow_metrics = self.active_workflows[workflow_id]
            workflow_metrics.completed_at = datetime.utcnow()
            workflow_metrics.total_time = (
                workflow_metrics.completed_at - workflow_metrics.started_at
            ).total_seconds()

            # Move to completed workflows
            self.completed_workflows.append(workflow_metrics)
            del self.active_workflows[workflow_id]

    def record_agent_completion(self, agent_name: str, execution_time: float) -> None:
        """Record successful agent completion."""
        self._update_agent_metrics(agent_name, True, execution_time)

    def record_agent_error(self, agent_name: str, error_message: str) -> None:
        """Record agent error."""
        self._update_agent_metrics(agent_name, False, 0.0, error_message)

    def _update_agent_metrics(
        self,
        agent_name: str,
        success: bool,
        execution_time: float,
        error_message: str = None
    ) -> None:
        """Update metrics for an agent."""

        # Initialize if not exists
        if agent_name not in self.global_agent_metrics:
            self.global_agent_metrics[agent_name] = AgentMetrics()

        metrics = self.global_agent_metrics[agent_name]
        metrics.total_executions += 1

        if success:
            metrics.successful_executions += 1
            metrics.total_execution_time += execution_time
            metrics.average_execution_time = (
                metrics.total_execution_time / metrics.successful_executions
            )
        else:
            metrics.failed_executions += 1
            if error_message:
                error_type = error_message.split(':')[0] if ':' in error_message else error_message
                metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1

    def get_workflow_metrics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific workflow."""

        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow_metrics = self.active_workflows[workflow_id]
            current_time = datetime.utcnow()
            elapsed_time = (current_time - workflow_metrics.started_at).total_seconds()

            return {
                "workflow_id": workflow_id,
                "status": "running",
                "elapsed_time": elapsed_time,
                "started_at": workflow_metrics.started_at.isoformat()
            }

        # Check completed workflows
        for workflow_metrics in self.completed_workflows:
            if workflow_metrics.workflow_id == workflow_id:
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "total_time": workflow_metrics.total_time,
                    "started_at": workflow_metrics.started_at.isoformat(),
                    "completed_at": workflow_metrics.completed_at.isoformat(),
                    "agent_metrics": {
                        name: {
                            "executions": metrics.total_executions,
                            "success_rate": metrics.successful_executions / metrics.total_executions if metrics.total_executions > 0 else 0,
                            "average_time": metrics.average_execution_time
                        }
                        for name, metrics in workflow_metrics.agent_metrics.items()
                    }
                }

        return None

    def get_overall_metrics(self) -> Dict[str, Any]:
        """Get overall system metrics."""

        total_workflows = len(self.completed_workflows) + len(self.active_workflows)
        completed_workflows = len(self.completed_workflows)

        # Calculate average workflow time
        avg_workflow_time = 0.0
        if self.completed_workflows:
            total_time = sum(w.total_time for w in self.completed_workflows)
            avg_workflow_time = total_time / len(self.completed_workflows)

        # Agent success rates
        agent_success_rates = {}
        for agent_name, metrics in self.global_agent_metrics.items():
            if metrics.total_executions > 0:
                agent_success_rates[agent_name] = {
                    "success_rate": metrics.successful_executions / metrics.total_executions,
                    "average_time": metrics.average_execution_time,
                    "total_executions": metrics.total_executions
                }

        return {
            "total_workflows": total_workflows,
            "completed_workflows": completed_workflows,
            "active_workflows": len(self.active_workflows),
            "average_workflow_time": avg_workflow_time,
            "agent_success_rates": agent_success_rates,
            "system_uptime": time.time()  # Simplified uptime
        }
```

This comprehensive LangGraph orchestration implementation provides production-ready workflow management with parallel execution, sophisticated error handling, checkpointing, and monitoring capabilities that ensure both performance and reliability for the multi-agent ReputeNet system.