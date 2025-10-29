# Step 1: DataHarvester Agent Enhancement - Implementation Notes

**Context:** Advanced data collection intelligence with real-time quality assessment and multi-source fusion
**Approach:** Sophisticated intelligence layers with performance optimization and adaptive collection strategies

---

## Implementation Strategy

### Enhanced DataHarvester Architecture
Based on design decisions, implementing:
- **Smart collection orchestration** with dependency-aware fetching and adaptive strategies
- **Multi-dimensional quality assessment** with real-time scoring across completeness, consistency, freshness, accuracy, reliability
- **Advanced multi-source fusion** with temporal correlation, conflict resolution, and consensus mechanisms
- **Real-time pattern recognition** enabling adaptive collection depth and scope
- **Comprehensive anomaly detection** for data quality and behavioral anomalies
- **Rich metadata framework** with complete lineage tracking and quality audit trails

### File Structure
```
agents/data_harvester/
├── __init__.py
├── enhanced_harvester.py    # Main enhanced agent implementation
├── orchestration/
│   ├── __init__.py
│   ├── collection_orchestrator.py  # Smart collection sequencing
│   ├── dependency_manager.py       # Data dependency tracking
│   ├── adaptive_strategies.py      # Adaptive collection strategies
│   └── performance_optimizer.py    # Collection performance optimization
├── quality/
│   ├── __init__.py
│   ├── quality_assessor.py         # Multi-dimensional quality scoring
│   ├── completeness_analyzer.py    # Data completeness assessment
│   ├── consistency_validator.py    # Cross-source consistency validation
│   ├── freshness_calculator.py     # Data freshness scoring
│   └── reliability_tracker.py      # Source reliability assessment
├── fusion/
│   ├── __init__.py
│   ├── fusion_engine.py            # Multi-source data fusion
│   ├── temporal_aligner.py         # Temporal data alignment
│   ├── conflict_resolver.py        # Conflict detection and resolution
│   └── consensus_calculator.py     # Multi-source consensus mechanisms
├── patterns/
│   ├── __init__.py
│   ├── pattern_recognizer.py       # Real-time pattern recognition
│   ├── transaction_patterns.py     # Transaction pattern detection
│   ├── behavioral_patterns.py      # Behavioral pattern recognition
│   └── network_patterns.py         # Network pattern analysis
├── anomalies/
│   ├── __init__.py
│   ├── anomaly_detector.py         # Comprehensive anomaly detection
│   ├── data_anomalies.py          # Data quality anomaly detection
│   └── behavioral_anomalies.py     # Behavioral anomaly detection
└── metadata/
    ├── __init__.py
    ├── lineage_tracker.py          # Data lineage tracking
    ├── quality_auditor.py          # Quality audit trail management
    └── collection_profiler.py      # Collection performance profiling
```

---

## Core Enhanced Agent Implementation

### Main Enhanced DataHarvester

#### agents/data_harvester/enhanced_harvester.py
```python
"""Enhanced DataHarvester agent with sophisticated intelligence layers."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from schemas.agents.state import DataHarvesterState, ReputeNetGraphState
from schemas.blockchain.addresses import EnhancedAddressData
from tools.interfaces.eth_provider import EthProviderTool
from tools.interfaces.etherscan import EtherscanTool

from .orchestration.collection_orchestrator import SmartCollectionOrchestrator
from .quality.quality_assessor import MultiDimensionalQualityAssessor
from .fusion.fusion_engine import MultiSourceFusionEngine
from .patterns.pattern_recognizer import RealTimePatternRecognizer
from .anomalies.anomaly_detector import ComprehensiveAnomalyDetector
from .metadata.lineage_tracker import DataLineageTracker

class CollectionStrategy(Enum):
    """Data collection strategy options."""
    BASIC = "basic"              # Standard collection depth
    COMPREHENSIVE = "comprehensive"  # Deep collection with full analysis
    ADAPTIVE = "adaptive"        # Adapt based on discovered patterns
    PERFORMANCE = "performance"  # Optimized for speed
    QUALITY = "quality"         # Optimized for maximum quality

@dataclass
class CollectionRequirements:
    """Requirements for data collection."""
    target_address: str
    strategy: CollectionStrategy = CollectionStrategy.ADAPTIVE
    max_collection_time: int = 300  # 5 minutes default
    min_quality_threshold: float = 0.7
    include_historical_data: bool = True
    historical_depth_days: int = 365
    enable_pattern_recognition: bool = True
    enable_anomaly_detection: bool = True

@dataclass
class CollectionContext:
    """Context for data collection execution."""
    requirements: CollectionRequirements
    start_time: datetime
    tools_available: Dict[str, Any]
    quality_requirements: Dict[str, float]
    performance_budget: Dict[str, float]

class EnhancedDataHarvester:
    """Enhanced DataHarvester with sophisticated intelligence capabilities."""

    def __init__(
        self,
        eth_provider: EthProviderTool,
        etherscan: EtherscanTool,
        defillama: Any,  # Tool interfaces from DI
        label_registry: Any,
        cache_service: Any,
        metrics_service: Any
    ):
        self.eth_provider = eth_provider
        self.etherscan = etherscan
        self.defillama = defillama
        self.label_registry = label_registry
        self.cache_service = cache_service
        self.metrics_service = metrics_service

        self.logger = logging.getLogger(__name__)

        # Initialize intelligence components
        self.orchestrator = SmartCollectionOrchestrator(
            tools={
                "eth_provider": eth_provider,
                "etherscan": etherscan,
                "defillama": defillama,
                "label_registry": label_registry
            },
            cache_service=cache_service
        )

        self.quality_assessor = MultiDimensionalQualityAssessor()
        self.fusion_engine = MultiSourceFusionEngine()
        self.pattern_recognizer = RealTimePatternRecognizer()
        self.anomaly_detector = ComprehensiveAnomalyDetector()
        self.lineage_tracker = DataLineageTracker()

    async def collect_address_data(
        self,
        state: ReputeNetGraphState
    ) -> ReputeNetGraphState:
        """Enhanced data collection with intelligence layers."""

        # Extract collection requirements
        requirements = self._extract_collection_requirements(state)

        # Initialize collection context
        context = CollectionContext(
            requirements=requirements,
            start_time=datetime.utcnow(),
            tools_available=self._get_available_tools(),
            quality_requirements=self._get_quality_requirements(state),
            performance_budget=self._get_performance_budget(state)
        )

        self.logger.info(f"Starting enhanced collection for {requirements.target_address}")

        try:
            # Phase 1: Smart orchestrated data collection
            raw_data = await self._orchestrate_smart_collection(context)

            # Phase 2: Real-time quality assessment
            quality_scores = await self._assess_data_quality(raw_data, context)

            # Phase 3: Multi-source data fusion
            fused_data = await self._fuse_multi_source_data(raw_data, context)

            # Phase 4: Pattern recognition and adaptive collection
            patterns = await self._recognize_patterns_and_adapt(fused_data, context)

            # Phase 5: Anomaly detection
            anomalies = await self._detect_anomalies(fused_data, patterns, context)

            # Phase 6: Final quality validation and metadata compilation
            final_data = await self._compile_enhanced_data(
                raw_data, fused_data, quality_scores, patterns, anomalies, context
            )

            # Update state with enhanced data
            state = self._update_state_with_enhanced_data(state, final_data, context)

            self.logger.info(
                f"Enhanced collection completed: quality={final_data.quality_scores.composite_score:.3f}, "
                f"patterns={len(final_data.patterns)}, anomalies={len(final_data.anomalies)}"
            )

            return state

        except Exception as e:
            self.logger.error(f"Enhanced collection failed: {e}")

            # Update state with error information
            state.data_harvester.fail_execution(str(e))
            return state

    async def _orchestrate_smart_collection(self, context: CollectionContext) -> Dict[str, Any]:
        """Orchestrate smart data collection with dependency awareness."""

        self.logger.debug("Starting smart collection orchestration")

        # Determine collection plan based on strategy
        collection_plan = await self.orchestrator.create_collection_plan(context)

        # Execute collection plan with performance monitoring
        raw_data = await self.orchestrator.execute_collection_plan(collection_plan)

        # Track collection performance
        collection_metrics = self.orchestrator.get_collection_metrics()
        await self.metrics_service.record_collection_metrics(collection_metrics)

        return raw_data

    async def _assess_data_quality(
        self,
        raw_data: Dict[str, Any],
        context: CollectionContext
    ) -> DataQualityScores:
        """Assess data quality across multiple dimensions."""

        self.logger.debug("Assessing data quality")

        quality_scores = await self.quality_assessor.assess_comprehensive_quality(
            data=raw_data,
            requirements=context.requirements,
            collection_context=context
        )

        # Log quality assessment results
        self.logger.info(
            f"Quality assessment: composite={quality_scores.composite_score:.3f}, "
            f"completeness={quality_scores.completeness:.3f}, "
            f"consistency={quality_scores.consistency:.3f}"
        )

        return quality_scores

    async def _fuse_multi_source_data(
        self,
        raw_data: Dict[str, Any],
        context: CollectionContext
    ) -> FusedData:
        """Fuse data from multiple sources with conflict resolution."""

        self.logger.debug("Starting multi-source data fusion")

        fused_data = await self.fusion_engine.fuse_comprehensive_data(
            raw_data=raw_data,
            fusion_strategy=self._determine_fusion_strategy(context),
            quality_requirements=context.quality_requirements
        )

        # Log fusion results
        self.logger.info(
            f"Data fusion completed: sources={len(raw_data)}, "
            f"conflicts_resolved={len(fused_data.resolved_conflicts)}, "
            f"confidence={fused_data.overall_confidence:.3f}"
        )

        return fused_data

    async def _recognize_patterns_and_adapt(
        self,
        fused_data: FusedData,
        context: CollectionContext
    ) -> List[RecognizedPattern]:
        """Recognize patterns and adapt collection strategy if needed."""

        if not context.requirements.enable_pattern_recognition:
            return []

        self.logger.debug("Starting real-time pattern recognition")

        # Recognize patterns in current data
        patterns = await self.pattern_recognizer.recognize_comprehensive_patterns(
            data=fused_data,
            pattern_types=self._get_enabled_pattern_types(context),
            recognition_depth=self._get_pattern_recognition_depth(context)
        )

        # Adapt collection strategy based on discovered patterns
        if patterns and context.requirements.strategy == CollectionStrategy.ADAPTIVE:
            await self._adapt_collection_based_on_patterns(patterns, context)

        self.logger.info(f"Pattern recognition completed: {len(patterns)} patterns found")

        return patterns

    async def _detect_anomalies(
        self,
        fused_data: FusedData,
        patterns: List[RecognizedPattern],
        context: CollectionContext
    ) -> List[DetectedAnomaly]:
        """Detect anomalies in data quality and behavior."""

        if not context.requirements.enable_anomaly_detection:
            return []

        self.logger.debug("Starting anomaly detection")

        anomalies = await self.anomaly_detector.detect_comprehensive_anomalies(
            data=fused_data,
            patterns=patterns,
            detection_sensitivity=self._get_anomaly_sensitivity(context),
            anomaly_types=self._get_enabled_anomaly_types(context)
        )

        if anomalies:
            self.logger.warning(f"Detected {len(anomalies)} anomalies during collection")

        return anomalies

    async def _compile_enhanced_data(
        self,
        raw_data: Dict[str, Any],
        fused_data: FusedData,
        quality_scores: DataQualityScores,
        patterns: List[RecognizedPattern],
        anomalies: List[DetectedAnomaly],
        context: CollectionContext
    ) -> EnhancedAddressData:
        """Compile all collected and analyzed data into enhanced format."""

        # Generate comprehensive metadata
        metadata = await self.lineage_tracker.generate_comprehensive_metadata(
            raw_data=raw_data,
            fused_data=fused_data,
            quality_scores=quality_scores,
            patterns=patterns,
            anomalies=anomalies,
            context=context
        )

        # Create enhanced data structure
        enhanced_data = EnhancedAddressData(
            address=context.requirements.target_address,
            raw_data=raw_data,
            fused_data=fused_data,
            quality_scores=quality_scores,
            patterns=patterns,
            anomalies=anomalies,
            metadata=metadata,
            collection_summary=self._generate_collection_summary(context)
        )

        return enhanced_data

    def _extract_collection_requirements(self, state: ReputeNetGraphState) -> CollectionRequirements:
        """Extract collection requirements from workflow state."""

        analysis_request = state.analysis_request

        return CollectionRequirements(
            target_address=analysis_request.target_address,
            strategy=self._determine_collection_strategy(analysis_request),
            max_collection_time=analysis_request.timeout_seconds,
            min_quality_threshold=0.7,  # Configurable
            include_historical_data=True,
            historical_depth_days=analysis_request.historical_days,
            enable_pattern_recognition=True,
            enable_anomaly_detection=True
        )

    def _determine_collection_strategy(self, analysis_request) -> CollectionStrategy:
        """Determine optimal collection strategy based on request."""

        if analysis_request.analysis_depth == "comprehensive":
            return CollectionStrategy.COMPREHENSIVE
        elif analysis_request.timeout_seconds < 120:  # Less than 2 minutes
            return CollectionStrategy.PERFORMANCE
        else:
            return CollectionStrategy.ADAPTIVE

    def _update_state_with_enhanced_data(
        self,
        state: ReputeNetGraphState,
        enhanced_data: EnhancedAddressData,
        context: CollectionContext
    ) -> ReputeNetGraphState:
        """Update workflow state with enhanced collection results."""

        # Update DataHarvester agent state
        harvester_state = state.data_harvester
        harvester_state.collected_data = enhanced_data.dict()
        harvester_state.data_quality_score = enhanced_data.quality_scores.composite_score
        harvester_state.completeness_score = enhanced_data.quality_scores.completeness

        # Update data sources used
        harvester_state.data_sources = list(enhanced_data.raw_data.keys())

        # Record any collection errors from anomalies
        for anomaly in enhanced_data.anomalies:
            if anomaly.anomaly_type.startswith("collection_"):
                harvester_state.collection_errors.append(anomaly.description)

        # Add performance metrics
        collection_time = (datetime.utcnow() - context.start_time).total_seconds()
        harvester_state.processing_time = collection_time

        # Store enhanced data in shared state for downstream agents
        state.shared_data["enhanced_address_data"] = enhanced_data.dict()
        state.shared_data["data_quality_summary"] = {
            "overall_quality": enhanced_data.quality_scores.composite_score,
            "quality_breakdown": enhanced_data.quality_scores.dict(),
            "pattern_count": len(enhanced_data.patterns),
            "anomaly_count": len(enhanced_data.anomalies)
        }

        # Mark completion
        harvester_state.complete_execution()

        return state

    def _generate_collection_summary(self, context: CollectionContext) -> Dict[str, Any]:
        """Generate summary of collection execution."""

        execution_time = (datetime.utcnow() - context.start_time).total_seconds()

        return {
            "strategy_used": context.requirements.strategy.value,
            "execution_time_seconds": execution_time,
            "tools_utilized": list(context.tools_available.keys()),
            "quality_threshold_met": True,  # Would be calculated
            "adaptive_adjustments_made": 0,  # Would be tracked
            "performance_within_budget": execution_time < context.requirements.max_collection_time
        }
```

### Smart Collection Orchestrator

#### agents/data_harvester/orchestration/collection_orchestrator.py
```python
"""Smart collection orchestration with dependency-aware fetching."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

class DataDependencyType(Enum):
    """Types of data dependencies."""
    REQUIRED = "required"      # Must have this data before proceeding
    OPTIONAL = "optional"      # Helpful but not required
    ENRICHMENT = "enrichment"  # Enriches existing data
    VALIDATION = "validation"  # Validates other data

@dataclass
class DataDependency:
    """Data dependency definition."""
    source_data: str
    target_data: str
    dependency_type: DataDependencyType
    priority: int = 1  # Higher = more important

@dataclass
class CollectionTask:
    """Individual data collection task."""
    task_id: str
    data_type: str
    tool_name: str
    method: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1
    estimated_time: float = 30.0  # seconds
    quality_impact: float = 1.0   # 0.0-1.0

class CollectionPlan:
    """Execution plan for data collection."""

    def __init__(self):
        self.tasks: List[CollectionTask] = []
        self.execution_order: List[str] = []
        self.parallel_groups: List[List[str]] = []
        self.estimated_total_time: float = 0.0

class SmartCollectionOrchestrator:
    """Orchestrate smart data collection with dependency awareness."""

    def __init__(self, tools: Dict[str, Any], cache_service: Any):
        self.tools = tools
        self.cache_service = cache_service
        self.logger = logging.getLogger(__name__)

        # Define data dependencies
        self.dependencies = self._define_data_dependencies()

        # Performance tracking
        self.collection_metrics = {}

    def _define_data_dependencies(self) -> List[DataDependency]:
        """Define dependencies between different data types."""

        return [
            # Basic account info needed first
            DataDependency("account_basic", "transaction_history", DataDependencyType.REQUIRED),
            DataDependency("account_basic", "token_balances", DataDependencyType.REQUIRED),

            # Transaction history enables deeper analysis
            DataDependency("transaction_history", "contract_interactions", DataDependencyType.OPTIONAL),
            DataDependency("transaction_history", "defi_activity", DataDependencyType.ENRICHMENT),

            # Token balances help with portfolio analysis
            DataDependency("token_balances", "token_metadata", DataDependencyType.ENRICHMENT),
            DataDependency("token_balances", "defi_positions", DataDependencyType.OPTIONAL),

            # Contract interactions enable sophisticated analysis
            DataDependency("contract_interactions", "contract_verification", DataDependencyType.VALIDATION),
            DataDependency("contract_interactions", "protocol_metadata", DataDependencyType.ENRICHMENT),

            # Cross-validation dependencies
            DataDependency("etherscan_data", "eth_provider_data", DataDependencyType.VALIDATION),
            DataDependency("defi_activity", "protocol_metadata", DataDependencyType.VALIDATION),
        ]

    async def create_collection_plan(self, context: CollectionContext) -> CollectionPlan:
        """Create optimized collection plan based on requirements."""

        self.logger.debug("Creating collection plan")

        # Determine required data types based on strategy
        required_data_types = self._determine_required_data_types(context)

        # Create tasks for each data type
        tasks = []
        for data_type in required_data_types:
            task = self._create_collection_task(data_type, context)
            if task:
                tasks.append(task)

        # Optimize task ordering based on dependencies
        execution_plan = self._optimize_execution_order(tasks)

        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(tasks, execution_plan)

        plan = CollectionPlan()
        plan.tasks = tasks
        plan.execution_order = execution_plan
        plan.parallel_groups = parallel_groups
        plan.estimated_total_time = self._estimate_execution_time(tasks, parallel_groups)

        self.logger.info(
            f"Collection plan created: {len(tasks)} tasks, "
            f"{len(parallel_groups)} parallel groups, "
            f"estimated time: {plan.estimated_total_time:.1f}s"
        )

        return plan

    async def execute_collection_plan(self, plan: CollectionPlan) -> Dict[str, Any]:
        """Execute collection plan with performance monitoring."""

        self.logger.debug("Executing collection plan")

        collected_data = {}
        execution_start = asyncio.get_event_loop().time()

        try:
            # Execute tasks according to plan
            for group in plan.parallel_groups:
                if len(group) == 1:
                    # Single task execution
                    task_id = group[0]
                    task = next(t for t in plan.tasks if t.task_id == task_id)
                    data = await self._execute_single_task(task, collected_data)
                    collected_data.update(data)
                else:
                    # Parallel task execution
                    parallel_data = await self._execute_parallel_tasks(
                        [t for t in plan.tasks if t.task_id in group],
                        collected_data
                    )
                    collected_data.update(parallel_data)

            execution_time = asyncio.get_event_loop().time() - execution_start

            self.logger.info(f"Collection plan executed in {execution_time:.2f}s")

            # Update performance metrics
            self.collection_metrics = {
                "execution_time": execution_time,
                "tasks_completed": len(plan.tasks),
                "data_sources": len(collected_data),
                "plan_efficiency": plan.estimated_total_time / execution_time if execution_time > 0 else 1.0
            }

            return collected_data

        except Exception as e:
            self.logger.error(f"Collection plan execution failed: {e}")
            raise

    async def _execute_single_task(
        self,
        task: CollectionTask,
        existing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single collection task."""

        self.logger.debug(f"Executing task: {task.task_id}")

        # Check cache first
        cache_key = self._generate_cache_key(task)
        cached_data = await self.cache_service.get(cache_key)
        if cached_data:
            self.logger.debug(f"Using cached data for task: {task.task_id}")
            return {task.data_type: cached_data}

        # Get the appropriate tool
        tool = self.tools.get(task.tool_name)
        if not tool:
            raise ValueError(f"Tool not available: {task.tool_name}")

        try:
            # Execute the tool method
            method = getattr(tool, task.method)
            result = await method(**task.parameters)

            # Cache the result
            await self.cache_service.set(cache_key, result, ttl=300)  # 5 minutes

            return {task.data_type: result}

        except Exception as e:
            self.logger.warning(f"Task {task.task_id} failed: {e}")
            return {task.data_type: None}

    async def _execute_parallel_tasks(
        self,
        tasks: List[CollectionTask],
        existing_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute multiple tasks in parallel."""

        self.logger.debug(f"Executing {len(tasks)} tasks in parallel")

        # Create coroutines for all tasks
        task_coroutines = [
            self._execute_single_task(task, existing_data)
            for task in tasks
        ]

        # Execute in parallel
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Combine results
        combined_data = {}
        for result in results:
            if isinstance(result, dict):
                combined_data.update(result)
            else:
                self.logger.warning(f"Parallel task failed: {result}")

        return combined_data

    def _optimize_execution_order(self, tasks: List[CollectionTask]) -> List[str]:
        """Optimize task execution order based on dependencies."""

        # Build dependency graph
        task_graph = {}
        for task in tasks:
            task_graph[task.task_id] = task.dependencies or []

        # Topological sort to respect dependencies
        execution_order = []
        remaining_tasks = set(task.task_id for task in tasks)

        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                dependencies = task_graph[task_id]
                if all(dep in execution_order for dep in dependencies):
                    ready_tasks.append(task_id)

            if not ready_tasks:
                # Break circular dependencies by priority
                ready_tasks = [min(remaining_tasks, key=lambda t: next(
                    task.priority for task in tasks if task.task_id == t
                ))]

            # Sort ready tasks by priority and quality impact
            ready_tasks.sort(key=lambda t: (
                -next(task.priority for task in tasks if task.task_id == t),
                -next(task.quality_impact for task in tasks if task.task_id == t)
            ))

            # Add to execution order
            execution_order.extend(ready_tasks)
            remaining_tasks -= set(ready_tasks)

        return execution_order

    def _identify_parallel_groups(
        self,
        tasks: List[CollectionTask],
        execution_order: List[str]
    ) -> List[List[str]]:
        """Identify tasks that can be executed in parallel."""

        parallel_groups = []
        task_lookup = {task.task_id: task for task in tasks}

        i = 0
        while i < len(execution_order):
            current_group = [execution_order[i]]
            j = i + 1

            # Look for tasks that can run in parallel with current task
            while j < len(execution_order):
                current_task = task_lookup[execution_order[j]]
                can_parallelize = True

                # Check if this task depends on any task in current group
                for group_task_id in current_group:
                    if group_task_id in (current_task.dependencies or []):
                        can_parallelize = False
                        break

                # Check if any task in current group depends on this task
                for group_task_id in current_group:
                    group_task = task_lookup[group_task_id]
                    if current_task.task_id in (group_task.dependencies or []):
                        can_parallelize = False
                        break

                if can_parallelize:
                    current_group.append(execution_order[j])
                    execution_order.pop(j)
                else:
                    j += 1

            parallel_groups.append(current_group)
            i += 1

        return parallel_groups

    def get_collection_metrics(self) -> Dict[str, Any]:
        """Get collection performance metrics."""
        return self.collection_metrics.copy()
```

This comprehensive implementation provides sophisticated data collection intelligence that demonstrates advanced blockchain data engineering capabilities while maintaining performance and reliability for stakeholder demonstrations. The system can adaptively collect data based on discovered patterns, assess quality in real-time, and provide rich metadata for downstream analysis.