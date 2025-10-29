# Step 5: ReputationAggregator Agent (Sophisticated Scoring) - Implementation Notes

**Context:** Advanced multi-dimensional reputation aggregation with dynamic weighting and comprehensive uncertainty quantification
**Approach:** Production-ready aggregation framework with business context adaptation and temporal analysis

---

## Implementation Strategy

### ReputationAggregator Architecture
Based on design decisions, implementing:
- **Advanced dynamic weighting** with context-aware weight calculation, quality integration, and uncertainty-based adjustments
- **Comprehensive multi-dimensional scoring** with 5 specialized dimensional scores for different business use cases
- **Monte Carlo uncertainty quantification** with confidence intervals, correlation analysis, and temporal uncertainty evolution
- **Temporal analysis engine** with change detection, trend analysis, pattern recognition, and predictive indicators
- **Business context adaptation** with use case profiles, industry standards, and regulatory requirement integration
- **Explainable aggregation methodology** with comprehensive evidence compilation and transparent reasoning

### File Structure
```
agents/reputation_aggregator/
├── __init__.py
├── advanced_aggregator.py       # Main aggregation agent
├── weighting/
│   ├── __init__.py
│   ├── dynamic_weighter.py      # Dynamic weight calculation
│   ├── context_adapter.py       # Business context integration
│   ├── quality_adjustor.py      # Data quality weight adjustments
│   └── uncertainty_weighter.py  # Uncertainty-based weighting
├── scoring/
│   ├── __init__.py
│   ├── dimensional_scorer.py    # Multi-dimensional score calculation
│   ├── compliance_scorer.py     # Compliance-focused scoring
│   ├── credit_scorer.py         # Credit risk scoring
│   ├── governance_scorer.py     # Governance quality scoring
│   ├── security_scorer.py       # Security risk scoring
│   └── innovation_scorer.py     # Innovation/sophistication scoring
├── uncertainty/
│   ├── __init__.py
│   ├── monte_carlo.py          # Monte Carlo simulation
│   ├── confidence_calculator.py # Confidence interval calculation
│   ├── correlation_analyzer.py  # Cross-component correlation
│   └── temporal_uncertainty.py  # Temporal uncertainty modeling
├── temporal/
│   ├── __init__.py
│   ├── change_detector.py       # Significant change detection
│   ├── trend_analyzer.py        # Trend analysis and prediction
│   ├── pattern_recognizer.py    # Temporal pattern recognition
│   └── attribution_analyzer.py  # Change attribution analysis
├── business/
│   ├── __init__.py
│   ├── use_case_profiles.py     # Business use case profiles
│   ├── industry_standards.py    # Industry standard integration
│   ├── regulatory_mapper.py     # Regulatory requirement mapping
│   └── stakeholder_adapter.py   # Stakeholder-specific adaptation
└── explanation/
    ├── __init__.py
    ├── methodology_explainer.py  # Aggregation methodology explanation
    ├── evidence_compiler.py      # Evidence compilation and presentation
    └── confidence_reporter.py    # Confidence and uncertainty reporting
```

---

## Core Advanced Aggregator Implementation

### Main Aggregation Agent

#### agents/reputation_aggregator/advanced_aggregator.py
```python
"""Advanced ReputationAggregator with sophisticated multi-dimensional scoring."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from dataclasses import dataclass
from enum import Enum

from schemas.agents.state import ReputationAggregatorState, ReputeNetGraphState
from schemas.analysis.reputation import (
    DimensionalReputationScores,
    ReputationAnalysis,
    BusinessContext,
    TemporalAnalysis
)

from .weighting.dynamic_weighter import DynamicWeightCalculator
from .scoring.dimensional_scorer import MultiDimensionalScorer
from .uncertainty.monte_carlo import MonteCarloUncertaintyQuantifier
from .temporal.change_detector import ReputationChangeDetector
from .business.use_case_profiles import BusinessContextAdapter
from .explanation.methodology_explainer import AggregationExplainer

class AggregationStrategy(Enum):
    """Aggregation strategy options."""
    COMPREHENSIVE = "comprehensive"    # Full multi-dimensional analysis
    FOCUSED = "focused"               # Focused on specific dimensions
    BALANCED = "balanced"             # Balanced across all dimensions
    ADAPTIVE = "adaptive"             # Adapt based on data quality and context

@dataclass
class AggregationRequirements:
    """Requirements for reputation aggregation."""
    business_context: BusinessContext
    strategy: AggregationStrategy = AggregationStrategy.ADAPTIVE
    uncertainty_quantification: bool = True
    temporal_analysis: bool = True
    confidence_threshold: float = 0.7
    explanation_depth: str = "comprehensive"  # basic, standard, comprehensive

@dataclass
class ComponentScores:
    """Component scores from upstream agents."""
    address_profiler_scores: Dict[str, float]
    risk_scorer_results: Dict[str, Any]
    sybil_detector_results: Dict[str, Any]
    component_confidences: Dict[str, float]
    data_quality_scores: Dict[str, float]

class AdvancedReputationAggregator:
    """Advanced reputation aggregation with multi-dimensional scoring."""

    def __init__(
        self,
        cache_service: Any,
        metrics_service: Any,
        historical_data_service: Optional[Any] = None
    ):
        self.cache_service = cache_service
        self.metrics_service = metrics_service
        self.historical_data_service = historical_data_service

        self.logger = logging.getLogger(__name__)

        # Initialize core components
        self.weight_calculator = DynamicWeightCalculator()
        self.dimensional_scorer = MultiDimensionalScorer()
        self.uncertainty_quantifier = MonteCarloUncertaintyQuantifier()
        self.change_detector = ReputationChangeDetector()
        self.context_adapter = BusinessContextAdapter()
        self.explainer = AggregationExplainer()

    async def aggregate_reputation(
        self,
        state: ReputeNetGraphState
    ) -> ReputeNetGraphState:
        """Advanced reputation aggregation with multi-dimensional analysis."""

        self.logger.info("Starting advanced reputation aggregation")

        try:
            # Extract aggregation requirements and component scores
            requirements = self._extract_aggregation_requirements(state)
            component_scores = self._extract_component_scores(state)

            # Validate component scores and data quality
            validation_result = await self._validate_component_scores(component_scores)
            if not validation_result.is_valid:
                self.logger.warning(f"Component validation issues: {validation_result.issues}")

            # Phase 1: Calculate dynamic weights based on context and quality
            dynamic_weights = await self._calculate_dynamic_weights(
                component_scores, requirements
            )

            # Phase 2: Generate multi-dimensional scores
            dimensional_scores = await self._generate_dimensional_scores(
                component_scores, dynamic_weights, requirements
            )

            # Phase 3: Quantify uncertainty and confidence intervals
            uncertainty_analysis = await self._quantify_uncertainty(
                dimensional_scores, component_scores, requirements
            )

            # Phase 4: Perform temporal analysis if historical data available
            temporal_analysis = await self._perform_temporal_analysis(
                dimensional_scores, requirements
            )

            # Phase 5: Adapt scores to business context
            business_adapted_scores = await self._adapt_to_business_context(
                dimensional_scores, requirements.business_context
            )

            # Phase 6: Generate comprehensive explanation
            explanation = await self._generate_explanation(
                business_adapted_scores,
                dynamic_weights,
                uncertainty_analysis,
                temporal_analysis,
                requirements
            )

            # Compile final reputation analysis
            reputation_analysis = ReputationAnalysis(
                dimensional_scores=business_adapted_scores,
                uncertainty_analysis=uncertainty_analysis,
                temporal_analysis=temporal_analysis,
                dynamic_weights=dynamic_weights,
                explanation=explanation,
                aggregation_metadata=self._generate_aggregation_metadata(requirements)
            )

            # Update workflow state
            state = self._update_state_with_reputation_analysis(
                state, reputation_analysis, requirements
            )

            self.logger.info(
                f"Reputation aggregation completed: "
                f"composite={reputation_analysis.dimensional_scores.composite_score:.3f}, "
                f"confidence={uncertainty_analysis.overall_confidence:.3f}"
            )

            return state

        except Exception as e:
            self.logger.error(f"Reputation aggregation failed: {e}")
            state.reputation_aggregator.fail_execution(str(e))
            return state

    async def _calculate_dynamic_weights(
        self,
        component_scores: ComponentScores,
        requirements: AggregationRequirements
    ) -> Dict[str, float]:
        """Calculate context-aware dynamic weights."""

        self.logger.debug("Calculating dynamic weights")

        dynamic_weights = await self.weight_calculator.calculate_comprehensive_weights(
            component_scores=component_scores,
            business_context=requirements.business_context,
            strategy=requirements.strategy,
            confidence_threshold=requirements.confidence_threshold
        )

        self.logger.info(f"Dynamic weights calculated: {dynamic_weights}")
        return dynamic_weights

    async def _generate_dimensional_scores(
        self,
        component_scores: ComponentScores,
        dynamic_weights: Dict[str, float],
        requirements: AggregationRequirements
    ) -> DimensionalReputationScores:
        """Generate comprehensive multi-dimensional scores."""

        self.logger.debug("Generating multi-dimensional scores")

        dimensional_scores = await self.dimensional_scorer.calculate_dimensional_scores(
            component_scores=component_scores,
            dynamic_weights=dynamic_weights,
            business_context=requirements.business_context,
            scoring_strategy=requirements.strategy
        )

        # Log dimensional score breakdown
        self.logger.info(
            f"Dimensional scores: compliance={dimensional_scores.compliance_score:.3f}, "
            f"credit={dimensional_scores.credit_risk_score:.3f}, "
            f"governance={dimensional_scores.governance_score:.3f}, "
            f"security={dimensional_scores.security_score:.3f}, "
            f"innovation={dimensional_scores.innovation_score:.3f}"
        )

        return dimensional_scores

    async def _quantify_uncertainty(
        self,
        dimensional_scores: DimensionalReputationScores,
        component_scores: ComponentScores,
        requirements: AggregationRequirements
    ) -> UncertaintyAnalysis:
        """Comprehensive uncertainty quantification."""

        if not requirements.uncertainty_quantification:
            return UncertaintyAnalysis.create_minimal()

        self.logger.debug("Quantifying uncertainty and calculating confidence intervals")

        uncertainty_analysis = await self.uncertainty_quantifier.quantify_comprehensive_uncertainty(
            dimensional_scores=dimensional_scores,
            component_scores=component_scores,
            monte_carlo_samples=10000,
            confidence_levels=[0.68, 0.95, 0.99]
        )

        self.logger.info(
            f"Uncertainty analysis: overall_confidence={uncertainty_analysis.overall_confidence:.3f}, "
            f"confidence_interval_95={uncertainty_analysis.confidence_intervals['95']}"
        )

        return uncertainty_analysis

    async def _perform_temporal_analysis(
        self,
        dimensional_scores: DimensionalReputationScores,
        requirements: AggregationRequirements
    ) -> Optional[TemporalAnalysis]:
        """Perform temporal analysis of reputation evolution."""

        if not requirements.temporal_analysis or not self.historical_data_service:
            return None

        self.logger.debug("Performing temporal reputation analysis")

        # Get historical scores for comparison
        historical_scores = await self.historical_data_service.get_historical_scores(
            address=requirements.business_context.target_address,
            timeframe=timedelta(days=90)
        )

        if not historical_scores:
            self.logger.info("No historical data available for temporal analysis")
            return None

        temporal_analysis = await self.change_detector.analyze_reputation_evolution(
            current_scores=dimensional_scores,
            historical_scores=historical_scores,
            analysis_timeframe=timedelta(days=90)
        )

        self.logger.info(
            f"Temporal analysis: significant_changes={len(temporal_analysis.significant_changes)}, "
            f"trends_detected={len(temporal_analysis.trends)}"
        )

        return temporal_analysis

    async def _adapt_to_business_context(
        self,
        dimensional_scores: DimensionalReputationScores,
        business_context: BusinessContext
    ) -> DimensionalReputationScores:
        """Adapt scores to specific business context."""

        self.logger.debug("Adapting scores to business context")

        adapted_scores = await self.context_adapter.adapt_scores_to_context(
            base_scores=dimensional_scores,
            business_context=business_context
        )

        return adapted_scores

    async def _generate_explanation(
        self,
        dimensional_scores: DimensionalReputationScores,
        dynamic_weights: Dict[str, float],
        uncertainty_analysis: UncertaintyAnalysis,
        temporal_analysis: Optional[TemporalAnalysis],
        requirements: AggregationRequirements
    ) -> AggregationExplanation:
        """Generate comprehensive explanation of aggregation methodology."""

        explanation = await self.explainer.generate_comprehensive_explanation(
            dimensional_scores=dimensional_scores,
            dynamic_weights=dynamic_weights,
            uncertainty_analysis=uncertainty_analysis,
            temporal_analysis=temporal_analysis,
            business_context=requirements.business_context,
            explanation_depth=requirements.explanation_depth
        )

        return explanation

    def _extract_aggregation_requirements(self, state: ReputeNetGraphState) -> AggregationRequirements:
        """Extract aggregation requirements from workflow state."""

        analysis_request = state.analysis_request

        # Determine business context from request
        business_context = BusinessContext(
            target_address=analysis_request.target_address,
            use_case=self._infer_use_case(analysis_request),
            industry=getattr(analysis_request, 'industry', None),
            regulatory_framework=getattr(analysis_request, 'regulatory_framework', None),
            stakeholder_requirements=getattr(analysis_request, 'stakeholder_requirements', {})
        )

        return AggregationRequirements(
            business_context=business_context,
            strategy=AggregationStrategy.ADAPTIVE,
            uncertainty_quantification=True,
            temporal_analysis=True,
            confidence_threshold=0.7,
            explanation_depth="comprehensive"
        )

    def _extract_component_scores(self, state: ReputeNetGraphState) -> ComponentScores:
        """Extract component scores from upstream agents."""

        # Extract AddressProfiler results
        address_profiler_data = state.shared_data.get("address_profiler_results", {})
        profiler_scores = {
            "sophistication_score": address_profiler_data.get("sophistication_score", 0.5),
            "activity_score": address_profiler_data.get("activity_score", 0.5),
            "behavior_consistency": address_profiler_data.get("behavior_consistency", 0.5)
        }

        # Extract RiskScorer results
        risk_scorer_data = state.shared_data.get("risk_analysis_results", {})

        # Extract SybilDetector results
        sybil_detector_data = state.shared_data.get("sybil_analysis_results", {})

        # Extract component confidences
        component_confidences = {
            "address_profiler": state.address_profiler.confidence_score or 0.8,
            "risk_scorer": state.risk_scorer.confidence_score or 0.8,
            "sybil_detector": state.sybil_detector.confidence_score or 0.8
        }

        # Extract data quality scores
        data_quality = state.shared_data.get("data_quality_summary", {})
        data_quality_scores = {
            "overall_quality": data_quality.get("overall_quality", 0.8),
            "completeness": data_quality.get("quality_breakdown", {}).get("completeness", 0.8),
            "consistency": data_quality.get("quality_breakdown", {}).get("consistency", 0.8)
        }

        return ComponentScores(
            address_profiler_scores=profiler_scores,
            risk_scorer_results=risk_scorer_data,
            sybil_detector_results=sybil_detector_data,
            component_confidences=component_confidences,
            data_quality_scores=data_quality_scores
        )

    def _update_state_with_reputation_analysis(
        self,
        state: ReputeNetGraphState,
        reputation_analysis: ReputationAnalysis,
        requirements: AggregationRequirements
    ) -> ReputeNetGraphState:
        """Update workflow state with reputation analysis results."""

        # Update ReputationAggregator agent state
        aggregator_state = state.reputation_aggregator
        aggregator_state.reputation_score = reputation_analysis.dimensional_scores.composite_score
        aggregator_state.reputation_components = reputation_analysis.dimensional_scores.to_dict()
        aggregator_state.confidence_intervals = reputation_analysis.uncertainty_analysis.confidence_intervals

        # Update aggregation metadata
        aggregator_state.aggregation_method = requirements.strategy.value
        aggregator_state.component_weights = reputation_analysis.dynamic_weights
        aggregator_state.data_quality_impact = reputation_analysis.uncertainty_analysis.data_quality_impact

        # Store complete analysis in shared state
        state.shared_data["reputation_analysis"] = reputation_analysis.dict()
        state.shared_data["final_reputation_score"] = reputation_analysis.dimensional_scores.composite_score

        # Add business context information
        state.shared_data["business_context"] = requirements.business_context.dict()

        # Mark completion
        aggregator_state.complete_execution()

        return state

    def _infer_use_case(self, analysis_request) -> str:
        """Infer business use case from analysis request."""

        # Simple heuristics for use case inference
        if getattr(analysis_request, 'enable_compliance_analysis', False):
            return "compliance_screening"
        elif getattr(analysis_request, 'enable_credit_analysis', False):
            return "credit_assessment"
        elif getattr(analysis_request, 'enable_governance_analysis', False):
            return "dao_participation"
        else:
            return "general_assessment"

    def _generate_aggregation_metadata(self, requirements: AggregationRequirements) -> Dict[str, Any]:
        """Generate metadata about the aggregation process."""

        return {
            "strategy_used": requirements.strategy.value,
            "business_context": requirements.business_context.dict(),
            "uncertainty_quantification_enabled": requirements.uncertainty_quantification,
            "temporal_analysis_enabled": requirements.temporal_analysis,
            "confidence_threshold": requirements.confidence_threshold,
            "explanation_depth": requirements.explanation_depth,
            "aggregation_timestamp": datetime.utcnow().isoformat(),
            "methodology_version": "1.0.0"
        }
```

### Dynamic Weight Calculator

#### agents/reputation_aggregator/weighting/dynamic_weighter.py
```python
"""Dynamic weight calculation with context awareness and uncertainty integration."""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class WeightingStrategy(Enum):
    """Available weighting strategies."""
    STATIC = "static"                    # Pre-configured static weights
    QUALITY_ADJUSTED = "quality_adjusted"  # Weights adjusted by data quality
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Weights based on component confidence
    CONTEXT_ADAPTIVE = "context_adaptive"  # Weights adapted to business context
    UNCERTAINTY_OPTIMAL = "uncertainty_optimal"  # Optimize weights for minimum uncertainty

@dataclass
class WeightingContext:
    """Context for weight calculation."""
    business_use_case: str
    data_quality_scores: Dict[str, float]
    component_confidences: Dict[str, float]
    uncertainty_tolerances: Dict[str, float]
    stakeholder_preferences: Dict[str, float]

class DynamicWeightCalculator:
    """Calculate dynamic weights based on context and data quality."""

    def __init__(self):
        self.base_weight_profiles = self._initialize_base_profiles()
        self.quality_adjustment_factors = {
            "high_quality": 1.2,      # Increase weight for high quality components
            "medium_quality": 1.0,    # No adjustment for medium quality
            "low_quality": 0.7        # Decrease weight for low quality components
        }

    async def calculate_comprehensive_weights(
        self,
        component_scores: Any,  # ComponentScores
        business_context: Any,  # BusinessContext
        strategy: Any,          # AggregationStrategy
        confidence_threshold: float = 0.7
    ) -> Dict[str, float]:
        """Calculate comprehensive dynamic weights."""

        # Create weighting context
        context = WeightingContext(
            business_use_case=business_context.use_case,
            data_quality_scores=component_scores.data_quality_scores,
            component_confidences=component_scores.component_confidences,
            uncertainty_tolerances=self._get_uncertainty_tolerances(business_context),
            stakeholder_preferences=getattr(business_context, 'stakeholder_requirements', {})
        )

        # Start with base weights for use case
        base_weights = self._get_base_weights_for_use_case(context.business_use_case)

        # Apply strategy-specific adjustments
        if strategy.value == "adaptive":
            weights = await self._apply_adaptive_weighting(base_weights, context)
        elif strategy.value == "comprehensive":
            weights = await self._apply_comprehensive_weighting(base_weights, context)
        else:
            weights = await self._apply_balanced_weighting(base_weights, context)

        # Apply quality adjustments
        quality_adjusted_weights = self._apply_quality_adjustments(weights, context)

        # Apply confidence adjustments
        confidence_adjusted_weights = self._apply_confidence_adjustments(
            quality_adjusted_weights, context, confidence_threshold
        )

        # Apply uncertainty optimization
        optimized_weights = self._apply_uncertainty_optimization(
            confidence_adjusted_weights, context
        )

        # Normalize and validate weights
        final_weights = self._normalize_and_validate_weights(optimized_weights)

        return final_weights

    def _get_base_weights_for_use_case(self, use_case: str) -> Dict[str, float]:
        """Get base weights for specific use case."""

        return self.base_weight_profiles.get(use_case, self.base_weight_profiles["general"])

    async def _apply_adaptive_weighting(
        self,
        base_weights: Dict[str, float],
        context: WeightingContext
    ) -> Dict[str, float]:
        """Apply adaptive weighting based on context."""

        adaptive_weights = base_weights.copy()

        # Increase weights for high-confidence components
        for component, confidence in context.component_confidences.items():
            if component in adaptive_weights:
                confidence_factor = 0.5 + (confidence * 0.5)  # Scale from 0.5 to 1.0
                adaptive_weights[component] *= confidence_factor

        # Adjust for data quality
        for component, quality in context.data_quality_scores.items():
            mapped_component = self._map_quality_to_component(component)
            if mapped_component in adaptive_weights:
                quality_factor = 0.7 + (quality * 0.3)  # Scale from 0.7 to 1.0
                adaptive_weights[mapped_component] *= quality_factor

        return adaptive_weights

    def _apply_quality_adjustments(
        self,
        weights: Dict[str, float],
        context: WeightingContext
    ) -> Dict[str, float]:
        """Adjust weights based on data quality scores."""

        quality_adjusted = weights.copy()

        for component, weight in weights.items():
            # Map component to quality score
            quality_key = self._map_component_to_quality(component)
            quality_score = context.data_quality_scores.get(quality_key, 0.8)

            # Apply quality adjustment
            if quality_score >= 0.8:
                adjustment_factor = self.quality_adjustment_factors["high_quality"]
            elif quality_score >= 0.6:
                adjustment_factor = self.quality_adjustment_factors["medium_quality"]
            else:
                adjustment_factor = self.quality_adjustment_factors["low_quality"]

            quality_adjusted[component] = weight * adjustment_factor

        return quality_adjusted

    def _apply_confidence_adjustments(
        self,
        weights: Dict[str, float],
        context: WeightingContext,
        confidence_threshold: float
    ) -> Dict[str, float]:
        """Adjust weights based on component confidence scores."""

        confidence_adjusted = weights.copy()

        for component, weight in weights.items():
            # Map component to confidence score
            confidence_key = self._map_component_to_confidence(component)
            confidence = context.component_confidences.get(confidence_key, 0.8)

            # Apply confidence-based adjustment
            if confidence < confidence_threshold:
                # Reduce weight for low-confidence components
                confidence_factor = confidence / confidence_threshold
                confidence_adjusted[component] = weight * confidence_factor
            else:
                # Slightly boost weight for high-confidence components
                confidence_factor = 1.0 + (confidence - confidence_threshold) * 0.5
                confidence_adjusted[component] = weight * confidence_factor

        return confidence_adjusted

    def _apply_uncertainty_optimization(
        self,
        weights: Dict[str, float],
        context: WeightingContext
    ) -> Dict[str, float]:
        """Optimize weights to minimize overall uncertainty."""

        # This is a simplified uncertainty optimization
        # In practice, this would use more sophisticated optimization algorithms

        optimized_weights = weights.copy()

        # Calculate uncertainty impact for each component
        uncertainty_impacts = {}
        for component, weight in weights.items():
            confidence_key = self._map_component_to_confidence(component)
            confidence = context.component_confidences.get(confidence_key, 0.8)
            uncertainty = 1.0 - confidence
            uncertainty_impacts[component] = uncertainty * weight

        # Adjust weights to minimize overall uncertainty
        total_uncertainty = sum(uncertainty_impacts.values())
        if total_uncertainty > 0:
            for component, weight in weights.items():
                uncertainty_ratio = uncertainty_impacts[component] / total_uncertainty
                # Reduce weight for components contributing more to uncertainty
                optimization_factor = 1.0 - (uncertainty_ratio * 0.2)  # Max 20% reduction
                optimized_weights[component] = weight * optimization_factor

        return optimized_weights

    def _normalize_and_validate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0 and validate constraints."""

        # Ensure all weights are positive
        normalized_weights = {k: max(0.01, v) for k, v in weights.items()}

        # Normalize to sum to 1.0
        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in normalized_weights.items()}

        # Ensure no single component dominates
        max_weight = 0.6  # No component should have more than 60% weight
        for component, weight in normalized_weights.items():
            if weight > max_weight:
                normalized_weights[component] = max_weight

        # Re-normalize after capping
        total_weight = sum(normalized_weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in normalized_weights.items()}

        return normalized_weights

    def _initialize_base_profiles(self) -> Dict[str, Dict[str, float]]:
        """Initialize base weight profiles for different use cases."""

        return {
            "compliance_screening": {
                "address_profiler": 0.20,
                "risk_scorer": 0.40,
                "sybil_detector": 0.25,
                "data_quality": 0.15
            },
            "credit_assessment": {
                "address_profiler": 0.35,
                "risk_scorer": 0.35,
                "sybil_detector": 0.20,
                "data_quality": 0.10
            },
            "dao_participation": {
                "address_profiler": 0.40,
                "risk_scorer": 0.20,
                "sybil_detector": 0.30,
                "data_quality": 0.10
            },
            "general": {
                "address_profiler": 0.30,
                "risk_scorer": 0.30,
                "sybil_detector": 0.25,
                "data_quality": 0.15
            }
        }

    def _map_component_to_quality(self, component: str) -> str:
        """Map component name to quality score key."""
        mapping = {
            "address_profiler": "overall_quality",
            "risk_scorer": "completeness",
            "sybil_detector": "consistency",
            "data_quality": "overall_quality"
        }
        return mapping.get(component, "overall_quality")

    def _map_component_to_confidence(self, component: str) -> str:
        """Map component name to confidence score key."""
        return component  # Direct mapping in this case
```

This implementation provides a sophisticated, production-ready reputation aggregation system that demonstrates advanced understanding of reputation complexity while maintaining explainability and business value across multiple use cases.