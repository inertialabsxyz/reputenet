# Step 6: Reporter Agent (Multi-Format Intelligence) - Implementation Notes

**Context:** Complete implementation of sophisticated multi-format intelligence generation with stakeholder customization
**Priority:** Critical for demonstrating business value and competitive differentiation

---

## Implementation Architecture

### Core Framework Structure
```python
# src/agents/reporter/core.py
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta

from pydantic import BaseModel, Field
from langchain.schema import BaseMessage
from langgraph import StateGraph

from ..base import BaseAgent
from ...schemas.state import WorkflowState
from ...schemas.reputation import ReputationAnalysis, MultiDimensionalScores
from ...schemas.reports import (
    IntelligenceReport, ReportConfiguration, StakeholderProfile,
    VisualAssets, EvidenceDocumentation
)

class ReportFormat(str, Enum):
    """Available report format types."""
    TECHNICAL_JSON = "technical_json"
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    VISUAL_DASHBOARD = "visual_dashboard"
    ALERT_REPORT = "alert_report"
    COMPLIANCE_REPORT = "compliance_report"

class StakeholderRole(str, Enum):
    """Stakeholder role types for content adaptation."""
    COMPLIANCE_OFFICER = "compliance_officer"
    RISK_ANALYST = "risk_analyst"
    BUSINESS_EXECUTIVE = "business_executive"
    TECHNICAL_TEAM = "technical_team"
    AUDIT_TEAM = "audit_team"
    REGULATORY_BODY = "regulatory_body"

@dataclass
class ReportConfiguration:
    """Configuration for intelligence report generation."""
    requested_formats: List[ReportFormat]
    stakeholder_profile: StakeholderProfile
    evidence_requirements: str = "comprehensive"
    visualization_requirements: Optional[Dict[str, Any]] = None
    export_formats: List[str] = None
    performance_requirements: Optional[Dict[str, Any]] = None

class ReporterAgent(BaseAgent):
    """Advanced multi-format intelligence generation agent."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.generation_engine = ReportGenerationEngine(config)
        self.stakeholder_adapter = StakeholderAdaptationEngine(config)
        self.evidence_compiler = EvidenceDocumentationEngine(config)
        self.visual_generator = VisualIntelligenceEngine(config)
        self.export_manager = ExportManager(config)

    async def generate_intelligence_report(
        self,
        state: WorkflowState
    ) -> WorkflowState:
        """Generate comprehensive intelligence report from reputation analysis."""

        try:
            # Extract reputation analysis
            reputation_analysis = state.reputation_aggregation_result
            if not reputation_analysis:
                raise ValueError("No reputation analysis available for reporting")

            # Determine report configuration
            report_config = self._determine_report_configuration(state)

            # Generate comprehensive intelligence report
            intelligence_report = await self._generate_report(
                reputation_analysis,
                report_config
            )

            # Update state with generated report
            state.reporter_result = intelligence_report
            state.processing_metadata.append({
                "agent": "reporter",
                "timestamp": datetime.utcnow().isoformat(),
                "report_formats": [f.value for f in report_config.requested_formats],
                "stakeholder_role": report_config.stakeholder_profile.role.value
            })

            return state

        except Exception as e:
            state.errors.append({
                "agent": "reporter",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            return state

    async def _generate_report(
        self,
        analysis: ReputationAnalysis,
        config: ReportConfiguration
    ) -> IntelligenceReport:
        """Generate comprehensive intelligence report."""

        # Stakeholder content adaptation
        adapted_content = await self.stakeholder_adapter.adapt_content(
            analysis,
            config.stakeholder_profile
        )

        # Evidence documentation compilation
        evidence_documentation = await self.evidence_compiler.compile_evidence(
            analysis,
            config.evidence_requirements
        )

        # Visual intelligence generation
        visual_assets = await self.visual_generator.generate_visualizations(
            adapted_content,
            config.visualization_requirements or {}
        )

        # Multi-format report generation
        report_formats = await self.generation_engine.generate_all_formats(
            adapted_content,
            evidence_documentation,
            visual_assets,
            config
        )

        # Export processing
        exported_reports = await self.export_manager.process_exports(
            report_formats,
            config.export_formats or []
        )

        return IntelligenceReport(
            target_address=analysis.address,
            formats=report_formats,
            evidence_documentation=evidence_documentation,
            visual_assets=visual_assets,
            exported_reports=exported_reports,
            generation_metadata=self._generate_report_metadata(config),
            confidence_score=self._calculate_report_confidence(analysis, config)
        )

    def _determine_report_configuration(
        self,
        state: WorkflowState
    ) -> ReportConfiguration:
        """Determine report configuration based on state and requirements."""

        # Default configuration for prototype
        return ReportConfiguration(
            requested_formats=[
                ReportFormat.TECHNICAL_JSON,
                ReportFormat.EXECUTIVE_SUMMARY,
                ReportFormat.DETAILED_ANALYSIS,
                ReportFormat.VISUAL_DASHBOARD
            ],
            stakeholder_profile=StakeholderProfile(
                role=StakeholderRole.RISK_ANALYST,
                expertise_level="expert",
                industry="defi",
                regulatory_framework="us_financial"
            ),
            evidence_requirements="comprehensive",
            visualization_requirements={
                "include_risk_analysis": True,
                "include_temporal_analysis": True,
                "include_network_analysis": True,
                "create_dashboard": True
            },
            export_formats=["pdf", "json", "html"]
        )
```

### Multi-Format Generation Engine
```python
# src/agents/reporter/generation_engine.py
from typing import Dict, Any
import json
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ReportGenerationEngine:
    """Multi-format intelligence report generation."""

    def __init__(self, config: dict):
        self.config = config
        self.template_env = Environment(
            loader=FileSystemLoader('src/templates/reports')
        )
        self.format_generators = {
            ReportFormat.TECHNICAL_JSON: TechnicalJSONGenerator(),
            ReportFormat.EXECUTIVE_SUMMARY: ExecutiveSummaryGenerator(self.template_env),
            ReportFormat.DETAILED_ANALYSIS: DetailedAnalysisGenerator(self.template_env),
            ReportFormat.VISUAL_DASHBOARD: VisualDashboardGenerator(),
            ReportFormat.ALERT_REPORT: AlertReportGenerator(self.template_env),
            ReportFormat.COMPLIANCE_REPORT: ComplianceReportGenerator(self.template_env)
        }

    async def generate_all_formats(
        self,
        content: AdaptedContent,
        evidence: EvidenceDocumentation,
        visuals: VisualAssets,
        config: ReportConfiguration
    ) -> Dict[str, Any]:
        """Generate reports in all requested formats."""

        generated_formats = {}

        for format_type in config.requested_formats:
            generator = self.format_generators[format_type]

            generated_report = await generator.generate(
                content=content,
                evidence=evidence,
                visuals=visuals,
                config=config
            )

            generated_formats[format_type.value] = generated_report

        return generated_formats

class TechnicalJSONGenerator:
    """Generate comprehensive technical JSON report."""

    async def generate(
        self,
        content: AdaptedContent,
        evidence: EvidenceDocumentation,
        visuals: VisualAssets,
        config: ReportConfiguration
    ) -> Dict[str, Any]:
        """Generate technical JSON report."""

        return {
            "report_metadata": {
                "generation_timestamp": datetime.utcnow().isoformat(),
                "report_version": "1.0",
                "target_address": content.address,
                "stakeholder_profile": config.stakeholder_profile.dict(),
                "confidence_score": content.overall_confidence
            },
            "reputation_analysis": {
                "multi_dimensional_scores": {
                    "compliance_score": content.reputation_scores.compliance_score,
                    "credit_risk_score": content.reputation_scores.credit_risk_score,
                    "governance_score": content.reputation_scores.governance_score,
                    "security_score": content.reputation_scores.security_score,
                    "innovation_score": content.reputation_scores.innovation_score,
                    "composite_score": content.reputation_scores.composite_score
                },
                "confidence_intervals": {
                    "compliance": content.confidence_intervals.compliance,
                    "credit_risk": content.confidence_intervals.credit_risk,
                    "governance": content.confidence_intervals.governance,
                    "security": content.confidence_intervals.security,
                    "innovation": content.confidence_intervals.innovation,
                    "composite": content.confidence_intervals.composite
                },
                "temporal_analysis": {
                    "significant_changes": [change.dict() for change in content.temporal_analysis.significant_changes],
                    "trends": content.temporal_analysis.trends.dict(),
                    "patterns": [pattern.dict() for pattern in content.temporal_analysis.patterns],
                    "predictive_indicators": content.temporal_analysis.predictive_indicators.dict()
                }
            },
            "component_analysis": {
                "data_harvester": content.component_results.data_harvester.dict(),
                "address_profiler": content.component_results.address_profiler.dict(),
                "risk_scorer": content.component_results.risk_scorer.dict(),
                "sybil_detector": content.component_results.sybil_detector.dict()
            },
            "evidence_documentation": {
                "methodology_explanation": evidence.methodology_explanation.dict(),
                "supporting_evidence": evidence.supporting_evidence.dict(),
                "confidence_documentation": evidence.confidence_documentation.dict(),
                "uncertainty_explanation": evidence.uncertainty_explanation.dict()
            },
            "visual_assets": {
                "chart_metadata": [chart.metadata.dict() for chart in visuals.charts.values()],
                "dashboard_metadata": visuals.dashboard.metadata.dict() if visuals.dashboard else None
            },
            "actionable_insights": {
                "risk_assessment": content.actionable_insights.risk_assessment.dict(),
                "recommendations": [rec.dict() for rec in content.actionable_insights.recommendations],
                "action_items": [item.dict() for item in content.actionable_insights.action_items],
                "follow_up_guidance": content.actionable_insights.follow_up_guidance.dict()
            }
        }

class ExecutiveSummaryGenerator:
    """Generate executive summary report."""

    def __init__(self, template_env: Environment):
        self.template_env = template_env
        self.template = template_env.get_template('executive_summary.html')

    async def generate(
        self,
        content: AdaptedContent,
        evidence: EvidenceDocumentation,
        visuals: VisualAssets,
        config: ReportConfiguration
    ) -> str:
        """Generate executive summary HTML report."""

        # Prepare executive summary data
        summary_data = {
            "address": content.address,
            "overall_risk_rating": self._calculate_overall_risk_rating(content.reputation_scores),
            "key_findings": self._extract_key_findings(content),
            "risk_highlights": self._extract_risk_highlights(content),
            "recommendations": self._extract_executive_recommendations(content.actionable_insights),
            "confidence_level": self._map_confidence_to_executive_level(content.overall_confidence),
            "methodology_summary": self._create_methodology_summary(evidence.methodology_explanation),
            "visual_charts": self._prepare_executive_visuals(visuals),
            "temporal_insights": self._extract_temporal_insights(content.temporal_analysis),
            "compliance_status": self._assess_compliance_status(content.reputation_scores),
            "next_steps": self._generate_next_steps(content.actionable_insights)
        }

        # Render template
        return self.template.render(**summary_data)

    def _calculate_overall_risk_rating(self, scores: MultiDimensionalScores) -> str:
        """Calculate overall risk rating for executive summary."""

        composite_score = scores.composite_score

        if composite_score >= 0.8:
            return "LOW_RISK"
        elif composite_score >= 0.6:
            return "MODERATE_RISK"
        elif composite_score >= 0.4:
            return "HIGH_RISK"
        else:
            return "CRITICAL_RISK"

    def _extract_key_findings(self, content: AdaptedContent) -> List[Dict[str, Any]]:
        """Extract key findings for executive presentation."""

        findings = []

        # Highest scoring dimension
        scores = content.reputation_scores
        score_dict = {
            "Compliance": scores.compliance_score,
            "Credit Risk": scores.credit_risk_score,
            "Governance": scores.governance_score,
            "Security": scores.security_score,
            "Innovation": scores.innovation_score
        }

        highest_dimension = max(score_dict, key=score_dict.get)
        findings.append({
            "type": "strength",
            "title": f"Strong {highest_dimension} Profile",
            "description": f"Demonstrates high {highest_dimension.lower()} score of {score_dict[highest_dimension]:.2f}",
            "impact": "positive"
        })

        # Lowest scoring dimension
        lowest_dimension = min(score_dict, key=score_dict.get)
        if score_dict[lowest_dimension] < 0.5:
            findings.append({
                "type": "concern",
                "title": f"Low {lowest_dimension} Score",
                "description": f"Requires attention with {lowest_dimension.lower()} score of {score_dict[lowest_dimension]:.2f}",
                "impact": "negative"
            })

        # Temporal changes
        if content.temporal_analysis.significant_changes:
            findings.append({
                "type": "change",
                "title": "Recent Reputation Changes Detected",
                "description": f"Identified {len(content.temporal_analysis.significant_changes)} significant changes",
                "impact": "neutral"
            })

        return findings

class DetailedAnalysisGenerator:
    """Generate comprehensive detailed analysis report."""

    def __init__(self, template_env: Environment):
        self.template_env = template_env
        self.template = template_env.get_template('detailed_analysis.html')

    async def generate(
        self,
        content: AdaptedContent,
        evidence: EvidenceDocumentation,
        visuals: VisualAssets,
        config: ReportConfiguration
    ) -> str:
        """Generate detailed analysis HTML report."""

        # Prepare detailed analysis data
        analysis_data = {
            "address": content.address,
            "generation_timestamp": datetime.utcnow().isoformat(),
            "multi_dimensional_analysis": self._create_dimensional_analysis(content.reputation_scores),
            "confidence_analysis": self._create_confidence_analysis(content.confidence_intervals),
            "temporal_analysis": self._create_temporal_analysis(content.temporal_analysis),
            "component_analysis": self._create_component_analysis(content.component_results),
            "methodology_documentation": evidence.methodology_explanation,
            "supporting_evidence": evidence.supporting_evidence,
            "uncertainty_documentation": evidence.uncertainty_explanation,
            "visual_charts": visuals.charts,
            "actionable_insights": content.actionable_insights,
            "appendices": self._create_appendices(content, evidence)
        }

        # Render template
        return self.template.render(**analysis_data)

    def _create_dimensional_analysis(
        self,
        scores: MultiDimensionalScores
    ) -> Dict[str, Any]:
        """Create detailed dimensional score analysis."""

        return {
            "compliance": {
                "score": scores.compliance_score,
                "interpretation": self._interpret_compliance_score(scores.compliance_score),
                "factors": self._extract_compliance_factors(scores),
                "benchmarking": self._benchmark_compliance_score(scores.compliance_score)
            },
            "credit_risk": {
                "score": scores.credit_risk_score,
                "interpretation": self._interpret_credit_risk_score(scores.credit_risk_score),
                "factors": self._extract_credit_risk_factors(scores),
                "benchmarking": self._benchmark_credit_risk_score(scores.credit_risk_score)
            },
            "governance": {
                "score": scores.governance_score,
                "interpretation": self._interpret_governance_score(scores.governance_score),
                "factors": self._extract_governance_factors(scores),
                "benchmarking": self._benchmark_governance_score(scores.governance_score)
            },
            "security": {
                "score": scores.security_score,
                "interpretation": self._interpret_security_score(scores.security_score),
                "factors": self._extract_security_factors(scores),
                "benchmarking": self._benchmark_security_score(scores.security_score)
            },
            "innovation": {
                "score": scores.innovation_score,
                "interpretation": self._interpret_innovation_score(scores.innovation_score),
                "factors": self._extract_innovation_factors(scores),
                "benchmarking": self._benchmark_innovation_score(scores.innovation_score)
            }
        }
```

### Visual Intelligence Engine
```python
# src/agents/reporter/visual_intelligence.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class VisualIntelligenceEngine:
    """Advanced visualization generation for stakeholder communication."""

    def __init__(self, config: dict):
        self.config = config
        self.chart_generators = {
            'risk_heatmap': RiskHeatmapGenerator(),
            'temporal_trends': TemporalTrendGenerator(),
            'network_analysis': NetworkAnalysisGenerator(),
            'comparative_analysis': ComparativeAnalysisGenerator(),
            'confidence_visualization': ConfidenceVisualizationGenerator(),
            'dimensional_radar': DimensionalRadarGenerator()
        }
        self.dashboard_composer = DashboardComposer()

    async def generate_visualizations(
        self,
        content: AdaptedContent,
        requirements: Dict[str, Any]
    ) -> VisualAssets:
        """Generate comprehensive visual intelligence assets."""

        visual_assets = {}

        # Risk assessment heatmap
        if requirements.get('include_risk_analysis', True):
            risk_heatmap = await self.chart_generators['risk_heatmap'].generate(
                content.reputation_scores,
                content.confidence_intervals
            )
            visual_assets['risk_heatmap'] = risk_heatmap

        # Temporal trend analysis
        if requirements.get('include_temporal_analysis', True):
            temporal_trends = await self.chart_generators['temporal_trends'].generate(
                content.temporal_analysis
            )
            visual_assets['temporal_trends'] = temporal_trends

        # Multi-dimensional radar chart
        if requirements.get('include_dimensional_analysis', True):
            dimensional_radar = await self.chart_generators['dimensional_radar'].generate(
                content.reputation_scores,
                content.confidence_intervals
            )
            visual_assets['dimensional_radar'] = dimensional_radar

        # Confidence interval visualization
        if requirements.get('include_confidence_analysis', True):
            confidence_viz = await self.chart_generators['confidence_visualization'].generate(
                content.confidence_intervals,
                content.uncertainty_quantification
            )
            visual_assets['confidence_visualization'] = confidence_viz

        # Interactive dashboard composition
        if requirements.get('create_dashboard', True):
            dashboard = await self.dashboard_composer.compose_dashboard(
                visual_assets,
                content
            )
            visual_assets['dashboard'] = dashboard

        return VisualAssets(
            charts=visual_assets,
            generation_metadata=self._generate_visual_metadata(requirements)
        )

class RiskHeatmapGenerator:
    """Generate sophisticated risk assessment heatmaps."""

    async def generate(
        self,
        scores: MultiDimensionalScores,
        confidence_intervals: ConfidenceIntervals
    ) -> go.Figure:
        """Generate risk heatmap visualization."""

        # Prepare heatmap data
        dimensions = ['Compliance', 'Credit Risk', 'Governance', 'Security', 'Innovation']
        risk_levels = ['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']

        # Create risk score matrix
        score_values = [
            scores.compliance_score,
            scores.credit_risk_score,
            scores.governance_score,
            scores.security_score,
            scores.innovation_score
        ]

        # Map scores to risk levels (inverted scale for risk)
        risk_matrix = []
        for score in score_values:
            risk_row = [0, 0, 0, 0]
            if score >= 0.8:
                risk_row[0] = score  # Low risk
            elif score >= 0.6:
                risk_row[1] = score  # Moderate risk
            elif score >= 0.4:
                risk_row[2] = score  # High risk
            else:
                risk_row[3] = score  # Critical risk
            risk_matrix.append(risk_row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=risk_levels,
            y=dimensions,
            colorscale='RdYlGn',
            reversescale=True,
            text=[[f'{val:.2f}' if val > 0 else '' for val in row] for row in risk_matrix],
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(
                title="Risk Score",
                titleside="right"
            )
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Multi-Dimensional Risk Assessment Heatmap',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial, sans-serif'}
            },
            xaxis_title="Risk Level",
            yaxis_title="Assessment Dimension",
            width=800,
            height=500,
            font=dict(family="Arial, sans-serif", size=12)
        )

        return fig

class DimensionalRadarGenerator:
    """Generate multi-dimensional radar chart."""

    async def generate(
        self,
        scores: MultiDimensionalScores,
        confidence_intervals: ConfidenceIntervals
    ) -> go.Figure:
        """Generate dimensional radar chart with confidence bands."""

        # Prepare radar chart data
        dimensions = ['Compliance', 'Credit Risk', 'Governance', 'Security', 'Innovation']

        score_values = [
            scores.compliance_score,
            scores.credit_risk_score,
            scores.governance_score,
            scores.security_score,
            scores.innovation_score
        ]

        # Add confidence intervals
        upper_bounds = [
            confidence_intervals.compliance[1],
            confidence_intervals.credit_risk[1],
            confidence_intervals.governance[1],
            confidence_intervals.security[1],
            confidence_intervals.innovation[1]
        ]

        lower_bounds = [
            confidence_intervals.compliance[0],
            confidence_intervals.credit_risk[0],
            confidence_intervals.governance[0],
            confidence_intervals.security[0],
            confidence_intervals.innovation[0]
        ]

        # Create radar chart
        fig = go.Figure()

        # Add upper confidence bound
        fig.add_trace(go.Scatterpolar(
            r=upper_bounds + [upper_bounds[0]],  # Close the polygon
            theta=dimensions + [dimensions[0]],
            fill=None,
            line=dict(color='lightblue', width=1, dash='dash'),
            name='Upper Confidence',
            showlegend=True
        ))

        # Add lower confidence bound
        fig.add_trace(go.Scatterpolar(
            r=lower_bounds + [lower_bounds[0]],
            theta=dimensions + [dimensions[0]],
            fill='tonext',
            fillcolor='rgba(173, 216, 230, 0.3)',
            line=dict(color='lightblue', width=1, dash='dash'),
            name='Confidence Interval',
            showlegend=True
        ))

        # Add actual scores
        fig.add_trace(go.Scatterpolar(
            r=score_values + [score_values[0]],
            theta=dimensions + [dimensions[0]],
            fill=None,
            line=dict(color='darkblue', width=3),
            marker=dict(size=8, color='darkblue'),
            name='Reputation Scores',
            showlegend=True
        ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=['0.2', '0.4', '0.6', '0.8', '1.0']
                )
            ),
            title={
                'text': 'Multi-Dimensional Reputation Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial, sans-serif'}
            },
            width=600,
            height=600,
            font=dict(family="Arial, sans-serif", size=12)
        )

        return fig

class TemporalTrendGenerator:
    """Generate temporal trend analysis visualizations."""

    async def generate(
        self,
        temporal_analysis: TemporalAnalysis
    ) -> go.Figure:
        """Generate temporal trend visualization."""

        # Create subplots for different temporal aspects
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Score Evolution Over Time',
                'Change Detection Analysis',
                'Trend Patterns',
                'Predictive Indicators'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Mock temporal data for visualization
        dates = pd.date_range(start='2024-01-01', end='2024-10-29', freq='D')

        # Score evolution (subplot 1,1)
        compliance_trend = 0.7 + 0.1 * np.sin(np.arange(len(dates)) / 10) + np.random.normal(0, 0.02, len(dates))
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=compliance_trend,
                mode='lines',
                name='Compliance Score',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Change detection (subplot 1,2)
        change_dates = dates[::30]  # Monthly changes
        change_magnitudes = [0.05, -0.03, 0.08, -0.02, 0.04, -0.01, 0.06, -0.04, 0.03, 0.02]
        fig.add_trace(
            go.Bar(
                x=change_dates,
                y=change_magnitudes,
                name='Score Changes',
                marker_color=['green' if x > 0 else 'red' for x in change_magnitudes]
            ),
            row=1, col=2
        )

        # Trend patterns (subplot 2,1)
        trend_data = np.cumsum(np.random.normal(0, 0.01, len(dates))) + 0.7
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=trend_data,
                mode='lines',
                name='Overall Trend',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )

        # Predictive indicators (subplot 2,2)
        prediction_horizon = pd.date_range(start='2024-10-30', end='2024-12-31', freq='D')
        predicted_scores = 0.75 + 0.05 * np.sin(np.arange(len(prediction_horizon)) / 5)
        fig.add_trace(
            go.Scatter(
                x=prediction_horizon,
                y=predicted_scores,
                mode='lines',
                name='Predicted Trend',
                line=dict(color='orange', width=2, dash='dash')
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title={
                'text': 'Temporal Reputation Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial, sans-serif'}
            },
            height=600,
            width=1000,
            font=dict(family="Arial, sans-serif", size=10)
        )

        return fig
```

### Export Manager
```python
# src/agents/reporter/export_manager.py
import json
import base64
from pathlib import Path
import weasyprint
from jinja2 import Environment, FileSystemLoader

class ExportManager:
    """Manage report exports in multiple formats."""

    def __init__(self, config: dict):
        self.config = config
        self.export_processors = {
            'pdf': PDFExportProcessor(),
            'json': JSONExportProcessor(),
            'html': HTMLExportProcessor(),
            'excel': ExcelExportProcessor(),
            'csv': CSVExportProcessor()
        }

    async def process_exports(
        self,
        report_formats: Dict[str, Any],
        export_formats: List[str]
    ) -> Dict[str, bytes]:
        """Process report exports in requested formats."""

        exported_reports = {}

        for export_format in export_formats:
            if export_format in self.export_processors:
                processor = self.export_processors[export_format]
                exported_data = await processor.export(report_formats)
                exported_reports[export_format] = exported_data

        return exported_reports

class PDFExportProcessor:
    """Export reports to PDF format."""

    async def export(self, report_formats: Dict[str, Any]) -> bytes:
        """Export comprehensive report to PDF."""

        # Use detailed analysis HTML as base for PDF
        if 'detailed_analysis' in report_formats:
            html_content = report_formats['detailed_analysis']
        elif 'executive_summary' in report_formats:
            html_content = report_formats['executive_summary']
        else:
            # Create simple HTML from JSON data
            html_content = self._create_simple_html(report_formats)

        # Convert HTML to PDF
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        return pdf_bytes

    def _create_simple_html(self, report_formats: Dict[str, Any]) -> str:
        """Create simple HTML from available report data."""

        # Extract JSON data if available
        json_data = report_formats.get('technical_json', {})

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reputation Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .section { margin-bottom: 20px; }
                .scores { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
                .score-card { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Blockchain Address Reputation Analysis</h1>
                <p>Generated: {timestamp}</p>
            </div>

            <div class="section">
                <h2>Multi-Dimensional Scores</h2>
                <div class="scores">
                    {score_cards}
                </div>
            </div>

            <div class="section">
                <h2>Technical Details</h2>
                <pre>{json_data}</pre>
            </div>
        </body>
        </html>
        """

        # Generate score cards
        score_cards = ""
        if 'reputation_analysis' in json_data:
            scores = json_data['reputation_analysis'].get('multi_dimensional_scores', {})
            for dimension, score in scores.items():
                score_cards += f"""
                <div class="score-card">
                    <h3>{dimension.replace('_', ' ').title()}</h3>
                    <p>Score: {score:.3f}</p>
                </div>
                """

        return html_template.format(
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            score_cards=score_cards,
            json_data=json.dumps(json_data, indent=2)
        )

class JSONExportProcessor:
    """Export reports to JSON format."""

    async def export(self, report_formats: Dict[str, Any]) -> bytes:
        """Export technical JSON report."""

        # Use technical JSON if available, otherwise compile from other formats
        if 'technical_json' in report_formats:
            json_data = report_formats['technical_json']
        else:
            json_data = self._compile_json_from_formats(report_formats)

        return json.dumps(json_data, indent=2, default=str).encode('utf-8')

    def _compile_json_from_formats(self, report_formats: Dict[str, Any]) -> Dict[str, Any]:
        """Compile JSON data from available report formats."""

        return {
            "available_formats": list(report_formats.keys()),
            "export_timestamp": datetime.utcnow().isoformat(),
            "report_data": report_formats
        }
```

### Template System
```python
# src/templates/reports/executive_summary.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary - Reputation Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .risk-rating {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .risk-low { background-color: #28a745; }
        .risk-moderate { background-color: #ffc107; color: #000; }
        .risk-high { background-color: #fd7e14; }
        .risk-critical { background-color: #dc3545; }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }

        .findings-list {
            list-style: none;
            padding: 0;
        }

        .finding-item {
            background: white;
            margin-bottom: 15px;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .finding-item.concern { border-left-color: #dc3545; }
        .finding-item.change { border-left-color: #ffc107; }

        .recommendations {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Executive Summary</h1>
        <h2>Blockchain Address Reputation Analysis</h2>
        <p><strong>Address:</strong> {{ address }}</p>
        <div class="risk-rating risk-{{ overall_risk_rating.lower().replace('_risk', '').replace('_', '-') }}">
            {{ overall_risk_rating.replace('_', ' ') }}
        </div>
        <p><strong>Confidence Level:</strong> {{ confidence_level }}</p>
    </div>

    <div class="summary-grid">
        <div class="card">
            <h3>🎯 Key Findings</h3>
            <ul class="findings-list">
                {% for finding in key_findings %}
                <li class="finding-item {{ finding.type }}">
                    <strong>{{ finding.title }}</strong><br>
                    {{ finding.description }}
                </li>
                {% endfor %}
            </ul>
        </div>

        <div class="card">
            <h3>⚠️ Risk Highlights</h3>
            {% for risk in risk_highlights %}
            <div class="finding-item">
                <strong>{{ risk.category }}</strong><br>
                {{ risk.description }}
                <br><small>Impact: {{ risk.impact }}</small>
            </div>
            {% endfor %}
        </div>

        <div class="card">
            <h3>📊 Compliance Status</h3>
            <div class="finding-item">
                <strong>Overall Status:</strong> {{ compliance_status.overall_status }}<br>
                <strong>Regulatory Score:</strong> {{ compliance_status.regulatory_score }}<br>
                <strong>AML Risk:</strong> {{ compliance_status.aml_risk_level }}
            </div>
        </div>

        <div class="card">
            <h3>📈 Temporal Insights</h3>
            {% for insight in temporal_insights %}
            <div class="finding-item">
                <strong>{{ insight.period }}:</strong> {{ insight.description }}<br>
                <small>Trend: {{ insight.trend_direction }}</small>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="chart-container">
        <h3>Multi-Dimensional Risk Assessment</h3>
        {% for chart_name, chart_data in visual_charts.items() %}
        <div>{{ chart_data.html|safe }}</div>
        {% endfor %}
    </div>

    <div class="recommendations">
        <h3>🎯 Strategic Recommendations</h3>
        {% for recommendation in recommendations %}
        <div class="finding-item">
            <strong>{{ recommendation.priority }} Priority:</strong> {{ recommendation.title }}<br>
            {{ recommendation.description }}<br>
            <small><strong>Expected Impact:</strong> {{ recommendation.expected_impact }}</small>
        </div>
        {% endfor %}

        <h3>📋 Next Steps</h3>
        <ol>
            {% for step in next_steps %}
            <li><strong>{{ step.action }}:</strong> {{ step.description }} ({{ step.timeframe }})</li>
            {% endfor %}
        </ol>
    </div>

    <div class="card" style="margin-top: 20px; text-align: center; color: #666;">
        <h4>Methodology Summary</h4>
        <p>{{ methodology_summary }}</p>
        <small>Generated: {{ generation_timestamp }} | Confidence: {{ confidence_level }}</small>
    </div>
</body>
</html>
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Complete missing step documentation for Phase 1", "status": "completed", "activeForm": "Completed missing step documentation for Phase 1"}, {"content": "Complete missing step documentation for Phase 2", "status": "completed", "activeForm": "Completed missing step documentation for Phase 2"}, {"content": "Complete missing step documentation for Phase 3", "status": "completed", "activeForm": "Completed missing step documentation for Phase 3"}, {"content": "Validate methodology compliance across all phases", "status": "pending", "activeForm": "Validating methodology compliance across all phases"}]