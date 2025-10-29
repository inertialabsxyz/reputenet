# Phase 3: Agent Implementation - Sophisticated Deep Prototype Runbook

**Target:** Advanced agents with stakeholder wow factor + business value demonstration
**Prerequisites:** Phase 2 complete (sophisticated mock infrastructure, LangGraph orchestration)
**Duration:** 3-4 days (24-30 hours total)
**Context:** Sophisticated prototype targeting business leaders + technical teams

---

## Implementation Strategy

### **Day 1: Foundation + Advanced Threat Detection (8 hours)**
- Morning: AddressProfiler foundation + sophisticated features (4 hours)
- Afternoon: RiskScorer advanced threat detection (4 hours)

### **Day 2: Expert Analysis + Coordination Detection (8 hours)**
- Morning: AddressProfiler expert features + RiskScorer expert analysis (4 hours)
- Afternoon: SybilDetector coordination analysis (4 hours)

### **Day 3: Integration + Business Intelligence (8 hours)**
- Morning: SybilDetector advanced capabilities (3 hours)
- Afternoon: ReputationAggregator multi-dimensional scoring (4 hours)
- Evening: Integration testing (1 hour)

### **Day 4: Polish + Stakeholder Presentation (6 hours)**
- Morning: Reporter stakeholder presentation (3 hours)
- Afternoon: Demo preparation + DataHarvester polish (3 hours)

---

## Step-by-Step Implementation

### Step 1: AddressProfiler Agent - Advanced Behavioral Analysis (6 hours)

#### 1.1 Foundation Features Implementation (2 hours)
```python
# src/reputenet/agents/profiler.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import structlog
from ..schema import Transaction, AddressFeatures
from ..utils.statistics import calculate_gini_coefficient, analyze_time_series

logger = structlog.get_logger()

class AddressProfilerAgent:
    """Advanced behavioral feature extraction showcasing deep blockchain expertise."""

    def __init__(self):
        self.feature_cache = {}
        self.sophistication_models = self._load_sophistication_models()

    def extract_features(self, address: str, raw_data: Dict[str, Any]) -> AddressFeatures:
        """Extract comprehensive behavioral features."""

        transactions = raw_data.get("txs", [])
        logs = raw_data.get("logs", [])
        tokens = raw_data.get("tokens", [])

        logger.info("Extracting features for address", address=address, tx_count=len(transactions))

        # Three-tier feature extraction
        foundation_features = self._extract_foundation_features(transactions)
        sophisticated_features = self._extract_sophisticated_features(transactions, logs, tokens)
        expert_features = self._extract_expert_features(transactions, foundation_features, sophisticated_features)

        # Combine all features
        all_features = {**foundation_features, **sophisticated_features, **expert_features}

        logger.info("Feature extraction complete",
                   address=address,
                   feature_count=len(all_features),
                   sophistication_score=all_features.get("defi_sophistication_score", 0))

        return AddressFeatures(**all_features)

    def _extract_foundation_features(self, transactions: List[Transaction]) -> Dict[str, float]:
        """Extract essential behavioral metrics."""

        if not transactions:
            return self._get_empty_foundation_features()

        # Temporal analysis
        timestamps = [tx.timestamp for tx in transactions]
        account_age = (max(timestamps) - min(timestamps)) / (24 * 3600)  # days

        # Transaction patterns
        tx_values = [tx.value for tx in transactions if tx.value > 0]
        gas_prices = [tx.gas_price for tx in transactions]

        # Network analysis
        counterparties = [tx.to_address for tx in transactions if tx.to_address]
        unique_counterparties = len(set(counterparties))

        return {
            # Temporal patterns
            "account_age_days": account_age,
            "transaction_frequency": len(transactions) / max(account_age, 1),
            "activity_consistency": self._calculate_activity_consistency(timestamps),

            # Economic patterns
            "total_volume_eth": sum(tx_values),
            "average_transaction_value": np.mean(tx_values) if tx_values else 0,
            "value_distribution_gini": calculate_gini_coefficient(tx_values) if len(tx_values) > 1 else 0,

            # Network patterns
            "unique_counterparties": unique_counterparties,
            "counterparty_concentration": self._calculate_concentration_ratio(counterparties),
            "counterparty_diversity_index": self._calculate_diversity_index(counterparties),

            # Gas intelligence
            "gas_optimization_score": self._analyze_gas_optimization(transactions),
            "gas_price_strategy_score": self._classify_gas_strategy(gas_prices),
            "mev_protection_signals": self._detect_mev_protection(transactions)
        }

    def _calculate_activity_consistency(self, timestamps: List[float]) -> float:
        """Calculate consistency of activity patterns."""
        if len(timestamps) < 2:
            return 0.0

        # Convert to daily activity counts
        daily_activity = {}
        for ts in timestamps:
            day = datetime.fromtimestamp(ts).date()
            daily_activity[day] = daily_activity.get(day, 0) + 1

        if len(daily_activity) < 2:
            return 0.0

        # Calculate coefficient of variation (lower = more consistent)
        activity_counts = list(daily_activity.values())
        mean_activity = np.mean(activity_counts)
        std_activity = np.std(activity_counts)

        if mean_activity == 0:
            return 0.0

        cv = std_activity / mean_activity
        # Convert to consistency score (0-1, higher = more consistent)
        return max(0, 1 - (cv / 2))  # Normalize CV to 0-1 range

    def _analyze_gas_optimization(self, transactions: List[Transaction]) -> float:
        """Analyze gas usage optimization sophistication."""

        gas_usage = [(tx.gas_used, tx.gas_price) for tx in transactions]
        if len(gas_usage) < 5:
            return 0.0

        # Analyze gas price vs network conditions (mock analysis)
        gas_prices = [gp for _, gp in gas_usage]

        # Check for sophisticated gas strategies
        optimization_signals = 0

        # Signal 1: Consistent gas price optimization
        gas_price_std = np.std(gas_prices)
        gas_price_mean = np.mean(gas_prices)
        if gas_price_mean > 0 and (gas_price_std / gas_price_mean) < 0.3:
            optimization_signals += 1

        # Signal 2: Gas limit optimization
        gas_limits = [gu for gu, _ in gas_usage]
        if len(set(gas_limits)) > len(gas_limits) * 0.8:  # Varied gas limits
            optimization_signals += 1

        # Signal 3: No obviously wasteful transactions
        wasteful_txs = sum(1 for gu, gp in gas_usage if gp > gas_price_mean * 2)
        if wasteful_txs / len(gas_usage) < 0.1:
            optimization_signals += 1

        return min(optimization_signals / 3.0, 1.0)
```

#### 1.2 Sophisticated DeFi Analysis (2 hours)
```python
    def _extract_sophisticated_features(self, transactions: List[Transaction],
                                     logs: List[Dict], tokens: List[Dict]) -> Dict[str, float]:
        """Extract advanced DeFi sophistication and strategy analysis."""

        # Protocol interaction analysis
        protocols = self._identify_protocols(transactions)
        defi_strategies = self._detect_defi_strategies(transactions, protocols)

        return {
            # DeFi sophistication metrics
            "defi_sophistication_score": self._calculate_defi_sophistication(defi_strategies),
            "yield_farming_intensity": self._analyze_yield_farming_patterns(transactions, protocols),
            "leverage_usage_sophistication": self._analyze_leverage_strategies(transactions),
            "liquidity_provision_score": self._analyze_lp_strategies(transactions, logs),
            "governance_participation_level": self._analyze_governance_activity(transactions),
            "protocol_diversification_index": self._calculate_protocol_diversification(protocols),

            # Risk management analysis
            "position_sizing_discipline": self._analyze_position_sizing(transactions),
            "diversification_score": self._calculate_diversification_score(transactions, tokens),
            "liquidation_avoidance_skill": self._analyze_liquidation_management(transactions),
            "risk_adjusted_returns": self._calculate_risk_adjusted_performance(transactions),

            # Capital efficiency metrics
            "capital_velocity": self._calculate_capital_velocity(transactions),
            "idle_capital_ratio": self._calculate_idle_capital_ratio(transactions),
            "compound_interest_optimization": self._analyze_compounding_behavior(transactions),
            "gas_to_profit_ratio": self._calculate_gas_efficiency_ratio(transactions)
        }

    def _calculate_defi_sophistication(self, strategies: Dict[str, Any]) -> float:
        """Calculate overall DeFi sophistication score."""

        sophistication_components = []

        # Strategy diversity (0-0.3)
        strategy_count = len([s for s in strategies.values() if s.get("detected", False)])
        strategy_score = min(strategy_count / 5.0, 1.0) * 0.3
        sophistication_components.append(strategy_score)

        # Strategy execution quality (0-0.4)
        execution_scores = [s.get("execution_quality", 0) for s in strategies.values() if s.get("detected")]
        execution_score = (np.mean(execution_scores) if execution_scores else 0) * 0.4
        sophistication_components.append(execution_score)

        # Advanced feature usage (0-0.3)
        advanced_features = [
            strategies.get("leverage_loops", {}).get("detected", False),
            strategies.get("flash_loans", {}).get("detected", False),
            strategies.get("governance_participation", {}).get("detected", False),
            strategies.get("cross_protocol_arbitrage", {}).get("detected", False)
        ]
        advanced_score = sum(advanced_features) / len(advanced_features) * 0.3
        sophistication_components.append(advanced_score)

        return sum(sophistication_components)

    def _detect_defi_strategies(self, transactions: List[Transaction],
                              protocols: Dict[str, int]) -> Dict[str, Any]:
        """Detect sophisticated DeFi strategies."""

        strategies = {}

        # Yield farming detection
        strategies["yield_farming"] = self._detect_yield_farming(transactions, protocols)

        # Leverage strategy detection
        strategies["leverage_loops"] = self._detect_leverage_loops(transactions)

        # Liquidity provision detection
        strategies["liquidity_provision"] = self._detect_liquidity_provision(transactions)

        # Arbitrage detection
        strategies["arbitrage"] = self._detect_arbitrage_behavior(transactions)

        # Flash loan usage
        strategies["flash_loans"] = self._detect_flash_loan_usage(transactions)

        # Governance participation
        strategies["governance_participation"] = self._detect_governance_participation(transactions)

        return strategies

    def _detect_yield_farming(self, transactions: List[Transaction],
                            protocols: Dict[str, int]) -> Dict[str, Any]:
        """Detect yield farming patterns and sophistication."""

        # Look for patterns indicating yield farming
        farming_indicators = {
            "liquidity_mining_signals": 0,
            "strategy_rotation_signals": 0,
            "yield_optimization_signals": 0,
            "execution_quality": 0
        }

        # Check for liquidity mining patterns
        lp_transactions = [tx for tx in transactions if self._is_liquidity_provision(tx)]
        if len(lp_transactions) > 2:
            farming_indicators["liquidity_mining_signals"] = 1

            # Analyze execution quality
            gas_efficiency = self._analyze_lp_gas_efficiency(lp_transactions)
            timing_quality = self._analyze_lp_timing(lp_transactions)
            farming_indicators["execution_quality"] = (gas_efficiency + timing_quality) / 2

        # Check for strategy rotation
        if len(protocols) > 2:
            rotation_patterns = self._analyze_protocol_rotation(transactions, protocols)
            farming_indicators["strategy_rotation_signals"] = rotation_patterns

        return {
            "detected": any(farming_indicators[k] > 0 for k in ["liquidity_mining_signals", "strategy_rotation_signals"]),
            "sophistication_score": np.mean(list(farming_indicators.values())),
            "indicators": farming_indicators
        }
```

#### 1.3 Expert-Level Analysis (2 hours)
```python
    def _extract_expert_features(self, transactions: List[Transaction],
                               foundation: Dict[str, float],
                               sophisticated: Dict[str, float]) -> Dict[str, float]:
        """Extract expert-level analysis demonstrating production-ready sophistication."""

        return {
            # Market impact and awareness
            "market_impact_awareness": self._analyze_market_impact_awareness(transactions),
            "slippage_optimization": self._analyze_slippage_management(transactions),
            "timing_strategy_sophistication": self._analyze_timing_strategies(transactions),
            "mev_avoidance_capability": self._analyze_mev_avoidance_patterns(transactions),

            # Coordination and clustering signals
            "coordination_probability": self._calculate_coordination_probability(transactions),
            "behavioral_fingerprint_uniqueness": self._calculate_behavioral_uniqueness(transactions),
            "timing_correlation_score": self._analyze_timing_correlations(transactions),

            # Institutional vs retail signals
            "institutional_probability": self._calculate_institutional_probability(transactions, foundation, sophisticated),
            "treasury_management_signals": self._detect_treasury_patterns(transactions),
            "compliance_awareness_score": self._analyze_compliance_patterns(transactions),
            "professional_trading_signals": self._detect_professional_patterns(transactions),
            "automated_strategy_probability": self._detect_automation_patterns(transactions)
        }

    def _calculate_institutional_probability(self, transactions: List[Transaction],
                                          foundation: Dict[str, float],
                                          sophisticated: Dict[str, float]) -> float:
        """Calculate probability that address represents institutional activity."""

        institutional_signals = []

        # Signal 1: Volume and sophistication
        volume_signal = min(foundation.get("total_volume_eth", 0) / 1000, 1.0)  # Normalize by 1000 ETH
        sophistication_signal = sophisticated.get("defi_sophistication_score", 0)
        institutional_signals.append((volume_signal + sophistication_signal) / 2)

        # Signal 2: Diversification and risk management
        diversification_signal = sophisticated.get("diversification_score", 0)
        risk_mgmt_signal = sophisticated.get("position_sizing_discipline", 0)
        institutional_signals.append((diversification_signal + risk_mgmt_signal) / 2)

        # Signal 3: Governance and long-term behavior
        governance_signal = sophisticated.get("governance_participation_level", 0)
        consistency_signal = foundation.get("activity_consistency", 0)
        institutional_signals.append((governance_signal + consistency_signal) / 2)

        # Signal 4: Professional execution patterns
        gas_optimization = foundation.get("gas_optimization_score", 0)
        capital_efficiency = sophisticated.get("capital_velocity", 0)
        execution_signal = (gas_optimization + min(capital_efficiency / 10, 1.0)) / 2
        institutional_signals.append(execution_signal)

        return np.mean(institutional_signals)

    def _analyze_market_impact_awareness(self, transactions: List[Transaction]) -> float:
        """Analyze awareness of market impact and sophisticated execution."""

        large_transactions = [tx for tx in transactions if tx.value > 10]  # >10 ETH
        if len(large_transactions) < 3:
            return 0.0

        awareness_signals = 0
        total_signals = 0

        for tx in large_transactions:
            total_signals += 1

            # Check for size splitting (multiple smaller transactions instead of one large)
            similar_time_txs = [t for t in transactions
                              if abs(t.timestamp - tx.timestamp) < 3600  # Within 1 hour
                              and t.to_address == tx.to_address]

            if len(similar_time_txs) > 1:
                awareness_signals += 1

        return awareness_signals / total_signals if total_signals > 0 else 0.0

    def _detect_professional_patterns(self, transactions: List[Transaction]) -> float:
        """Detect patterns indicating professional/algorithmic trading."""

        if len(transactions) < 10:
            return 0.0

        professional_signals = []

        # Signal 1: Consistent timing patterns
        time_intervals = []
        sorted_txs = sorted(transactions, key=lambda x: x.timestamp)
        for i in range(1, len(sorted_txs)):
            interval = sorted_txs[i].timestamp - sorted_txs[i-1].timestamp
            time_intervals.append(interval)

        if time_intervals:
            cv_intervals = np.std(time_intervals) / np.mean(time_intervals) if np.mean(time_intervals) > 0 else 1
            timing_consistency = max(0, 1 - cv_intervals)  # Lower CV = more consistent
            professional_signals.append(timing_consistency)

        # Signal 2: Gas price consistency (algorithmic optimization)
        gas_prices = [tx.gas_price for tx in transactions]
        if len(gas_prices) > 1:
            cv_gas = np.std(gas_prices) / np.mean(gas_prices) if np.mean(gas_prices) > 0 else 1
            gas_consistency = max(0, 1 - cv_gas)
            professional_signals.append(gas_consistency)

        # Signal 3: Round number avoidance (algorithmic precision)
        round_number_txs = sum(1 for tx in transactions if tx.value % 1 == 0)  # Whole ETH amounts
        round_avoidance = 1 - (round_number_txs / len(transactions))
        professional_signals.append(round_avoidance)

        return np.mean(professional_signals) if professional_signals else 0.0
```

### Step 2: RiskScorer Agent - Advanced Threat Detection (6 hours)

#### 2.1 Foundation Risk Detection (2 hours)
```python
# src/reputenet/agents/risk_scorer.py
import numpy as np
from typing import Dict, List, Any, Set
import structlog
from datetime import datetime, timedelta
from ..schema import Transaction, RiskOutput
from ..tools.labels import LabelRegistry

logger = structlog.get_logger()

class RiskScorerAgent:
    """Advanced multi-level risk detection showcasing security expertise."""

    def __init__(self, label_registry: LabelRegistry):
        self.labels = label_registry
        self.threat_intelligence = self._load_threat_intelligence()
        self.risk_models = self._load_risk_models()

    def assess_risk(self, address: str, raw_data: Dict[str, Any],
                   features: Dict[str, Any]) -> RiskOutput:
        """Perform comprehensive multi-level risk assessment."""

        transactions = raw_data.get("txs", [])

        logger.info("Assessing risk for address", address=address, tx_count=len(transactions))

        # Three-level risk analysis
        foundation_risks = self._assess_foundation_risks(transactions)
        advanced_risks = self._assess_advanced_risks(transactions, features)
        expert_risks = self._assess_expert_risks(transactions, features, foundation_risks, advanced_risks)

        # Calculate composite risk score
        composite_score = self._calculate_composite_risk_score({
            "foundation": foundation_risks,
            "advanced": advanced_risks,
            "expert": expert_risks
        })

        # Compile evidence
        evidence = self._compile_risk_evidence(foundation_risks, advanced_risks, expert_risks)

        logger.info("Risk assessment complete",
                   address=address,
                   risk_score=composite_score["composite_risk_score"],
                   high_severity_findings=len(evidence.get("high_severity_findings", [])))

        return RiskOutput(
            risk_score=composite_score["composite_risk_score"],
            evidence=evidence["evidence_summary"],
            components=composite_score["category_scores"]
        )

    def _assess_foundation_risks(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Assess essential security risks that every system should detect."""

        foundation_risks = {
            "known_threats": self._detect_known_threats(transactions),
            "suspicious_approvals": self._detect_suspicious_approvals(transactions),
            "basic_anomalies": self._detect_basic_anomalies(transactions),
            "sanctions_violations": self._detect_sanctions_violations(transactions)
        }

        return {
            "category_score": self._calculate_foundation_risk_score(foundation_risks),
            "findings": foundation_risks,
            "severity": "foundation"
        }

    def _detect_known_threats(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect interactions with known malicious contracts."""

        threat_interactions = {
            "mixer_interactions": [],
            "scam_contracts": [],
            "phishing_contracts": [],
            "blacklisted_addresses": []
        }

        for tx in transactions:
            # Check against known threat databases
            if self._is_mixer_contract(tx.to_address):
                threat_interactions["mixer_interactions"].append({
                    "tx_hash": tx.hash,
                    "contract": tx.to_address,
                    "value": tx.value,
                    "timestamp": tx.timestamp
                })

            if self.labels.is_flagged_contract(tx.to_address):
                label_info = self.labels.get_label(tx.to_address)
                threat_interactions["scam_contracts"].append({
                    "tx_hash": tx.hash,
                    "contract": tx.to_address,
                    "label": label_info,
                    "value": tx.value
                })

        # Calculate threat score
        total_threat_interactions = sum(len(threats) for threats in threat_interactions.values())
        threat_score = min(total_threat_interactions * 10, 100)  # 10 points per interaction, max 100

        return {
            "threat_score": threat_score,
            "threat_interactions": threat_interactions,
            "total_interactions": total_threat_interactions
        }

    def _detect_suspicious_approvals(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect dangerous approval patterns."""

        approval_txs = [tx for tx in transactions if self._is_approval_transaction(tx)]

        suspicious_patterns = {
            "unlimited_approvals": [],
            "approval_for_all_patterns": [],
            "never_revoked_approvals": [],
            "rapid_approval_sequences": []
        }

        for tx in approval_txs:
            # Check for unlimited approvals (max uint256)
            if self._is_unlimited_approval(tx):
                suspicious_patterns["unlimited_approvals"].append({
                    "tx_hash": tx.hash,
                    "spender": tx.to_address,
                    "timestamp": tx.timestamp
                })

            # Check for ERC-721/1155 approval for all
            if self._is_approval_for_all(tx):
                suspicious_patterns["approval_for_all_patterns"].append({
                    "tx_hash": tx.hash,
                    "operator": tx.to_address,
                    "timestamp": tx.timestamp
                })

        # Check for never-revoked approvals
        never_revoked = self._find_never_revoked_approvals(approval_txs, transactions)
        suspicious_patterns["never_revoked_approvals"].extend(never_revoked)

        # Calculate approval risk score
        risk_components = [
            len(suspicious_patterns["unlimited_approvals"]) * 15,  # 15 points each
            len(suspicious_patterns["approval_for_all_patterns"]) * 20,  # 20 points each
            len(suspicious_patterns["never_revoked_approvals"]) * 10,  # 10 points each
        ]

        approval_risk_score = min(sum(risk_components), 100)

        return {
            "approval_risk_score": approval_risk_score,
            "suspicious_patterns": suspicious_patterns,
            "total_suspicious_approvals": sum(len(p) for p in suspicious_patterns.values())
        }

    def _is_mixer_contract(self, address: str) -> bool:
        """Check if address is a known mixer contract."""
        mixer_contracts = {
            "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",  # Tornado Cash 1 ETH
            "0x910Cbd523D972eb0a6f4cAe4618aD62622b39DbF",  # Tornado Cash 10 ETH
            "0x12D66f87A04A9E220743712cE6d9bB1B5616B8Fc",  # Tornado Cash 0.1 ETH
        }
        return address.lower() in [addr.lower() for addr in mixer_contracts]
```

#### 2.2 Advanced Threat Analysis (3 hours)
```python
    def _assess_advanced_risks(self, transactions: List[Transaction],
                             features: Dict[str, Any]) -> Dict[str, Any]:
        """Assess sophisticated threats requiring advanced detection."""

        advanced_risks = {
            "defi_exploitation": self._detect_defi_exploitation_patterns(transactions),
            "mev_abuse": self._detect_mev_abuse_patterns(transactions),
            "bridge_security_risks": self._detect_bridge_security_risks(transactions),
            "governance_manipulation": self._detect_governance_manipulation(transactions),
            "flash_loan_attacks": self._detect_flash_loan_attacks(transactions)
        }

        return {
            "category_score": self._calculate_advanced_risk_score(advanced_risks),
            "findings": advanced_risks,
            "severity": "advanced"
        }

    def _detect_defi_exploitation_patterns(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect sophisticated DeFi exploitation strategies."""

        exploitation_patterns = {
            "oracle_manipulation_attacks": [],
            "liquidation_manipulation": [],
            "sandwich_attack_participation": [],
            "protocol_governance_attacks": [],
            "reentrancy_exploits": []
        }

        # Detect oracle manipulation patterns
        oracle_attacks = self._detect_oracle_manipulation(transactions)
        exploitation_patterns["oracle_manipulation_attacks"] = oracle_attacks

        # Detect liquidation manipulation
        liquidation_manipulation = self._detect_liquidation_manipulation(transactions)
        exploitation_patterns["liquidation_manipulation"] = liquidation_manipulation

        # Detect sandwich attacks
        sandwich_attacks = self._detect_sandwich_attacks(transactions)
        exploitation_patterns["sandwich_attack_participation"] = sandwich_attacks

        # Calculate exploitation sophistication score
        total_exploits = sum(len(exploits) for exploits in exploitation_patterns.values())
        sophistication_factors = [
            len(exploitation_patterns["oracle_manipulation_attacks"]) * 25,  # Highly sophisticated
            len(exploitation_patterns["liquidation_manipulation"]) * 20,
            len(exploitation_patterns["sandwich_attack_participation"]) * 15,
        ]

        exploitation_score = min(sum(sophistication_factors), 100)

        return {
            "exploitation_score": exploitation_score,
            "patterns": exploitation_patterns,
            "sophistication_level": self._calculate_exploitation_sophistication(exploitation_patterns)
        }

    def _detect_oracle_manipulation(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect oracle manipulation attack patterns."""

        oracle_attacks = []

        # Look for patterns indicating oracle manipulation
        for tx in transactions:
            manipulation_signals = {
                "large_single_sided_liquidity": False,
                "rapid_price_impact": False,
                "cross_protocol_arbitrage": False,
                "flash_loan_usage": False
            }

            # Check for large liquidity operations
            if tx.value > 100 and self._is_liquidity_operation(tx):  # >100 ETH liquidity operation
                manipulation_signals["large_single_sided_liquidity"] = True

            # Check for flash loan usage in same block/transaction
            if self._involves_flash_loan(tx):
                manipulation_signals["flash_loan_usage"] = True

            # If multiple signals present, flag as potential oracle manipulation
            signal_count = sum(manipulation_signals.values())
            if signal_count >= 2:
                oracle_attacks.append({
                    "tx_hash": tx.hash,
                    "manipulation_signals": manipulation_signals,
                    "signal_strength": signal_count,
                    "value": tx.value,
                    "timestamp": tx.timestamp,
                    "methodology": self._classify_manipulation_method(manipulation_signals)
                })

        return oracle_attacks

    def _detect_sandwich_attacks(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect sandwich attack patterns."""

        sandwich_attacks = []

        # Sort transactions by timestamp
        sorted_txs = sorted(transactions, key=lambda x: x.timestamp)

        # Look for sandwich patterns (front-run -> victim -> back-run)
        for i in range(len(sorted_txs) - 2):
            tx1, tx2, tx3 = sorted_txs[i], sorted_txs[i+1], sorted_txs[i+2]

            # Check if transactions are close in time (same block or adjacent blocks)
            if (tx3.timestamp - tx1.timestamp) < 60:  # Within 1 minute

                # Check for sandwich pattern characteristics
                if (self._is_same_token_pair(tx1, tx3) and  # Same trading pair
                    self._is_opposite_direction(tx1, tx3) and  # Opposite directions
                    tx1.gas_price > tx2.gas_price and  # Front-run has higher gas
                    tx3.gas_price < tx2.gas_price):  # Back-run has lower gas

                    profit_estimate = self._estimate_sandwich_profit(tx1, tx2, tx3)

                    sandwich_attacks.append({
                        "front_run_tx": tx1.hash,
                        "victim_tx": tx2.hash,
                        "back_run_tx": tx3.hash,
                        "estimated_profit": profit_estimate,
                        "victim_impact": self._calculate_victim_impact(tx1, tx2, tx3),
                        "sophistication": self._score_sandwich_sophistication(tx1, tx2, tx3)
                    })

        return sandwich_attacks

    def _detect_flash_loan_attacks(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Detect flash loan attack patterns."""

        flash_loan_attacks = []

        flash_loan_txs = [tx for tx in transactions if self._involves_flash_loan(tx)]

        for tx in flash_loan_txs:
            attack_analysis = {
                "tx_hash": tx.hash,
                "loan_amount": self._extract_flash_loan_amount(tx),
                "attack_vector": self._identify_attack_vector(tx),
                "target_protocols": self._identify_target_protocols(tx),
                "profit_extracted": self._calculate_profit_extraction(tx),
                "sophistication_level": self._score_attack_sophistication(tx)
            }

            # Classify attack type
            if self._is_oracle_manipulation_attack(tx):
                attack_analysis["attack_type"] = "oracle_manipulation"
                attack_analysis["oracle_targets"] = self._identify_oracle_targets(tx)
            elif self._is_governance_attack(tx):
                attack_analysis["attack_type"] = "governance_manipulation"
                attack_analysis["governance_impact"] = self._analyze_governance_impact(tx)
            elif self._is_liquidity_attack(tx):
                attack_analysis["attack_type"] = "liquidity_manipulation"
                attack_analysis["liquidity_impact"] = self._analyze_liquidity_impact(tx)

            # Only include if significant profit or sophisticated methodology
            if (attack_analysis["profit_extracted"] > 1000 or  # >$1000 profit
                attack_analysis["sophistication_level"] > 0.7):
                flash_loan_attacks.append(attack_analysis)

        return flash_loan_attacks
```

#### 2.3 Expert Threat Intelligence (1 hour)
```python
    def _assess_expert_risks(self, transactions: List[Transaction],
                           features: Dict[str, Any],
                           foundation_risks: Dict[str, Any],
                           advanced_risks: Dict[str, Any]) -> Dict[str, Any]:
        """Expert-level threat detection showcasing production-ready capabilities."""

        expert_risks = {
            "advanced_money_laundering": self._detect_advanced_money_laundering(transactions),
            "institutional_fraud": self._detect_institutional_fraud_patterns(transactions, features),
            "zero_day_exploitation": self._detect_zero_day_exploitation(transactions),
            "attack_attribution": self._analyze_attack_attribution(transactions, foundation_risks, advanced_risks),
            "threat_actor_profiling": self._profile_threat_actor(transactions, features)
        }

        return {
            "category_score": self._calculate_expert_risk_score(expert_risks),
            "findings": expert_risks,
            "severity": "expert"
        }

    def _detect_advanced_money_laundering(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect sophisticated money laundering and obfuscation techniques."""

        laundering_analysis = {
            "multi_hop_obfuscation": self._detect_multi_hop_laundering(transactions),
            "temporal_obfuscation": self._detect_temporal_laundering(transactions),
            "volume_fragmentation": self._detect_volume_fragmentation(transactions),
            "chain_hopping_patterns": self._detect_chain_hopping(transactions),
            "legitimate_activity_mixing": self._detect_legitimate_mixing(transactions)
        }

        # Calculate laundering sophistication score
        sophistication_score = 0
        evidence_strength = 0

        for method, analysis in laundering_analysis.items():
            if analysis.get("detected", False):
                sophistication_score += analysis.get("sophistication_score", 0)
                evidence_strength += analysis.get("evidence_strength", 0)

        return {
            "laundering_sophistication": min(sophistication_score, 100),
            "evidence_strength": min(evidence_strength, 100),
            "obfuscation_methods": laundering_analysis,
            "detection_evasion_capability": self._analyze_evasion_capability(laundering_analysis)
        }

    def _profile_threat_actor(self, transactions: List[Transaction],
                            features: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced threat actor profiling and attribution."""

        actor_profile = {
            "sophistication_level": self._calculate_actor_sophistication(transactions, features),
            "operational_patterns": self._analyze_operational_patterns(transactions),
            "target_preferences": self._analyze_target_preferences(transactions),
            "attack_methodology": self._analyze_attack_methodology(transactions),
            "resource_indicators": self._analyze_resource_indicators(transactions, features)
        }

        # Classify threat actor type
        actor_classification = self._classify_threat_actor(actor_profile)

        return {
            "actor_classification": actor_classification,
            "confidence_level": self._calculate_attribution_confidence(actor_profile),
            "profile_details": actor_profile,
            "threat_level": self._assess_threat_level(actor_profile, actor_classification)
        }

    def _calculate_composite_risk_score(self, risk_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate sophisticated composite risk score with dynamic weighting."""

        # Base risk weights
        risk_weights = {
            "foundation": 0.3,    # Basic security threats
            "advanced": 0.45,     # Sophisticated exploitation
            "expert": 0.25        # Advanced threat intelligence
        }

        # Extract category scores
        category_scores = {}
        for category, analysis in risk_analyses.items():
            category_scores[category] = analysis.get("category_score", 0)

        # Calculate weighted composite score
        composite_score = sum(
            score * risk_weights[category]
            for category, score in category_scores.items()
        )

        # Apply sophistication bonus for advanced threats
        if category_scores.get("expert", 0) > 70:
            composite_score = min(composite_score * 1.1, 100)  # 10% bonus for expert-level threats

        return {
            "composite_risk_score": round(composite_score, 1),
            "category_scores": category_scores,
            "risk_weights": risk_weights,
            "sophistication_bonus_applied": category_scores.get("expert", 0) > 70
        }
```

### Step 3: SybilDetector Agent - Coordination Analysis (6 hours)

#### 3.1 Core Coordination Detection (3 hours)
```python
# src/reputenet/agents/sybil_detector.py
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
import structlog
from ..schema import Transaction, SybilOutput
from ..utils.graph_analysis import GraphAnalyzer

logger = structlog.get_logger()

class SybilDetectorAgent:
    """Advanced coordination pattern detection and sybil behavior analysis."""

    def __init__(self):
        self.graph_analyzer = GraphAnalyzer()
        self.coordination_models = self._load_coordination_models()
        self.cluster_cache = {}

    def detect_sybil_behavior(self, address: str, raw_data: Dict[str, Any],
                            features: Dict[str, Any],
                            address_network: Dict[str, Any]) -> SybilOutput:
        """Perform comprehensive sybil and coordination analysis."""

        transactions = raw_data.get("txs", [])

        logger.info("Detecting sybil behavior", address=address, tx_count=len(transactions))

        # Multi-level coordination analysis
        cluster_analysis = self._detect_cluster_membership(address, transactions, address_network)
        coordination_analysis = self._analyze_coordination_patterns(address, transactions, features)
        temporal_analysis = self._analyze_temporal_coordination(transactions)
        behavioral_analysis = self._analyze_behavioral_similarity(address, features, address_network)

        # Calculate composite sybil score
        sybil_score = self._calculate_sybil_score({
            "cluster": cluster_analysis,
            "coordination": coordination_analysis,
            "temporal": temporal_analysis,
            "behavioral": behavioral_analysis
        })

        # Compile evidence
        signals = self._compile_sybil_signals(cluster_analysis, coordination_analysis,
                                            temporal_analysis, behavioral_analysis)

        logger.info("Sybil detection complete",
                   address=address,
                   sybil_score=sybil_score,
                   signals_detected=len(signals))

        return SybilOutput(
            sybil_score=sybil_score,
            signals=signals,
            cluster_id=cluster_analysis.get("cluster_id")
        )

    def _detect_cluster_membership(self, address: str, transactions: List[Transaction],
                                 address_network: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if address belongs to coordinated cluster."""

        # Build transaction graph
        graph = self._build_transaction_graph(transactions, address_network)

        # Detect communities/clusters
        clusters = self.graph_analyzer.detect_communities(graph)

        # Find address cluster
        address_cluster = None
        for cluster_id, members in clusters.items():
            if address in members:
                address_cluster = cluster_id
                break

        if address_cluster is None:
            return {"cluster_detected": False, "cluster_id": None}

        cluster_members = clusters[address_cluster]

        # Analyze cluster characteristics
        cluster_analysis = {
            "cluster_detected": True,
            "cluster_id": address_cluster,
            "cluster_size": len(cluster_members),
            "cluster_members": list(cluster_members),
            "cluster_cohesion": self._calculate_cluster_cohesion(cluster_members, graph),
            "shared_patterns": self._analyze_shared_patterns(cluster_members, address_network)
        }

        # Detect cluster coordination level
        coordination_metrics = self._analyze_cluster_coordination(cluster_members, address_network)
        cluster_analysis.update(coordination_metrics)

        return cluster_analysis

    def _analyze_coordination_patterns(self, address: str, transactions: List[Transaction],
                                     features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sophisticated coordination patterns."""

        coordination_signals = {
            "shared_funding_sources": self._detect_shared_funding(address, transactions),
            "synchronized_activities": self._detect_synchronized_activities(transactions),
            "coordinated_strategies": self._detect_coordinated_strategies(transactions, features),
            "timing_correlations": self._analyze_timing_correlations(transactions)
        }

        # Calculate coordination score
        coordination_score = 0
        evidence_count = 0

        for signal_type, analysis in coordination_signals.items():
            if analysis.get("detected", False):
                coordination_score += analysis.get("strength", 0) * analysis.get("weight", 1)
                evidence_count += 1

        return {
            "coordination_detected": evidence_count > 0,
            "coordination_score": min(coordination_score, 100),
            "evidence_count": evidence_count,
            "coordination_signals": coordination_signals
        }

    def _detect_shared_funding(self, address: str, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect shared funding sources indicating coordination."""

        # Analyze incoming transactions for funding patterns
        incoming_txs = [tx for tx in transactions if tx.to_address.lower() == address.lower()]

        funding_sources = defaultdict(list)
        for tx in incoming_txs:
            funding_sources[tx.from_address].append({
                "amount": tx.value,
                "timestamp": tx.timestamp,
                "tx_hash": tx.hash
            })

        # Look for suspicious funding patterns
        suspicious_funding = {
            "large_single_source": [],
            "multiple_exact_amounts": [],
            "burst_funding": [],
            "exchange_pattern_funding": []
        }

        for source, funding_txs in funding_sources.items():
            # Check for large funding from single source
            total_funding = sum(tx["amount"] for tx in funding_txs)
            if total_funding > 10 and len(funding_txs) > 1:  # >10 ETH in multiple txs
                suspicious_funding["large_single_source"].append({
                    "source": source,
                    "total_amount": total_funding,
                    "transaction_count": len(funding_txs)
                })

            # Check for exact amount patterns
            amounts = [tx["amount"] for tx in funding_txs]
            if len(set(amounts)) == 1 and len(amounts) > 2:  # Same amount multiple times
                suspicious_funding["multiple_exact_amounts"].append({
                    "source": source,
                    "amount": amounts[0],
                    "repetitions": len(amounts)
                })

            # Check for burst funding (multiple transactions in short time)
            if len(funding_txs) > 2:
                timestamps = [tx["timestamp"] for tx in funding_txs]
                time_span = max(timestamps) - min(timestamps)
                if time_span < 3600:  # All within 1 hour
                    suspicious_funding["burst_funding"].append({
                        "source": source,
                        "transaction_count": len(funding_txs),
                        "time_span_minutes": time_span / 60
                    })

        # Calculate shared funding strength
        total_suspicious = sum(len(patterns) for patterns in suspicious_funding.values())
        strength = min(total_suspicious * 20, 100)  # 20 points per suspicious pattern

        return {
            "detected": total_suspicious > 0,
            "strength": strength,
            "weight": 0.3,
            "suspicious_patterns": suspicious_funding
        }

    def _detect_synchronized_activities(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Detect synchronized activity patterns indicating coordination."""

        # Analyze transaction timing for synchronization signals
        timestamps = [tx.timestamp for tx in transactions]

        synchronization_signals = {
            "burst_patterns": [],
            "regular_intervals": [],
            "coordinated_timing": []
        }

        # Detect burst activity patterns
        if len(timestamps) > 5:
            # Group transactions by time windows
            time_windows = defaultdict(list)
            for ts in timestamps:
                window = int(ts / 3600) * 3600  # 1-hour windows
                time_windows[window].append(ts)

            # Look for burst windows (many transactions in short time)
            for window, window_txs in time_windows.items():
                if len(window_txs) > 5:  # >5 transactions in 1 hour
                    synchronization_signals["burst_patterns"].append({
                        "window_start": window,
                        "transaction_count": len(window_txs),
                        "intensity": len(window_txs) / 1  # txs per hour
                    })

        # Detect regular interval patterns
        if len(timestamps) > 3:
            intervals = []
            sorted_timestamps = sorted(timestamps)
            for i in range(1, len(sorted_timestamps)):
                interval = sorted_timestamps[i] - sorted_timestamps[i-1]
                intervals.append(interval)

            # Check for regular patterns
            if intervals:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                if interval_mean > 0 and (interval_std / interval_mean) < 0.5:  # Low coefficient of variation
                    synchronization_signals["regular_intervals"].append({
                        "average_interval_hours": interval_mean / 3600,
                        "regularity_score": 1 - (interval_std / interval_mean),
                        "pattern_length": len(intervals)
                    })

        # Calculate synchronization strength
        total_patterns = sum(len(patterns) for patterns in synchronization_signals.values())
        strength = min(total_patterns * 25, 100)  # 25 points per pattern

        return {
            "detected": total_patterns > 0,
            "strength": strength,
            "weight": 0.25,
            "synchronization_patterns": synchronization_signals
        }
```

#### 3.2 Advanced Behavioral Analysis (3 hours)
```python
    def _analyze_behavioral_similarity(self, address: str, features: Dict[str, Any],
                                     address_network: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral similarity with other addresses."""

        # Get features for comparison addresses
        comparison_addresses = self._get_comparison_addresses(address, address_network)

        similarity_analysis = {
            "feature_similarity": [],
            "strategy_similarity": [],
            "timing_similarity": [],
            "gas_pattern_similarity": []
        }

        # Compare features with other addresses
        for comp_addr, comp_features in comparison_addresses.items():
            feature_similarity = self._calculate_feature_similarity(features, comp_features)

            if feature_similarity > 0.8:  # High similarity threshold
                similarity_analysis["feature_similarity"].append({
                    "address": comp_addr,
                    "similarity_score": feature_similarity,
                    "similar_features": self._identify_similar_features(features, comp_features)
                })

        # Analyze strategy similarity
        strategy_similarity = self._analyze_strategy_similarity(address, features, comparison_addresses)
        similarity_analysis["strategy_similarity"] = strategy_similarity

        # Calculate behavioral similarity strength
        total_similar = sum(len(similarities) for similarities in similarity_analysis.values())
        strength = min(total_similar * 15, 100)  # 15 points per similar address

        return {
            "detected": total_similar > 0,
            "strength": strength,
            "weight": 0.2,
            "similarity_analysis": similarity_analysis,
            "unique_behavior_score": 100 - strength  # Inverse of similarity
        }

    def _analyze_temporal_coordination(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Analyze temporal patterns indicating coordination."""

        timestamps = [tx.timestamp for tx in transactions]

        temporal_patterns = {
            "time_of_day_clustering": self._analyze_time_of_day_patterns(timestamps),
            "day_of_week_patterns": self._analyze_day_of_week_patterns(timestamps),
            "market_event_correlation": self._analyze_market_event_correlation(timestamps),
            "coordination_windows": self._detect_coordination_windows(timestamps)
        }

        # Calculate temporal coordination score
        coordination_indicators = 0

        for pattern_type, analysis in temporal_patterns.items():
            if analysis.get("coordination_detected", False):
                coordination_indicators += analysis.get("strength", 0)

        strength = min(coordination_indicators, 100)

        return {
            "detected": coordination_indicators > 0,
            "strength": strength,
            "weight": 0.25,
            "temporal_patterns": temporal_patterns
        }

    def _calculate_sybil_score(self, analyses: Dict[str, Dict]) -> int:
        """Calculate composite sybil score from multiple analyses."""

        # Weight different types of evidence
        weights = {
            "cluster": 0.35,      # Cluster membership is strong evidence
            "coordination": 0.30,  # Direct coordination patterns
            "temporal": 0.20,     # Temporal coordination
            "behavioral": 0.15    # Behavioral similarity
        }

        # Calculate weighted score
        weighted_score = 0
        total_weight = 0

        for analysis_type, analysis in analyses.items():
            if analysis.get("detected", False):
                score = analysis.get("strength", 0)
                weight = weights.get(analysis_type, 0)
                weighted_score += score * weight
                total_weight += weight

        # Normalize by total weight used
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0

        # Apply cluster size bonus for large coordinated groups
        cluster_analysis = analyses.get("cluster", {})
        if cluster_analysis.get("cluster_detected", False):
            cluster_size = cluster_analysis.get("cluster_size", 0)
            if cluster_size > 10:
                size_bonus = min((cluster_size - 10) * 2, 20)  # Up to 20 point bonus
                final_score = min(final_score + size_bonus, 100)

        return round(final_score)

    def _compile_sybil_signals(self, cluster_analysis: Dict[str, Any],
                             coordination_analysis: Dict[str, Any],
                             temporal_analysis: Dict[str, Any],
                             behavioral_analysis: Dict[str, Any]) -> List[str]:
        """Compile evidence signals for sybil behavior."""

        signals = []

        # Cluster membership signals
        if cluster_analysis.get("cluster_detected", False):
            cluster_size = cluster_analysis.get("cluster_size", 0)
            signals.append(f"cluster_membership:size_{cluster_size}")

            if cluster_analysis.get("cluster_cohesion", 0) > 0.7:
                signals.append("high_cluster_cohesion")

        # Coordination signals
        coordination_signals = coordination_analysis.get("coordination_signals", {})
        for signal_type, analysis in coordination_signals.items():
            if analysis.get("detected", False):
                strength = analysis.get("strength", 0)
                if strength > 50:
                    signals.append(f"{signal_type}:strength_{strength}")

        # Temporal coordination signals
        temporal_patterns = temporal_analysis.get("temporal_patterns", {})
        for pattern_type, analysis in temporal_patterns.items():
            if analysis.get("coordination_detected", False):
                signals.append(f"temporal_{pattern_type}")

        # Behavioral similarity signals
        similarity_analysis = behavioral_analysis.get("similarity_analysis", {})
        for similarity_type, similarities in similarity_analysis.items():
            if len(similarities) > 0:
                signals.append(f"{similarity_type}:count_{len(similarities)}")

        return signals

    def _build_transaction_graph(self, transactions: List[Transaction],
                               address_network: Dict[str, Any]) -> nx.Graph:
        """Build graph representation of transaction relationships."""

        graph = nx.Graph()

        # Add transaction edges
        for tx in transactions:
            # Add edge between sender and receiver
            graph.add_edge(tx.from_address, tx.to_address,
                          weight=tx.value,
                          timestamp=tx.timestamp,
                          tx_hash=tx.hash)

        # Add network information if available
        for addr1, connections in address_network.items():
            for addr2, connection_data in connections.items():
                if graph.has_edge(addr1, addr2):
                    # Update existing edge with network data
                    graph[addr1][addr2].update(connection_data)
                else:
                    # Add new edge based on network relationship
                    graph.add_edge(addr1, addr2, **connection_data)

        return graph

    def _analyze_cluster_coordination(self, cluster_members: List[str],
                                    address_network: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze coordination patterns within identified cluster."""

        coordination_metrics = {
            "funding_coordination": self._analyze_cluster_funding(cluster_members, address_network),
            "activity_synchronization": self._analyze_cluster_activity_sync(cluster_members, address_network),
            "strategy_alignment": self._analyze_cluster_strategy_alignment(cluster_members, address_network),
            "communication_patterns": self._analyze_cluster_communication(cluster_members, address_network)
        }

        # Calculate overall cluster coordination score
        coordination_scores = [metrics.get("score", 0) for metrics in coordination_metrics.values()]
        overall_coordination = np.mean(coordination_scores) if coordination_scores else 0

        return {
            "cluster_coordination_score": overall_coordination,
            "coordination_metrics": coordination_metrics,
            "coordination_strength": "high" if overall_coordination > 70 else "medium" if overall_coordination > 40 else "low"
        }
```

### Step 4: Integration and Testing (4 hours)

#### 4.1 Integration Testing (2 hours)
```python
# tests/integration/test_sophisticated_pipeline.py
import pytest
import time
from src.reputenet.config import Config
from src.reputenet.di import ServiceContainer
from src.reputenet.schema import ReputationInput

class TestSophisticatedPipeline:
    """Integration tests for sophisticated agent pipeline."""

    def setup_method(self):
        """Setup test environment."""
        self.config = Config(mock_mode=True, environment="test")
        self.container = ServiceContainer(self.config)
        self.graph = self.container.get_reputation_graph()

    def test_institutional_analysis_demo(self):
        """Test institutional treasury analysis scenario."""

        # Create institutional profile input
        institutional_address = "0x1234567890123456789012345678901234567890"

        input_data = ReputationInput(
            targets=[institutional_address],
            lookback_days=90,
            max_txs=2000
        )

        start_time = time.time()

        # Execute analysis
        reports = self.graph.analyze_reputation(input_data)

        duration = time.time() - start_time

        # Validate performance
        assert duration < 60, f"Analysis took {duration:.2f}s, expected <60s"

        # Validate results
        assert institutional_address in reports
        report = reports[institutional_address]

        # Check institutional signals
        assert "institutional_probability" in report
        assert report["institutional_probability"] > 0.7, "Should detect institutional behavior"

        # Check sophistication metrics
        assert "defi_sophistication_score" in report
        assert report["defi_sophistication_score"] > 0.6, "Should detect sophisticated DeFi usage"

        # Check risk assessment
        assert "risk_score" in report
        assert 0 <= report["risk_score"] <= 100, "Risk score should be in valid range"

    def test_threat_detection_demo(self):
        """Test advanced threat detection scenario."""

        # Create suspicious address input
        suspicious_address = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"

        input_data = ReputationInput(
            targets=[suspicious_address],
            lookback_days=30,
            max_txs=1000
        )

        # Execute analysis
        reports = self.graph.analyze_reputation(input_data)

        # Validate threat detection
        assert suspicious_address in reports
        report = reports[suspicious_address]

        # Should detect high risk
        assert report.get("risk_score", 0) > 60, "Should detect high risk behavior"

        # Should have detailed evidence
        assert "evidence" in report
        assert len(report["evidence"]) > 0, "Should provide risk evidence"

        # Check for advanced threat detection
        risk_components = report.get("components", {})
        assert "mev_abuse" in risk_components or "governance_risks" in risk_components

    def test_sybil_cluster_detection(self):
        """Test sybil cluster detection scenario."""

        # Create cluster of related addresses
        cluster_addresses = [
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333"
        ]

        input_data = ReputationInput(
            targets=cluster_addresses,
            lookback_days=60,
            max_txs=1500
        )

        # Execute analysis
        reports = self.graph.analyze_reputation(input_data)

        # Validate cluster detection
        sybil_scores = [reports[addr].get("sybil_score", 0) for addr in cluster_addresses]

        # At least one address should have elevated sybil score
        assert max(sybil_scores) > 40, "Should detect coordination patterns"

        # Check for coordination evidence
        for addr in cluster_addresses:
            report = reports[addr]
            assert "sybil_signals" in report or "coordination_evidence" in report

    def test_multi_address_performance(self):
        """Test performance with multiple addresses."""

        # Create multiple address input
        addresses = [f"0x{i:040x}" for i in range(10)]

        input_data = ReputationInput(
            targets=addresses,
            lookback_days=90,
            max_txs=2000
        )

        start_time = time.time()

        # Execute analysis
        reports = self.graph.analyze_reputation(input_data)

        duration = time.time() - start_time

        # Performance validation
        assert duration < 120, f"Multi-address analysis took {duration:.2f}s, expected <120s"

        # Result validation
        assert len(reports) == 10, "Should analyze all addresses"

        for addr in addresses:
            assert addr in reports, f"Missing report for {addr}"
            report = reports[addr]
            assert "reputation_score" in report
            assert 0 <= report["reputation_score"] <= 100

    def test_feature_extraction_sophistication(self):
        """Test sophisticated feature extraction capabilities."""

        test_address = "0xfeaturetest123456789012345678901234567890"

        input_data = ReputationInput(
            targets=[test_address],
            lookback_days=90,
            max_txs=2000
        )

        reports = self.graph.analyze_reputation(input_data)
        report = reports[test_address]

        # Check for sophisticated features
        expected_features = [
            "defi_sophistication_score",
            "capital_efficiency_metrics",
            "risk_management_patterns",
            "institutional_probability",
            "coordination_probability"
        ]

        for feature in expected_features:
            assert feature in report, f"Missing sophisticated feature: {feature}"

    def test_explainability_and_evidence(self):
        """Test explainability and evidence compilation."""

        test_address = "0xexplaintest123456789012345678901234567890"

        input_data = ReputationInput(
            targets=[test_address],
            lookback_days=60,
            max_txs=1000
        )

        reports = self.graph.analyze_reputation(input_data)
        report = reports[test_address]

        # Check for explainability components
        assert "methodology_explanation" in report or "evidence" in report
        assert "confidence_intervals" in report or "uncertainty_analysis" in report

        # Risk evidence should be detailed
        if report.get("risk_score", 0) > 30:
            assert "risk_evidence" in report or "evidence" in report
            evidence = report.get("evidence", report.get("risk_evidence", []))
            assert len(evidence) > 0, "High risk should have evidence"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

#### 4.2 Performance Optimization (2 hours)
```python
# src/reputenet/utils/performance.py
import cProfile
import pstats
import time
from functools import wraps
from typing import Dict, Any, Callable
import structlog

logger = structlog.get_logger()

class PerformanceOptimizer:
    """Performance optimization and monitoring for sophisticated agents."""

    def __init__(self):
        self.performance_cache = {}
        self.bottleneck_analysis = {}

    def profile_agent_performance(self, agent_func: Callable) -> Callable:
        """Decorator to profile agent performance."""

        @wraps(agent_func)
        def wrapper(*args, **kwargs):
            agent_name = agent_func.__name__

            # Start profiling
            profiler = cProfile.Profile()
            start_time = time.time()

            profiler.enable()
            result = agent_func(*args, **kwargs)
            profiler.disable()

            duration = time.time() - start_time

            # Analyze performance
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')

            # Log performance metrics
            logger.info("Agent performance analysis",
                       agent=agent_name,
                       duration=duration,
                       function_calls=stats.total_calls)

            # Cache performance data
            self.performance_cache[agent_name] = {
                "duration": duration,
                "function_calls": stats.total_calls,
                "timestamp": time.time()
            }

            return result

        return wrapper

    def optimize_feature_extraction(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Optimized feature extraction with caching and parallel processing."""

        # Check cache first
        cache_key = self._generate_feature_cache_key(transactions)
        if cache_key in self.performance_cache:
            logger.info("Feature cache hit", cache_key=cache_key)
            return self.performance_cache[cache_key]

        # Parallel feature extraction for independent features
        with ThreadPoolExecutor(max_workers=4) as executor:
            feature_futures = {
                "temporal": executor.submit(self._extract_temporal_features, transactions),
                "economic": executor.submit(self._extract_economic_features, transactions),
                "network": executor.submit(self._extract_network_features, transactions),
                "behavioral": executor.submit(self._extract_behavioral_features, transactions)
            }

            # Collect results
            features = {}
            for category, future in feature_futures.items():
                try:
                    category_features = future.result(timeout=30)
                    features.update(category_features)
                except Exception as e:
                    logger.error("Feature extraction failed", category=category, error=str(e))

        # Cache results
        self.performance_cache[cache_key] = features

        return features

    def optimize_risk_analysis(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Optimized risk analysis with incremental updates."""

        # Use incremental analysis for large transaction sets
        if len(transactions) > 1000:
            return self._incremental_risk_analysis(transactions)
        else:
            return self._full_risk_analysis(transactions)

    def _incremental_risk_analysis(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Incremental risk analysis for large datasets."""

        # Process transactions in batches
        batch_size = 500
        risk_results = []

        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            batch_risk = self._analyze_risk_batch(batch)
            risk_results.append(batch_risk)

        # Aggregate batch results
        return self._aggregate_risk_results(risk_results)

    def benchmark_pipeline_performance(self, input_data: ReputationInput) -> Dict[str, Any]:
        """Benchmark complete pipeline performance."""

        benchmarks = {}

        # Test different scenarios
        scenarios = [
            ("single_address", ReputationInput(targets=input_data.targets[:1])),
            ("multiple_addresses", input_data),
            ("high_volume", ReputationInput(targets=input_data.targets, max_txs=5000))
        ]

        for scenario_name, scenario_input in scenarios:
            start_time = time.time()

            try:
                # Run pipeline
                graph = self._get_test_graph()
                reports = graph.analyze_reputation(scenario_input)

                duration = time.time() - start_time

                benchmarks[scenario_name] = {
                    "duration": duration,
                    "addresses_analyzed": len(reports),
                    "avg_time_per_address": duration / len(reports) if reports else 0,
                    "status": "success"
                }

            except Exception as e:
                duration = time.time() - start_time
                benchmarks[scenario_name] = {
                    "duration": duration,
                    "status": "failed",
                    "error": str(e)
                }

        return benchmarks
```

---

## Demo Preparation and Validation (2 hours)

### Demo Scenario Setup
```python
# scripts/prepare_demo_scenarios.py
from src.reputenet.config import load_config
from src.reputenet.di import ServiceContainer
from src.reputenet.schema import ReputationInput

def prepare_institutional_demo():
    """Prepare institutional treasury analysis demo."""

    config = load_config()
    container = ServiceContainer(config)
    graph = container.get_reputation_graph()

    # Institutional addresses for demo
    institutional_addresses = [
        "0x1234567890123456789012345678901234567890",  # Large DeFi treasury
        "0x2345678901234567890123456789012345678901",  # Sophisticated yield farmer
        "0x3456789012345678901234567890123456789012"   # Professional trading operation
    ]

    print("🏛️  Institutional Analysis Demo")
    print("=" * 50)

    for addr in institutional_addresses:
        input_data = ReputationInput(targets=[addr], lookback_days=90)
        reports = graph.analyze_reputation(input_data)

        report = reports[addr]

        print(f"\n📊 Address: {addr[:10]}...")
        print(f"🏆 Reputation Score: {report.get('reputation_score', 'N/A')}/100")
        print(f"🏢 Institutional Probability: {report.get('institutional_probability', 0):.2f}")
        print(f"🧠 DeFi Sophistication: {report.get('defi_sophistication_score', 0):.2f}")
        print(f"⚠️  Risk Score: {report.get('risk_score', 0)}/100")

        if report.get('institutional_probability', 0) > 0.7:
            print("✅ Detected: Institutional Treasury Management")

        if report.get('defi_sophistication_score', 0) > 0.8:
            print("✅ Detected: Advanced DeFi Strategies")

def prepare_threat_detection_demo():
    """Prepare advanced threat detection demo."""

    config = load_config()
    container = ServiceContainer(config)
    graph = container.get_reputation_graph()

    # Threat addresses for demo
    threat_addresses = [
        "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",  # MEV abuse
        "0xbcdefabcdefabcdefabcdefabcdefabcdefabcde",  # Flash loan attacks
        "0xcdefabcdefabcdefabcdefabcdefabcdefabcdef"   # Money laundering
    ]

    print("\n🚨 Advanced Threat Detection Demo")
    print("=" * 50)

    for addr in threat_addresses:
        input_data = ReputationInput(targets=[addr], lookback_days=60)
        reports = graph.analyze_reputation(input_data)

        report = reports[addr]

        print(f"\n🎯 Address: {addr[:10]}...")
        print(f"⚠️  Risk Score: {report.get('risk_score', 0)}/100")
        print(f"🤖 Sybil Score: {report.get('sybil_score', 0)}/100")

        # Show detected threats
        evidence = report.get('evidence', [])
        for threat in evidence[:3]:  # Show top 3 threats
            print(f"🔍 Detected: {threat}")

        if report.get('risk_score', 0) > 70:
            print("❌ HIGH RISK: Advanced threat patterns detected")
        elif report.get('risk_score', 0) > 40:
            print("⚠️  MEDIUM RISK: Suspicious patterns detected")

if __name__ == "__main__":
    prepare_institutional_demo()
    prepare_threat_detection_demo()

    print("\n🎉 Demo scenarios prepared successfully!")
    print("📝 Run: python scripts/prepare_demo_scenarios.py")
```

---

## Success Validation

### ✅ Phase 3 Complete Checklist

**Technical Excellence:**
- [ ] All 6 agents implement sophisticated domain logic
- [ ] Multi-level detection (foundation, advanced, expert) functional
- [ ] Performance targets met (<60s comprehensive analysis)
- [ ] Explainable methodology throughout
- [ ] Production-ready architecture patterns

**Stakeholder Value:**
- [ ] Two compelling demo scenarios ready for presentation
- [ ] Clear competitive differentiation demonstrated
- [ ] Business value articulated for multiple use cases
- [ ] Technical sophistication impresses technical stakeholders

**Integration Quality:**
- [ ] Complete pipeline executes without errors
- [ ] Error handling maintains stability
- [ ] Comprehensive logging enables monitoring
- [ ] Type safety maintained throughout

---

## Next Phase Ready

**Phase 4: Production Readiness** can begin with sophisticated agent intelligence and compelling stakeholder demonstrations.

**🚀 Result: Advanced blockchain intelligence platform ready for high-impact stakeholder engagement with clear competitive advantages and production-ready sophistication.**