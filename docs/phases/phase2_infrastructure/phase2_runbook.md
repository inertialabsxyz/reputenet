# Phase 2: Core Infrastructure - Implementation Runbook

**Target:** Mock-first infrastructure with LangGraph orchestration
**Prerequisites:** Phase 1 complete (project structure, dependencies, configuration)
**Duration:** 2-3 days (16-20 hours total)
**Context:** Single developer, prototype timeline, realistic mock data

---

## Implementation Schedule

### Day 1: Mock Data Foundation (6-8 hours)
- **Morning:** Mock data generation system (4 hours)
- **Afternoon:** Tool adapters with mock backends (4 hours)

### Day 2: LangGraph Pipeline (6-8 hours)
- **Morning:** Data schemas and state management (3 hours)
- **Afternoon:** LangGraph orchestration framework (4 hours)

### Day 3: Integration & Testing (4-6 hours)
- **Morning:** Dependency injection and service layer (3 hours)
- **Afternoon:** Integration testing and validation (3 hours)

---

## Step-by-Step Implementation

### Step 1: Mock Data Generation System (4 hours)

#### 1.1 Create Mock Data Infrastructure (1 hour)
```bash
# Create mock data directories
mkdir -p src/reputenet/mock_data/{generators,templates,cache}
mkdir -p mock_data/{addresses,transactions,protocols}

# Create mock data modules
touch src/reputenet/mock_data/__init__.py
touch src/reputenet/mock_data/generators.py
touch src/reputenet/mock_data/providers.py
touch src/reputenet/mock_data/templates.py
```

#### 1.2 Implement Address Profile Generator (2 hours)
```python
# src/reputenet/mock_data/generators.py
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from faker import Faker
from faker.providers import BaseProvider

class EthereumProvider(BaseProvider):
    """Custom Faker provider for Ethereum data."""

    def eth_address(self) -> str:
        """Generate valid Ethereum address."""
        return "0x" + self.generator.hexify(text="^" * 40, upper=False)

    def tx_hash(self) -> str:
        """Generate valid transaction hash."""
        return "0x" + self.generator.hexify(text="^" * 64, upper=False)

    def block_number(self, days_ago: int = 0) -> int:
        """Generate realistic block number."""
        current_block = 18_500_000
        blocks_per_day = 7200
        return current_block - (days_ago * blocks_per_day) + random.randint(-1000, 1000)

class MockAddressGenerator:
    """Generate realistic address profiles."""

    def __init__(self):
        self.faker = Faker()
        self.faker.add_provider(EthereumProvider)

    def generate_profile(self, profile_type: str) -> Dict[str, Any]:
        """Generate address profile based on type."""
        profiles = {
            "normal_user": self._generate_normal_user,
            "whale": self._generate_whale,
            "suspicious": self._generate_suspicious,
            "bot": self._generate_bot
        }

        generator = profiles.get(profile_type, self._generate_normal_user)
        return generator()

    def _generate_normal_user(self) -> Dict[str, Any]:
        """Generate normal user profile."""
        return {
            "address": self.faker.eth_address(),
            "profile_type": "normal_user",
            "balance_eth": round(random.uniform(0.1, 50), 3),
            "transaction_frequency": random.randint(1, 10),  # per day
            "protocols": random.sample(["uniswap", "opensea", "aave"], k=random.randint(1, 2)),
            "risk_level": "low",
            "created_days_ago": random.randint(30, 730)
        }

    def _generate_whale(self) -> Dict[str, Any]:
        """Generate whale profile."""
        return {
            "address": self.faker.eth_address(),
            "profile_type": "whale",
            "balance_eth": round(random.uniform(100, 10000), 2),
            "transaction_frequency": random.randint(20, 100),
            "protocols": ["uniswap", "aave", "compound", "curve", "opensea"],
            "risk_level": "low",
            "created_days_ago": random.randint(365, 1460)
        }

    def _generate_suspicious(self) -> Dict[str, Any]:
        """Generate suspicious address profile."""
        return {
            "address": self.faker.eth_address(),
            "profile_type": "suspicious",
            "balance_eth": round(random.uniform(0.01, 5), 3),
            "transaction_frequency": random.randint(50, 200),  # Burst activity
            "protocols": ["tornado_cash", "bridges"],
            "risk_level": "high",
            "created_days_ago": random.randint(1, 90)
        }

    def generate_transaction_history(self, profile: Dict[str, Any], days: int = 90) -> List[Dict[str, Any]]:
        """Generate realistic transaction history."""
        transactions = []
        current_date = datetime.now()

        for day in range(days):
            date = current_date - timedelta(days=day)

            # Generate daily transactions based on profile
            daily_count = self._get_daily_tx_count(profile, date)

            for _ in range(daily_count):
                tx = self._generate_transaction(profile, date)
                transactions.append(tx)

        return sorted(transactions, key=lambda x: x["timestamp"])

    def _generate_transaction(self, profile: Dict[str, Any], date: datetime) -> Dict[str, Any]:
        """Generate single realistic transaction."""
        protocols = profile["protocols"]
        selected_protocol = random.choice(protocols) if protocols else "transfer"

        return {
            "hash": self.faker.tx_hash(),
            "from": profile["address"],
            "to": self._get_protocol_contract(selected_protocol),
            "value": self._get_transaction_value(profile, selected_protocol),
            "gas_used": self._get_gas_usage(selected_protocol),
            "gas_price": random.randint(20, 200) * 10**9,
            "timestamp": date.timestamp(),
            "block_number": self.faker.block_number(days_ago=(datetime.now() - date).days),
            "status": 1 if random.random() > 0.02 else 0,  # 2% failure rate
            "protocol": selected_protocol
        }

    def _get_protocol_contract(self, protocol: str) -> str:
        """Get contract address for protocol."""
        contracts = {
            "uniswap": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "aave": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            "tornado_cash": "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",
            "opensea": "0x00000000006c3852cbEf3e08E8dF289169EdE581",
            "transfer": self.faker.eth_address()
        }
        return contracts.get(protocol, self.faker.eth_address())
```

#### 1.3 Create Mock Data Provider (1 hour)
```python
# src/reputenet/mock_data/providers.py
import diskcache
from typing import Dict, List, Any, Optional
from .generators import MockAddressGenerator

class MockDataProvider:
    """Provide cached mock blockchain data."""

    def __init__(self, cache_dir: str = "./.cache/mock_data"):
        self.cache = diskcache.Cache(cache_dir)
        self.generator = MockAddressGenerator()

    def get_address_data(self, address: str, lookback_days: int = 90, max_txs: int = 2000) -> Dict[str, Any]:
        """Get comprehensive address data."""
        cache_key = f"address:{address}:{lookback_days}:{max_txs}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate profile based on address characteristics
        profile_type = self._infer_profile_type(address)
        profile = self.generator.generate_profile(profile_type)
        profile["address"] = address  # Use provided address

        # Generate transaction history
        transactions = self.generator.generate_transaction_history(profile, lookback_days)
        transactions = transactions[:max_txs]  # Limit transaction count

        # Generate logs and token data
        logs = self._generate_logs(transactions)
        tokens = self._generate_token_interactions(transactions)

        result = {
            "address": address,
            "profile": profile,
            "transactions": transactions,
            "logs": logs,
            "tokens": tokens,
            "first_seen": min(tx["timestamp"] for tx in transactions) if transactions else None,
            "last_seen": max(tx["timestamp"] for tx in transactions) if transactions else None
        }

        # Cache result
        self.cache[cache_key] = result
        return result

    def _infer_profile_type(self, address: str) -> str:
        """Infer profile type from address characteristics."""
        # Simple heuristic based on address
        if address.endswith(("0", "1", "2")):
            return "normal_user"
        elif address.endswith(("3", "4")):
            return "whale"
        elif address.endswith(("5", "6")):
            return "bot"
        else:
            return "suspicious"
```

### Step 2: Tool Adapters with Mock Implementation (4 hours)

#### 2.1 Create Base Tool Interface (30 minutes)
```python
# src/reputenet/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseProviderTool(ABC):
    """Base interface for external API tools."""

    @abstractmethod
    def get_transaction_history(self, address: str, **kwargs) -> List[Dict[str, Any]]:
        """Get transaction history for address."""
        pass

    @abstractmethod
    def get_logs(self, address: str, **kwargs) -> List[Dict[str, Any]]:
        """Get event logs for address."""
        pass
```

#### 2.2 Implement Mock Tool Adapters (3 hours)
```python
# src/reputenet/tools/eth_provider.py
from typing import Dict, List, Any, Optional
import structlog
from .base import BaseProviderTool
from ..mock_data.providers import MockDataProvider

logger = structlog.get_logger()

class EthProviderTool(BaseProviderTool):
    """Ethereum RPC provider with mock backend."""

    def __init__(self, mock_mode: bool = True, rpc_url: Optional[str] = None):
        self.mock_mode = mock_mode
        self.rpc_url = rpc_url

        if mock_mode:
            self.provider = MockDataProvider()
        else:
            # Real implementation would go here
            raise NotImplementedError("Real API integration not implemented")

    def get_normalized_history(self, address: str, lookback_days: int, max_txs: int) -> Dict[str, Any]:
        """Get normalized transaction history."""
        logger.info("Fetching transaction history",
                   address=address,
                   lookback_days=lookback_days,
                   max_txs=max_txs,
                   mock_mode=self.mock_mode)

        if self.mock_mode:
            data = self.provider.get_address_data(address, lookback_days, max_txs)
            return {
                "txs": data["transactions"],
                "logs": data["logs"],
                "tokens": data["tokens"]
            }

    def get_logs(self, address: str, from_block: Optional[int] = None, to_block: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get event logs for address."""
        if self.mock_mode:
            data = self.provider.get_address_data(address)
            return data["logs"]

# src/reputenet/tools/etherscan.py
class EtherscanTool(BaseProviderTool):
    """Etherscan API with mock backend."""

    def __init__(self, mock_mode: bool = True, api_key: Optional[str] = None):
        self.mock_mode = mock_mode
        self.api_key = api_key

        if mock_mode:
            self.provider = MockDataProvider()

    def get_transaction_history(self, address: str, **kwargs) -> List[Dict[str, Any]]:
        """Get transaction history from Etherscan."""
        if self.mock_mode:
            data = self.provider.get_address_data(address)
            return data["transactions"]

# src/reputenet/tools/defillama.py
class DefiLlamaTool:
    """DefiLlama API with mock backend."""

    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode

    def get_protocol_metadata(self, contract: str) -> Dict[str, Any]:
        """Get protocol metadata."""
        if self.mock_mode:
            # Return mock protocol data
            mock_protocols = {
                "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": {
                    "name": "Uniswap V3",
                    "category": "DEX",
                    "tvl": 1500000000,
                    "risk_score": 0.1
                },
                "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9": {
                    "name": "AAVE V2",
                    "category": "Lending",
                    "tvl": 5000000000,
                    "risk_score": 0.05
                }
            }

            return mock_protocols.get(contract, {
                "name": "Unknown Protocol",
                "category": "Other",
                "tvl": 0,
                "risk_score": 0.5
            })
```

#### 2.3 Create Label Registry (30 minutes)
```python
# src/reputenet/tools/labels.py
from typing import Optional, Dict, Any

class LabelRegistry:
    """Address labeling service with mock data."""

    def __init__(self, mock_mode: bool = True):
        self.mock_mode = mock_mode

        if mock_mode:
            self.labels = self._load_mock_labels()

    def _load_mock_labels(self) -> Dict[str, Dict[str, Any]]:
        """Load mock address labels."""
        return {
            "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936": {
                "label": "Tornado Cash",
                "category": "mixer",
                "risk_level": "high"
            },
            "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": {
                "label": "Uniswap V3 Router",
                "category": "dex",
                "risk_level": "low"
            }
        }

    def is_flagged_contract(self, address: str) -> bool:
        """Check if contract is flagged as risky."""
        label_data = self.labels.get(address, {})
        return label_data.get("risk_level") == "high"

    def get_label(self, address: str) -> Optional[str]:
        """Get label for address."""
        label_data = self.labels.get(address, {})
        return label_data.get("label")
```

### Step 3: Data Schemas and State Management (3 hours)

#### 3.1 Complete GraphState Implementation (1 hour)
```python
# src/reputenet/schema.py (extend existing)
from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

class Transaction(BaseModel):
    """Individual transaction data."""
    hash: str
    from_address: str = Field(alias="from")
    to_address: str = Field(alias="to")
    value: float
    gas_used: int
    gas_price: int
    timestamp: float
    block_number: int
    status: int
    protocol: Optional[str] = None

class AddressFeatures(BaseModel):
    """Extracted features for an address."""
    account_age_days: float
    transaction_frequency: float
    protocol_diversity: float
    counterparty_uniqueness: float
    value_patterns: Dict[str, Any]
    gas_patterns: Dict[str, Any]
    time_patterns: Dict[str, Any]

class RiskOutput(BaseModel):
    """Risk assessment output."""
    risk_score: int = Field(ge=0, le=100)
    evidence: List[str]
    components: Dict[str, Any]

class SybilOutput(BaseModel):
    """Sybil detection output."""
    sybil_score: int = Field(ge=0, le=100)
    signals: List[str]
    cluster_id: Optional[str] = None

class ReputationComponents(BaseModel):
    """Reputation score components."""
    activity_quality: float = Field(ge=0, le=1)
    protocol_diversity: float = Field(ge=0, le=1)
    counterparty_uniqueness: float = Field(ge=0, le=1)
    risk: float = Field(ge=0, le=1)
    sybil: float = Field(ge=0, le=1)

class GraphState(TypedDict):
    """LangGraph state for reputation analysis."""
    params: ReputationInput
    raw: Dict[str, Any]
    features: Dict[str, AddressFeatures]
    risk: Dict[str, RiskOutput]
    sybil: Dict[str, SybilOutput]
    reputation: Dict[str, ReputationOutput]
    reports: Dict[str, Dict[str, Any]]
```

#### 3.2 Create State Validation (1 hour)
```python
# src/reputenet/utils/validation.py
from typing import Dict, Any, List
import structlog
from ..schema import GraphState, ReputationInput

logger = structlog.get_logger()

class StateValidator:
    """Validate state transitions in LangGraph."""

    def validate_harvest_output(self, state: GraphState) -> bool:
        """Validate DataHarvester output."""
        try:
            raw_data = state["raw"]

            for address in state["params"]["targets"]:
                if address not in raw_data:
                    logger.error("Missing address in raw data", address=address)
                    return False

                addr_data = raw_data[address]
                required_keys = ["transactions", "logs", "tokens"]

                if not all(key in addr_data for key in required_keys):
                    logger.error("Missing required keys in address data",
                               address=address,
                               keys=list(addr_data.keys()))
                    return False

            return True

        except Exception as e:
            logger.error("State validation failed", error=str(e))
            return False

    def validate_features_output(self, state: GraphState) -> bool:
        """Validate AddressProfiler output."""
        try:
            features = state["features"]

            for address in state["params"]["targets"]:
                if address not in features:
                    return False

                addr_features = features[address]
                required_metrics = [
                    "account_age_days",
                    "transaction_frequency",
                    "protocol_diversity",
                    "counterparty_uniqueness"
                ]

                if not all(metric in addr_features for metric in required_metrics):
                    return False

            return True

        except Exception as e:
            logger.error("Features validation failed", error=str(e))
            return False
```

#### 3.3 Create State Utilities (1 hour)
```python
# src/reputenet/utils/state.py
from typing import Dict, Any
from copy import deepcopy
from ..schema import GraphState

def create_initial_state(params: ReputationInput) -> GraphState:
    """Create initial graph state."""
    return GraphState(
        params=params,
        raw={},
        features={},
        risk={},
        sybil={},
        reputation={},
        reports={}
    )

def update_state_safely(state: GraphState, updates: Dict[str, Any]) -> GraphState:
    """Update state with error handling."""
    try:
        new_state = deepcopy(state)
        new_state.update(updates)
        return new_state
    except Exception as e:
        logger.error("State update failed", error=str(e))
        return state

def extract_state_summary(state: GraphState) -> Dict[str, Any]:
    """Extract summary information from state."""
    return {
        "addresses_processed": len(state.get("raw", {})),
        "features_extracted": len(state.get("features", {})),
        "risk_assessed": len(state.get("risk", {})),
        "sybil_analyzed": len(state.get("sybil", {})),
        "reputation_computed": len(state.get("reputation", {})),
        "reports_generated": len(state.get("reports", {}))
    }
```

### Step 4: LangGraph Orchestration Framework (4 hours)

#### 4.1 Create Graph Builder (2 hours)
```python
# src/reputenet/graph.py
from typing import Dict, Any
import time
import structlog
from langgraph.graph import StateGraph, END

from .schema import GraphState, ReputationInput
from .tools.eth_provider import EthProviderTool
from .tools.etherscan import EtherscanTool
from .tools.defillama import DefiLlamaTool
from .tools.labels import LabelRegistry
from .utils.state import create_initial_state, extract_state_summary
from .utils.validation import StateValidator

logger = structlog.get_logger()

class ReputationGraph:
    """LangGraph orchestration for reputation analysis."""

    def __init__(self,
                 eth_provider: EthProviderTool,
                 etherscan: EtherscanTool,
                 defillama: DefiLlamaTool,
                 labels: LabelRegistry):
        self.eth_provider = eth_provider
        self.etherscan = etherscan
        self.defillama = defillama
        self.labels = labels
        self.validator = StateValidator()

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the reputation analysis graph."""
        workflow = StateGraph(GraphState)

        # Add processing nodes
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
        logger.info("Starting reputation analysis",
                   targets=input_data.targets,
                   lookback_days=input_data.lookback_days)

        start_time = time.time()

        # Initialize state
        initial_state = create_initial_state(input_data)

        try:
            # Execute graph
            final_state = self.graph.invoke(initial_state)

            duration = time.time() - start_time
            summary = extract_state_summary(final_state)

            logger.info("Reputation analysis complete",
                       duration=duration,
                       **summary)

            return final_state["reports"]

        except Exception as e:
            logger.error("Reputation analysis failed", error=str(e))
            raise

    def _node_data_harvest(self, state: GraphState) -> GraphState:
        """Node: Harvest blockchain data."""
        node_start = time.time()
        logger.info("Executing DataHarvester node")

        raw_data = {}

        try:
            for address in state["params"]["targets"]:
                logger.info("Harvesting data for address", address=address)

                # Get data from mock provider
                addr_data = self.eth_provider.get_normalized_history(
                    address=address,
                    lookback_days=state["params"]["lookback_days"],
                    max_txs=state["params"]["max_txs"]
                )

                raw_data[address] = addr_data

            state["raw"] = raw_data

            # Validate output
            if not self.validator.validate_harvest_output(state):
                logger.warning("Harvest validation failed, continuing anyway")

            duration = time.time() - node_start
            logger.info("DataHarvester node complete",
                       duration=duration,
                       addresses=len(raw_data))

        except Exception as e:
            logger.error("DataHarvester node failed", error=str(e))
            # Continue with empty data for prototype
            state["raw"] = {}

        return state
```

#### 4.2 Implement Core Analysis Nodes (2 hours)
```python
# Continue in graph.py

    def _node_address_profile(self, state: GraphState) -> GraphState:
        """Node: Extract address features."""
        node_start = time.time()
        logger.info("Executing AddressProfiler node")

        features = {}

        try:
            for address, raw_data in state["raw"].items():
                txs = raw_data.get("txs", [])

                if not txs:
                    logger.warning("No transactions for address", address=address)
                    continue

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

            duration = time.time() - node_start
            logger.info("AddressProfiler node complete",
                       duration=duration,
                       addresses=len(features))

        except Exception as e:
            logger.error("AddressProfiler node failed", error=str(e))
            state["features"] = {}

        return state

    def _node_risk_score(self, state: GraphState) -> GraphState:
        """Node: Calculate risk scores."""
        node_start = time.time()
        logger.info("Executing RiskScorer node")

        risk_data = {}

        try:
            for address, raw_data in state["raw"].items():
                txs = raw_data.get("txs", [])

                risk_score = 0
                evidence = []

                # Check for mixer interactions
                mixer_count = sum(1 for tx in txs if self._is_mixer_interaction(tx))
                if mixer_count > 0:
                    risk_score += 30
                    evidence.append(f"mixer_interactions:{mixer_count}")

                # Check for suspicious contracts
                suspicious_count = sum(1 for tx in txs if self.labels.is_flagged_contract(tx.get("to", "")))
                if suspicious_count > 0:
                    risk_score += 20
                    evidence.append(f"flagged_contracts:{suspicious_count}")

                # Check failure rate
                failure_rate = sum(1 for tx in txs if tx.get("status") == 0) / max(len(txs), 1)
                if failure_rate > 0.1:
                    risk_score += 15
                    evidence.append(f"high_failure_rate:{failure_rate:.2f}")

                risk_data[address] = {
                    "risk_score": min(risk_score, 100),
                    "evidence": evidence,
                    "components": {
                        "mixer_interactions": mixer_count,
                        "flagged_contracts": suspicious_count,
                        "failure_rate": failure_rate
                    }
                }

            state["risk"] = risk_data

            duration = time.time() - node_start
            logger.info("RiskScorer node complete",
                       duration=duration,
                       addresses=len(risk_data))

        except Exception as e:
            logger.error("RiskScorer node failed", error=str(e))
            state["risk"] = {}

        return state

    def _calculate_protocol_diversity(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate protocol interaction diversity."""
        protocols = set()
        for tx in transactions:
            if tx.get("protocol"):
                protocols.add(tx["protocol"])

        # Normalize by typical protocol count
        return min(len(protocols) / 5.0, 1.0)

    def _is_mixer_interaction(self, tx: Dict[str, Any]) -> bool:
        """Check if transaction is mixer interaction."""
        mixer_contracts = {
            "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",  # Tornado Cash
        }
        return tx.get("to") in mixer_contracts
```

### Step 5: Dependency Injection and Integration (3 hours)

#### 5.1 Create Service Container (1 hour)
```python
# src/reputenet/di.py
from typing import Dict, Any, Optional
from .config import Config
from .tools.eth_provider import EthProviderTool
from .tools.etherscan import EtherscanTool
from .tools.defillama import DefiLlamaTool
from .tools.labels import LabelRegistry
from .graph import ReputationGraph

class ServiceContainer:
    """Dependency injection container."""

    def __init__(self, config: Config):
        self.config = config
        self._services: Dict[str, Any] = {}

    def get_eth_provider(self) -> EthProviderTool:
        """Get Ethereum provider tool."""
        if "eth_provider" not in self._services:
            self._services["eth_provider"] = EthProviderTool(
                mock_mode=self.config.mock_mode,
                rpc_url=self.config.rpc_url_mainnet
            )
        return self._services["eth_provider"]

    def get_etherscan(self) -> EtherscanTool:
        """Get Etherscan tool."""
        if "etherscan" not in self._services:
            self._services["etherscan"] = EtherscanTool(
                mock_mode=self.config.mock_mode,
                api_key=self.config.etherscan_api_key
            )
        return self._services["etherscan"]

    def get_defillama(self) -> DefiLlamaTool:
        """Get DefiLlama tool."""
        if "defillama" not in self._services:
            self._services["defillama"] = DefiLlamaTool(
                mock_mode=self.config.mock_mode
            )
        return self._services["defillama"]

    def get_labels(self) -> LabelRegistry:
        """Get label registry."""
        if "labels" not in self._services:
            self._services["labels"] = LabelRegistry(
                mock_mode=self.config.mock_mode
            )
        return self._services["labels"]

    def get_reputation_graph(self) -> ReputationGraph:
        """Get configured reputation graph."""
        if "reputation_graph" not in self._services:
            self._services["reputation_graph"] = ReputationGraph(
                eth_provider=self.get_eth_provider(),
                etherscan=self.get_etherscan(),
                defillama=self.get_defillama(),
                labels=self.get_labels()
            )
        return self._services["reputation_graph"]
```

#### 5.2 Update CLI and API Integration (1 hour)
```python
# Update src/reputenet/cli.py
import sys
import json
from .config import load_config
from .di import ServiceContainer
from .schema import ReputationInput

def main():
    """Main CLI entry point."""
    config = load_config()
    container = ServiceContainer(config)

    if len(sys.argv) < 2:
        print("Usage: reputenet <address> [--chain-id <id>] [--lookback-days <days>]")
        return

    address = sys.argv[1]

    # Parse optional arguments
    chain_id = 1
    lookback_days = 90

    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--chain-id" and i + 1 < len(sys.argv):
            chain_id = int(sys.argv[i + 1])
        elif arg == "--lookback-days" and i + 1 < len(sys.argv):
            lookback_days = int(sys.argv[i + 1])

    # Create input
    input_data = ReputationInput(
        chain_id=chain_id,
        targets=[address],
        lookback_days=lookback_days,
        max_txs=2000
    )

    print(f"🔍 Analyzing reputation for {address}")
    print(f"📊 Mode: {'Mock' if config.mock_mode else 'Live'}")

    try:
        # Get reputation graph and analyze
        graph = container.get_reputation_graph()
        reports = graph.analyze_reputation(input_data)

        if address in reports:
            report = reports[address]
            print(f"\n✅ Analysis complete!")
            print(f"🏆 Reputation Score: {report.get('reputation_score', 'N/A')}/100")

            if config.log_level == "DEBUG":
                print(f"\n📄 Full Report:")
                print(json.dumps(report, indent=2))
        else:
            print("❌ Analysis failed - no report generated")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")

# Update src/reputenet/api.py
from fastapi import FastAPI, HTTPException
from .config import load_config
from .di import ServiceContainer
from .schema import ReputationInput, ReputationOutput

def create_app() -> FastAPI:
    """Create FastAPI application."""
    config = load_config()
    container = ServiceContainer(config)

    app = FastAPI(
        title="ReputeNet API",
        description="Multi-Agent System for On-Chain Reputation Analysis",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "mode": "mock" if config.mock_mode else "live",
            "version": "0.1.0"
        }

    @app.post("/analyze", response_model=Dict[str, ReputationOutput])
    async def analyze_reputation(input_data: ReputationInput):
        """Analyze reputation for given addresses."""
        try:
            graph = container.get_reputation_graph()
            reports = graph.analyze_reputation(input_data)

            # Convert to ReputationOutput format
            result = {}
            for address, report in reports.items():
                result[address] = ReputationOutput(**report)

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
```

#### 5.3 Create Integration Tests (1 hour)
```python
# tests/test_integration.py
import pytest
from src.reputenet.config import Config
from src.reputenet.di import ServiceContainer
from src.reputenet.schema import ReputationInput

def test_end_to_end_pipeline():
    """Test complete pipeline execution."""
    # Setup
    config = Config(mock_mode=True, environment="test")
    container = ServiceContainer(config)
    graph = container.get_reputation_graph()

    # Test input
    input_data = ReputationInput(
        targets=["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"],
        lookback_days=30,
        max_txs=1000
    )

    # Execute
    reports = graph.analyze_reputation(input_data)

    # Validate
    assert len(reports) == 1
    address = input_data.targets[0]
    assert address in reports

    report = reports[address]
    assert "reputation_score" in report
    assert 0 <= report["reputation_score"] <= 100
    assert "components" in report
    assert "flags" in report

def test_multiple_addresses():
    """Test pipeline with multiple addresses."""
    config = Config(mock_mode=True, environment="test")
    container = ServiceContainer(config)
    graph = container.get_reputation_graph()

    # Test multiple addresses
    addresses = [f"0x{i:040x}" for i in range(5)]
    input_data = ReputationInput(targets=addresses, lookback_days=30)

    reports = graph.analyze_reputation(input_data)

    assert len(reports) == 5
    for address in addresses:
        assert address in reports
        assert "reputation_score" in reports[address]

def test_performance_benchmark():
    """Test pipeline performance."""
    import time

    config = Config(mock_mode=True, environment="test")
    container = ServiceContainer(config)
    graph = container.get_reputation_graph()

    input_data = ReputationInput(
        targets=["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"],
        lookback_days=90,
        max_txs=2000
    )

    start_time = time.time()
    reports = graph.analyze_reputation(input_data)
    duration = time.time() - start_time

    # Should complete under 30 seconds for prototype
    assert duration < 30
    assert len(reports) == 1
```

---

## Validation and Testing (3 hours)

### Final Integration Test
```bash
# 1. Install and setup
uv pip install -e ".[dev]"
cp .env.template .env

# 2. Run unit tests
pytest tests/ -v

# 3. Test CLI
reputenet 0x742c4af20a2e0c8e82be16ab44d9421b1b78e569 --lookback-days 30

# 4. Test API
reputenet-api &
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"targets": ["0x742c4af20a2e0c8e82be16ab44d9421b1b78e569"], "lookback_days": 30}'

# 5. Performance test
time reputenet 0x742c4af20a2e0c8e82be16ab44d9421b1b78e569

# 6. Container test
docker build -t reputenet .
docker run -p 8000:8000 reputenet
```

---

## Success Criteria

### ✅ Phase 2 Complete When:

1. **End-to-End Pipeline** ✅ - Complete reputation analysis with mock data
2. **Realistic Mock Data** ✅ - Generated data resembles blockchain patterns
3. **LangGraph Execution** ✅ - All 6 nodes execute without errors
4. **Tool Integration** ✅ - Mock adapters provide consistent responses
5. **Type Safety** ✅ - All data flows through Pydantic validation
6. **Performance Target** ✅ - Sub-30 second execution for single address
7. **Service Integration** ✅ - Clean dependency injection and configuration
8. **API Functionality** ✅ - Both CLI and HTTP API work correctly

**Next Phase Ready:** Phase 3 (Agent Implementation) can begin with working infrastructure and realistic mock data.

**🎯 Result: Complete mock infrastructure ready for agent development with realistic blockchain data patterns and working LangGraph orchestration.**