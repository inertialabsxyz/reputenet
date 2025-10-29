# Step 1: Mock Data Generation System - Implementation Notes

**Context:** Realistic blockchain mock data generation for comprehensive prototype testing
**Approach:** Three-tier complexity system with sophisticated DeFi protocol simulation

---

## Implementation Strategy

### Mock Data Architecture
Based on design decisions, implementing:
- **Three-tier complexity system** (foundation, sophisticated, expert)
- **Realistic blockchain patterns** with proper statistical distributions
- **DeFi protocol simulation** with actual protocol mechanics
- **Configurable generation** supporting different scenarios and volumes
- **Validation framework** ensuring data quality and consistency

### File Structure
```
mock_data/
├── __init__.py
├── generators/
│   ├── __init__.py
│   ├── base.py                 # Base generator classes
│   ├── addresses.py            # Address generation with relationships
│   ├── transactions.py         # Transaction pattern generation
│   ├── blocks.py              # Block and timestamp generation
│   ├── defi_protocols.py      # DeFi protocol interaction simulation
│   ├── erc20_tokens.py        # ERC-20 token transactions
│   ├── nft_activity.py        # NFT trading and collection patterns
│   └── governance.py          # Governance participation patterns
├── scenarios/
│   ├── __init__.py
│   ├── foundation.py          # Basic blockchain activity
│   ├── sophisticated.py       # Advanced DeFi strategies
│   ├── expert.py             # Complex institutional patterns
│   ├── threat_scenarios.py    # Security threat patterns
│   └── compliance_scenarios.py # Compliance test cases
├── validation/
│   ├── __init__.py
│   ├── consistency.py         # Data consistency validation
│   ├── realism.py            # Realism scoring and validation
│   └── performance.py        # Generation performance tracking
└── fixtures/
    ├── real_patterns.json     # Real blockchain pattern templates
    ├── protocol_configs.json  # DeFi protocol configurations
    └── address_classifications.json # Known address types
```

---

## Core Generator Implementation

### Base Generator Framework

#### mock_data/generators/base.py
```python
"""Base classes for mock data generation."""

import random
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field
from faker import Faker


class ComplexityLevel(Enum):
    """Mock data complexity levels."""
    FOUNDATION = "foundation"
    SOPHISTICATED = "sophisticated"
    EXPERT = "expert"


class GenerationConfig(BaseModel):
    """Configuration for mock data generation."""
    complexity: ComplexityLevel = ComplexityLevel.FOUNDATION
    seed: Optional[int] = None
    start_date: datetime = Field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = Field(default_factory=datetime.now)

    # Volume controls
    address_count: int = 100
    transaction_count: int = 1000
    block_count: int = 500

    # Realism controls
    gas_price_volatility: float = 0.3
    transaction_value_distribution: str = "lognormal"
    temporal_clustering: float = 0.7

    # Protocol participation rates
    defi_participation_rate: float = 0.3
    nft_participation_rate: float = 0.15
    governance_participation_rate: float = 0.05


class BaseGenerator(ABC):
    """Abstract base class for all mock data generators."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.fake = Faker()

        # Set seeds for reproducibility
        if config.seed:
            random.seed(config.seed)
            np.random.seed(config.seed)
            Faker.seed(config.seed)
            secrets.SystemRandom().seed(config.seed)

    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """Generate mock data according to configuration."""
        pass

    def _generate_ethereum_address(self) -> str:
        """Generate a valid Ethereum address."""
        return "0x" + secrets.token_hex(20)

    def _generate_transaction_hash(self) -> str:
        """Generate a valid transaction hash."""
        return "0x" + secrets.token_hex(32)

    def _generate_block_hash(self) -> str:
        """Generate a valid block hash."""
        return "0x" + secrets.token_hex(32)

    def _generate_realistic_timestamp(self, base_time: Optional[datetime] = None) -> int:
        """Generate realistic timestamp with temporal clustering."""
        if base_time is None:
            base_time = self.fake.date_time_between(
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

        # Add temporal clustering - activities cluster around certain times
        if random.random() < self.config.temporal_clustering:
            # Cluster around business hours (more activity)
            hour_offset = np.random.normal(0, 4)  # ±4 hours standard deviation
            clustered_time = base_time + timedelta(hours=hour_offset)
        else:
            # Random time within the day
            hour_offset = random.uniform(-12, 12)
            clustered_time = base_time + timedelta(hours=hour_offset)

        return int(clustered_time.timestamp())

    def _generate_realistic_gas_price(self, base_price: int = 20_000_000_000) -> int:
        """Generate realistic gas price with volatility."""
        # Log-normal distribution for gas prices
        volatility = self.config.gas_price_volatility
        multiplier = np.random.lognormal(0, volatility)
        return int(base_price * multiplier)

    def _generate_realistic_value(self, median_value: float = 0.1) -> int:
        """Generate realistic transaction value in wei."""
        if self.config.transaction_value_distribution == "lognormal":
            # Log-normal distribution - many small values, few large ones
            log_value = np.random.lognormal(np.log(median_value), 1.5)
        elif self.config.transaction_value_distribution == "pareto":
            # Pareto distribution - power law for wealth distribution
            log_value = np.random.pareto(1.16) * median_value
        else:
            # Exponential distribution
            log_value = np.random.exponential(median_value)

        # Convert to wei (18 decimals)
        return int(log_value * 10**18)


@dataclass
class MockDataValidationResult:
    """Result of mock data validation."""
    is_valid: bool
    realism_score: float  # 0.0 to 1.0
    issues: List[str]
    statistics: Dict[str, Any]


class MockDataValidator:
    """Validates generated mock data for realism and consistency."""

    def __init__(self, real_patterns: Optional[Dict] = None):
        self.real_patterns = real_patterns or {}

    def validate_dataset(self, data: Dict[str, Any]) -> MockDataValidationResult:
        """Validate complete dataset for realism and consistency."""
        issues = []
        stats = {}

        # Validate addresses
        if "addresses" in data:
            address_issues, address_stats = self._validate_addresses(data["addresses"])
            issues.extend(address_issues)
            stats["addresses"] = address_stats

        # Validate transactions
        if "transactions" in data:
            tx_issues, tx_stats = self._validate_transactions(data["transactions"])
            issues.extend(tx_issues)
            stats["transactions"] = tx_stats

        # Calculate overall realism score
        realism_score = self._calculate_realism_score(stats)

        return MockDataValidationResult(
            is_valid=len(issues) == 0,
            realism_score=realism_score,
            issues=issues,
            statistics=stats
        )

    def _validate_addresses(self, addresses: List[Dict]) -> tuple[List[str], Dict]:
        """Validate address data for realism."""
        issues = []
        stats = {}

        # Check address format
        invalid_addresses = [addr for addr in addresses
                           if not addr.get("address", "").startswith("0x") or
                           len(addr.get("address", "")) != 42]
        if invalid_addresses:
            issues.append(f"Invalid address format: {len(invalid_addresses)} addresses")

        # Check balance distribution
        balances = [float(addr.get("balance", 0)) for addr in addresses]
        if balances:
            stats["balance_distribution"] = {
                "mean": np.mean(balances),
                "median": np.median(balances),
                "std": np.std(balances),
                "max": np.max(balances),
                "min": np.min(balances)
            }

            # Check for unrealistic balance distribution
            if np.std(balances) / np.mean(balances) < 0.5:
                issues.append("Balance distribution too uniform (unrealistic)")

        return issues, stats

    def _validate_transactions(self, transactions: List[Dict]) -> tuple[List[str], Dict]:
        """Validate transaction data for realism."""
        issues = []
        stats = {}

        # Check transaction hash format
        invalid_hashes = [tx for tx in transactions
                         if not tx.get("hash", "").startswith("0x") or
                         len(tx.get("hash", "")) != 66]
        if invalid_hashes:
            issues.append(f"Invalid transaction hash format: {len(invalid_hashes)} transactions")

        # Check gas price distribution
        gas_prices = [int(tx.get("gasPrice", 0)) for tx in transactions]
        if gas_prices:
            stats["gas_price_distribution"] = {
                "mean": np.mean(gas_prices),
                "median": np.median(gas_prices),
                "std": np.std(gas_prices)
            }

            # Check for unrealistic gas price patterns
            if np.std(gas_prices) == 0:
                issues.append("Gas prices are constant (unrealistic)")

        # Check timestamp ordering
        timestamps = [int(tx.get("timestamp", 0)) for tx in transactions]
        if timestamps and not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            issues.append("Transaction timestamps are not properly ordered")

        return issues, stats

    def _calculate_realism_score(self, stats: Dict) -> float:
        """Calculate overall realism score based on statistics."""
        score = 1.0

        # Penalize unrealistic patterns
        if "addresses" in stats:
            addr_stats = stats["addresses"]
            if "balance_distribution" in addr_stats:
                # Penalize if distribution is too uniform
                cv = addr_stats["balance_distribution"]["std"] / addr_stats["balance_distribution"]["mean"]
                if cv < 0.5:
                    score -= 0.2

        if "transactions" in stats:
            tx_stats = stats["transactions"]
            if "gas_price_distribution" in tx_stats:
                # Penalize if gas prices are too uniform
                if tx_stats["gas_price_distribution"]["std"] == 0:
                    score -= 0.3

        return max(0.0, score)
```

### Address Generation with Relationships

#### mock_data/generators/addresses.py
```python
"""Address generation with realistic relationships and classifications."""

import random
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from .base import BaseGenerator, GenerationConfig, ComplexityLevel


class AddressType(Enum):
    """Types of Ethereum addresses."""
    EOA = "externally_owned_account"
    CONTRACT = "smart_contract"
    EXCHANGE = "exchange"
    DEFI_PROTOCOL = "defi_protocol"
    INSTITUTIONAL = "institutional"
    WHALE = "whale"
    MEV_BOT = "mev_bot"
    GOVERNANCE = "governance_contract"


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AddressRelationship:
    """Relationship between two addresses."""
    from_address: str
    to_address: str
    relationship_type: str
    strength: float  # 0.0 to 1.0
    first_interaction: int  # timestamp
    last_interaction: int   # timestamp
    interaction_count: int


class AddressGenerator(BaseGenerator):
    """Generate realistic addresses with relationships and classifications."""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.known_protocols = self._load_protocol_addresses()
        self.address_types = {}
        self.relationships = []

    def generate(self) -> Dict[str, List]:
        """Generate comprehensive address dataset."""
        addresses = self._generate_address_pool()
        relationships = self._generate_address_relationships(addresses)
        clusters = self._identify_address_clusters(addresses, relationships)

        return {
            "addresses": addresses,
            "relationships": relationships,
            "clusters": clusters
        }

    def _generate_address_pool(self) -> List[Dict]:
        """Generate pool of addresses with different types and characteristics."""
        addresses = []

        # Determine address type distribution based on complexity
        type_distribution = self._get_address_type_distribution()

        for i in range(self.config.address_count):
            address_type = np.random.choice(
                list(type_distribution.keys()),
                p=list(type_distribution.values())
            )

            address_data = self._generate_address_by_type(address_type, i)
            addresses.append(address_data)
            self.address_types[address_data["address"]] = address_type

        return addresses

    def _get_address_type_distribution(self) -> Dict[AddressType, float]:
        """Get address type distribution based on complexity level."""
        if self.config.complexity == ComplexityLevel.FOUNDATION:
            return {
                AddressType.EOA: 0.70,
                AddressType.CONTRACT: 0.15,
                AddressType.EXCHANGE: 0.10,
                AddressType.DEFI_PROTOCOL: 0.05
            }
        elif self.config.complexity == ComplexityLevel.SOPHISTICATED:
            return {
                AddressType.EOA: 0.50,
                AddressType.CONTRACT: 0.20,
                AddressType.EXCHANGE: 0.10,
                AddressType.DEFI_PROTOCOL: 0.10,
                AddressType.INSTITUTIONAL: 0.05,
                AddressType.WHALE: 0.03,
                AddressType.MEV_BOT: 0.02
            }
        else:  # EXPERT
            return {
                AddressType.EOA: 0.40,
                AddressType.CONTRACT: 0.25,
                AddressType.EXCHANGE: 0.10,
                AddressType.DEFI_PROTOCOL: 0.10,
                AddressType.INSTITUTIONAL: 0.08,
                AddressType.WHALE: 0.03,
                AddressType.MEV_BOT: 0.02,
                AddressType.GOVERNANCE: 0.02
            }

    def _generate_address_by_type(self, address_type: AddressType, index: int) -> Dict:
        """Generate address data based on type."""
        address = self._generate_ethereum_address()
        base_data = {
            "address": address,
            "type": address_type.value,
            "first_seen": self._generate_realistic_timestamp(),
            "transaction_count": 0,
            "balance": "0",
            "is_contract": address_type != AddressType.EOA,
            "labels": [],
            "risk_score": 0.0,
            "risk_level": RiskLevel.LOW.value
        }

        # Type-specific generation
        if address_type == AddressType.EOA:
            return self._generate_eoa(base_data)
        elif address_type == AddressType.CONTRACT:
            return self._generate_contract(base_data)
        elif address_type == AddressType.EXCHANGE:
            return self._generate_exchange(base_data)
        elif address_type == AddressType.DEFI_PROTOCOL:
            return self._generate_defi_protocol(base_data)
        elif address_type == AddressType.INSTITUTIONAL:
            return self._generate_institutional(base_data)
        elif address_type == AddressType.WHALE:
            return self._generate_whale(base_data)
        elif address_type == AddressType.MEV_BOT:
            return self._generate_mev_bot(base_data)
        elif address_type == AddressType.GOVERNANCE:
            return self._generate_governance_contract(base_data)
        else:
            return base_data

    def _generate_eoa(self, base_data: Dict) -> Dict:
        """Generate Externally Owned Account data."""
        # Realistic balance distribution for EOAs
        balance_eth = np.random.lognormal(-2, 2)  # Most have small balances
        balance_wei = max(0, int(balance_eth * 10**18))

        # Transaction count based on balance (more money = more activity)
        if balance_eth > 10:
            tx_count = np.random.poisson(100)
        elif balance_eth > 1:
            tx_count = np.random.poisson(50)
        else:
            tx_count = np.random.poisson(10)

        base_data.update({
            "balance": str(balance_wei),
            "transaction_count": tx_count,
            "labels": ["eoa", "user"],
            "risk_score": random.uniform(0.0, 0.3),  # Most EOAs are low risk
            "activity_score": min(1.0, tx_count / 100),
            "sophistication_score": random.uniform(0.0, 0.5)
        })

        return base_data

    def _generate_contract(self, base_data: Dict) -> Dict:
        """Generate smart contract data."""
        contract_types = ["token", "nft", "dex", "lending", "governance", "multisig"]
        contract_type = random.choice(contract_types)

        # Contracts generally have higher transaction counts
        tx_count = np.random.poisson(200)

        base_data.update({
            "balance": str(random.randint(0, 10**17)),  # Usually small balance
            "transaction_count": tx_count,
            "labels": ["contract", contract_type],
            "contract_type": contract_type,
            "deployment_timestamp": base_data["first_seen"],
            "risk_score": random.uniform(0.1, 0.6),
            "activity_score": min(1.0, tx_count / 300),
            "verification_status": random.choice(["verified", "unverified"])
        })

        return base_data

    def _generate_exchange(self, base_data: Dict) -> Dict:
        """Generate exchange address data."""
        exchanges = ["binance", "coinbase", "kraken", "okex", "huobi", "bybit"]
        exchange_name = random.choice(exchanges)

        # Exchanges have very high transaction counts and balances
        tx_count = np.random.poisson(5000)
        balance_eth = np.random.lognormal(8, 1)  # Large balances
        balance_wei = int(balance_eth * 10**18)

        base_data.update({
            "balance": str(balance_wei),
            "transaction_count": tx_count,
            "labels": ["exchange", "centralized_exchange", exchange_name],
            "exchange_name": exchange_name,
            "risk_score": random.uniform(0.1, 0.4),  # Generally trusted
            "activity_score": 1.0,  # Maximum activity
            "compliance_status": "kyc_required"
        })

        return base_data

    def _generate_defi_protocol(self, base_data: Dict) -> Dict:
        """Generate DeFi protocol address data."""
        protocols = ["uniswap", "aave", "compound", "makerdao", "curve", "balancer"]
        protocol_name = random.choice(protocols)

        # DeFi protocols have high transaction counts
        tx_count = np.random.poisson(1000)

        base_data.update({
            "balance": str(random.randint(10**15, 10**19)),  # Variable balances
            "transaction_count": tx_count,
            "labels": ["defi", "protocol", protocol_name],
            "protocol_name": protocol_name,
            "protocol_type": random.choice(["dex", "lending", "yield_farming"]),
            "risk_score": random.uniform(0.2, 0.7),
            "activity_score": min(1.0, tx_count / 1500),
            "tvl_usd": random.randint(1000000, 1000000000)  # Total Value Locked
        })

        return base_data

    def _generate_institutional(self, base_data: Dict) -> Dict:
        """Generate institutional address data."""
        institutions = ["treasury", "fund", "dao", "foundation", "corporation"]
        institution_type = random.choice(institutions)

        # Large balances, moderate transaction counts
        balance_eth = np.random.lognormal(6, 1)  # Large institutional balances
        balance_wei = int(balance_eth * 10**18)
        tx_count = np.random.poisson(150)

        base_data.update({
            "balance": str(balance_wei),
            "transaction_count": tx_count,
            "labels": ["institutional", institution_type],
            "institution_type": institution_type,
            "risk_score": random.uniform(0.1, 0.3),  # Generally low risk
            "activity_score": min(1.0, tx_count / 200),
            "sophistication_score": random.uniform(0.7, 1.0),  # High sophistication
            "compliance_status": "institutional_grade"
        })

        return base_data

    def _generate_whale(self, base_data: Dict) -> Dict:
        """Generate whale address data."""
        # Very large balances
        balance_eth = np.random.lognormal(7, 0.5)  # Whale-level balances
        balance_wei = int(balance_eth * 10**18)
        tx_count = np.random.poisson(300)

        base_data.update({
            "balance": str(balance_wei),
            "transaction_count": tx_count,
            "labels": ["whale", "high_net_worth"],
            "risk_score": random.uniform(0.2, 0.5),
            "activity_score": min(1.0, tx_count / 400),
            "sophistication_score": random.uniform(0.6, 0.9),
            "influence_score": random.uniform(0.7, 1.0)
        })

        return base_data

    def _generate_mev_bot(self, base_data: Dict) -> Dict:
        """Generate MEV bot address data."""
        tx_count = np.random.poisson(2000)  # Very high transaction count

        base_data.update({
            "balance": str(random.randint(10**16, 10**18)),
            "transaction_count": tx_count,
            "labels": ["mev_bot", "automated", "arbitrage"],
            "risk_score": random.uniform(0.3, 0.8),  # Higher risk due to MEV
            "activity_score": 1.0,
            "sophistication_score": random.uniform(0.8, 1.0),
            "mev_type": random.choice(["arbitrage", "liquidation", "sandwich"])
        })

        return base_data

    def _generate_governance_contract(self, base_data: Dict) -> Dict:
        """Generate governance contract data."""
        tx_count = np.random.poisson(100)

        base_data.update({
            "balance": str(random.randint(0, 10**16)),
            "transaction_count": tx_count,
            "labels": ["governance", "dao", "voting"],
            "risk_score": random.uniform(0.1, 0.4),
            "activity_score": min(1.0, tx_count / 150),
            "governance_type": random.choice(["dao", "multisig", "timelock"]),
            "voting_power": random.uniform(0.1, 1.0)
        })

        return base_data

    def _generate_address_relationships(self, addresses: List[Dict]) -> List[AddressRelationship]:
        """Generate relationships between addresses."""
        relationships = []

        # Generate various types of relationships
        for _ in range(self.config.address_count * 2):  # 2x relationships as addresses
            from_addr, to_addr = random.sample(addresses, 2)

            relationship = self._create_realistic_relationship(
                from_addr["address"],
                to_addr["address"],
                self.address_types[from_addr["address"]],
                self.address_types[to_addr["address"]]
            )

            if relationship:
                relationships.append(relationship)

        return relationships

    def _create_realistic_relationship(
        self,
        from_addr: str,
        to_addr: str,
        from_type: AddressType,
        to_type: AddressType
    ) -> Optional[AddressRelationship]:
        """Create realistic relationship based on address types."""

        # Define relationship probabilities based on address types
        relationship_rules = {
            (AddressType.EOA, AddressType.EXCHANGE): ("user_to_exchange", 0.8),
            (AddressType.EOA, AddressType.DEFI_PROTOCOL): ("defi_interaction", 0.6),
            (AddressType.INSTITUTIONAL, AddressType.DEFI_PROTOCOL): ("institutional_defi", 0.9),
            (AddressType.MEV_BOT, AddressType.DEFI_PROTOCOL): ("mev_arbitrage", 0.95),
            (AddressType.WHALE, AddressType.EXCHANGE): ("whale_trading", 0.7),
        }

        rule_key = (from_type, to_type)
        if rule_key not in relationship_rules:
            # Generic relationship
            relationship_type = "generic_transfer"
            strength = random.uniform(0.1, 0.3)
        else:
            relationship_type, base_strength = relationship_rules[rule_key]
            strength = random.uniform(base_strength - 0.2, base_strength)

        # Generate interaction timing
        start_time = self._generate_realistic_timestamp()
        end_time = start_time + random.randint(3600, 86400 * 30)  # 1 hour to 30 days
        interaction_count = max(1, int(strength * 100))

        return AddressRelationship(
            from_address=from_addr,
            to_address=to_addr,
            relationship_type=relationship_type,
            strength=strength,
            first_interaction=start_time,
            last_interaction=end_time,
            interaction_count=interaction_count
        )

    def _identify_address_clusters(
        self,
        addresses: List[Dict],
        relationships: List[AddressRelationship]
    ) -> List[Dict]:
        """Identify address clusters based on relationships."""
        # Simple clustering based on strong relationships
        clusters = []
        processed_addresses = set()

        for relationship in relationships:
            if relationship.strength > 0.7:  # Strong relationship threshold
                from_addr = relationship.from_address
                to_addr = relationship.to_address

                if from_addr not in processed_addresses and to_addr not in processed_addresses:
                    cluster = {
                        "cluster_id": len(clusters),
                        "addresses": [from_addr, to_addr],
                        "cluster_type": relationship.relationship_type,
                        "strength": relationship.strength,
                        "first_activity": relationship.first_interaction,
                        "last_activity": relationship.last_interaction
                    }
                    clusters.append(cluster)
                    processed_addresses.update([from_addr, to_addr])

        return clusters

    def _load_protocol_addresses(self) -> Dict[str, str]:
        """Load known protocol addresses for realistic generation."""
        # In a real implementation, this would load from fixtures/protocol_configs.json
        return {
            "uniswap_v3_factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
            "aave_lending_pool": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            "compound_comptroller": "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B",
            "makerdao_dai": "0x6B175474E89094C44Da98b954EedeAC495271d0F"
        }
```

### Transaction Pattern Generation

#### mock_data/generators/transactions.py
```python
"""Transaction pattern generation with realistic blockchain behavior."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from .base import BaseGenerator, GenerationConfig, ComplexityLevel


class TransactionType(Enum):
    """Types of blockchain transactions."""
    TRANSFER = "transfer"
    CONTRACT_CALL = "contract_call"
    CONTRACT_CREATION = "contract_creation"
    DEFI_SWAP = "defi_swap"
    DEFI_LIQUIDITY = "defi_liquidity"
    NFT_TRANSFER = "nft_transfer"
    NFT_MINT = "nft_mint"
    GOVERNANCE_VOTE = "governance_vote"
    MEV_ARBITRAGE = "mev_arbitrage"
    MULTI_SEND = "multi_send"


@dataclass
class TransactionPattern:
    """Pattern for generating realistic transaction sequences."""
    pattern_type: str
    frequency_per_day: float
    value_range: Tuple[float, float]  # ETH
    gas_limit_range: Tuple[int, int]
    typical_addresses: List[str]
    success_rate: float


class TransactionGenerator(BaseGenerator):
    """Generate realistic transaction patterns."""

    def __init__(self, config: GenerationConfig, addresses: List[Dict]):
        super().__init__(config)
        self.addresses = addresses
        self.address_lookup = {addr["address"]: addr for addr in addresses}
        self.transaction_patterns = self._create_transaction_patterns()
        self.current_block = 18500000  # Starting block number
        self.current_timestamp = int(self.config.start_date.timestamp())

    def generate(self) -> Dict[str, List]:
        """Generate comprehensive transaction dataset."""
        transactions = []
        blocks = []

        # Generate transactions in chronological order
        total_days = (self.config.end_date - self.config.start_date).days
        transactions_per_day = self.config.transaction_count // total_days

        for day in range(total_days):
            day_start = self.config.start_date + timedelta(days=day)
            day_transactions = self._generate_daily_transactions(
                day_start, transactions_per_day
            )
            transactions.extend(day_transactions)

            # Generate blocks for this day
            day_blocks = self._generate_daily_blocks(day_start, day_transactions)
            blocks.extend(day_blocks)

        # Sort by timestamp
        transactions.sort(key=lambda x: x["timestamp"])
        blocks.sort(key=lambda x: x["timestamp"])

        return {
            "transactions": transactions,
            "blocks": blocks,
            "transaction_patterns": self._analyze_generated_patterns(transactions)
        }

    def _create_transaction_patterns(self) -> Dict[TransactionType, TransactionPattern]:
        """Create realistic transaction patterns based on complexity."""
        base_patterns = {
            TransactionType.TRANSFER: TransactionPattern(
                pattern_type="simple_transfer",
                frequency_per_day=50.0,
                value_range=(0.001, 10.0),
                gas_limit_range=(21000, 21000),
                typical_addresses=[],
                success_rate=0.98
            ),
            TransactionType.CONTRACT_CALL: TransactionPattern(
                pattern_type="contract_interaction",
                frequency_per_day=30.0,
                value_range=(0.0, 5.0),
                gas_limit_range=(50000, 500000),
                typical_addresses=[],
                success_rate=0.95
            )
        }

        if self.config.complexity in [ComplexityLevel.SOPHISTICATED, ComplexityLevel.EXPERT]:
            base_patterns.update({
                TransactionType.DEFI_SWAP: TransactionPattern(
                    pattern_type="defi_swap",
                    frequency_per_day=20.0,
                    value_range=(0.1, 100.0),
                    gas_limit_range=(150000, 300000),
                    typical_addresses=[],
                    success_rate=0.96
                ),
                TransactionType.DEFI_LIQUIDITY: TransactionPattern(
                    pattern_type="liquidity_provision",
                    frequency_per_day=5.0,
                    value_range=(1.0, 1000.0),
                    gas_limit_range=(200000, 400000),
                    typical_addresses=[],
                    success_rate=0.94
                ),
                TransactionType.NFT_TRANSFER: TransactionPattern(
                    pattern_type="nft_transfer",
                    frequency_per_day=15.0,
                    value_range=(0.1, 50.0),
                    gas_limit_range=(80000, 150000),
                    typical_addresses=[],
                    success_rate=0.97
                )
            })

        if self.config.complexity == ComplexityLevel.EXPERT:
            base_patterns.update({
                TransactionType.MEV_ARBITRAGE: TransactionPattern(
                    pattern_type="mev_arbitrage",
                    frequency_per_day=10.0,
                    value_range=(0.5, 500.0),
                    gas_limit_range=(300000, 800000),
                    typical_addresses=[],
                    success_rate=0.88  # MEV bots sometimes fail
                ),
                TransactionType.GOVERNANCE_VOTE: TransactionPattern(
                    pattern_type="governance_participation",
                    frequency_per_day=2.0,
                    value_range=(0.0, 0.1),
                    gas_limit_range=(100000, 200000),
                    typical_addresses=[],
                    success_rate=0.99
                ),
                TransactionType.MULTI_SEND: TransactionPattern(
                    pattern_type="batch_operations",
                    frequency_per_day=8.0,
                    value_range=(0.1, 20.0),
                    gas_limit_range=(500000, 1000000),
                    typical_addresses=[],
                    success_rate=0.92
                )
            })

        return base_patterns

    def _generate_daily_transactions(
        self,
        day_start: datetime,
        target_count: int
    ) -> List[Dict]:
        """Generate transactions for a single day."""
        transactions = []

        # Distribute transactions throughout the day with realistic patterns
        for _ in range(target_count):
            # Choose transaction type based on patterns
            tx_type = self._choose_transaction_type()
            pattern = self.transaction_patterns[tx_type]

            # Generate transaction based on type
            transaction = self._generate_transaction_by_type(
                tx_type, pattern, day_start
            )

            if transaction:
                transactions.append(transaction)

        return transactions

    def _choose_transaction_type(self) -> TransactionType:
        """Choose transaction type based on complexity and probability."""
        available_types = list(self.transaction_patterns.keys())
        weights = [pattern.frequency_per_day for pattern in self.transaction_patterns.values()]

        # Normalize weights
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        return np.random.choice(available_types, p=probabilities)

    def _generate_transaction_by_type(
        self,
        tx_type: TransactionType,
        pattern: TransactionPattern,
        day_start: datetime
    ) -> Optional[Dict]:
        """Generate specific transaction based on type and pattern."""

        # Basic transaction structure
        tx = {
            "hash": self._generate_transaction_hash(),
            "type": tx_type.value,
            "timestamp": self._generate_daily_timestamp(day_start),
            "block_number": self.current_block + random.randint(0, 100),
            "transaction_index": random.randint(0, 200),
            "nonce": random.randint(0, 1000),
            "gas_limit": random.randint(*pattern.gas_limit_range),
            "gas_price": self._generate_realistic_gas_price(),
            "gas_used": 0,  # Will be calculated
            "status": 1 if random.random() < pattern.success_rate else 0,
            "value": "0",
            "input_data": "0x",
            "logs": []
        }

        # Calculate gas used (usually less than limit)
        if tx["status"] == 1:  # Successful transaction
            tx["gas_used"] = random.randint(
                int(tx["gas_limit"] * 0.7),
                tx["gas_limit"]
            )
        else:  # Failed transaction
            tx["gas_used"] = tx["gas_limit"]  # Used all gas

        # Type-specific generation
        if tx_type == TransactionType.TRANSFER:
            return self._generate_transfer_transaction(tx, pattern)
        elif tx_type == TransactionType.CONTRACT_CALL:
            return self._generate_contract_call_transaction(tx, pattern)
        elif tx_type == TransactionType.DEFI_SWAP:
            return self._generate_defi_swap_transaction(tx, pattern)
        elif tx_type == TransactionType.DEFI_LIQUIDITY:
            return self._generate_defi_liquidity_transaction(tx, pattern)
        elif tx_type == TransactionType.NFT_TRANSFER:
            return self._generate_nft_transaction(tx, pattern)
        elif tx_type == TransactionType.MEV_ARBITRAGE:
            return self._generate_mev_transaction(tx, pattern)
        elif tx_type == TransactionType.GOVERNANCE_VOTE:
            return self._generate_governance_transaction(tx, pattern)
        elif tx_type == TransactionType.MULTI_SEND:
            return self._generate_multi_send_transaction(tx, pattern)
        else:
            return self._generate_generic_transaction(tx, pattern)

    def _generate_transfer_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate simple ETH transfer transaction."""
        from_addr, to_addr = self._select_realistic_address_pair("transfer")

        value_eth = random.uniform(*pattern.value_range)
        tx.update({
            "from": from_addr,
            "to": to_addr,
            "value": str(int(value_eth * 10**18)),
            "method_name": "transfer",
            "description": "ETH transfer"
        })

        return tx

    def _generate_contract_call_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate contract interaction transaction."""
        from_addr = self._select_address_by_type(["eoa", "institutional"])
        contract_addr = self._select_address_by_type(["contract"])

        if not from_addr or not contract_addr:
            return None

        methods = ["approve", "transfer", "withdraw", "deposit", "claim", "stake"]
        method = random.choice(methods)

        tx.update({
            "from": from_addr,
            "to": contract_addr,
            "value": "0",  # Most contract calls don't send ETH
            "method_name": method,
            "input_data": f"0x{random.randbytes(100).hex()}",
            "description": f"Contract call: {method}"
        })

        return tx

    def _generate_defi_swap_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate DeFi swap transaction."""
        from_addr = self._select_address_by_type(["eoa", "institutional", "mev_bot"])
        defi_addr = self._select_address_by_type(["defi"])

        if not from_addr or not defi_addr:
            return None

        value_eth = random.uniform(*pattern.value_range)
        swap_types = ["exactTokensForTokens", "exactETHForTokens", "exactTokensForETH"]
        swap_type = random.choice(swap_types)

        tx.update({
            "from": from_addr,
            "to": defi_addr,
            "value": str(int(value_eth * 10**18)) if "ETH" in swap_type else "0",
            "method_name": swap_type,
            "input_data": f"0x{random.randbytes(200).hex()}",
            "description": f"DeFi swap: {swap_type}",
            "protocol": "uniswap_v3",  # Example protocol
            "slippage_tolerance": random.uniform(0.1, 3.0)
        })

        # Add swap-specific logs
        tx["logs"] = [
            {
                "address": defi_addr,
                "topics": ["0x" + random.randbytes(32).hex()],
                "data": "0x" + random.randbytes(64).hex()
            }
        ]

        return tx

    def _generate_defi_liquidity_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate DeFi liquidity provision transaction."""
        from_addr = self._select_address_by_type(["eoa", "institutional", "whale"])
        defi_addr = self._select_address_by_type(["defi"])

        if not from_addr or not defi_addr:
            return None

        value_eth = random.uniform(*pattern.value_range)
        liquidity_methods = ["addLiquidity", "removeLiquidity", "addLiquidityETH"]
        method = random.choice(liquidity_methods)

        tx.update({
            "from": from_addr,
            "to": defi_addr,
            "value": str(int(value_eth * 10**18)) if "ETH" in method else "0",
            "method_name": method,
            "input_data": f"0x{random.randbytes(300).hex()}",
            "description": f"Liquidity operation: {method}",
            "protocol": "uniswap_v3",
            "pool_fee": random.choice([500, 3000, 10000])  # Fee tiers
        })

        return tx

    def _generate_nft_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate NFT transaction."""
        from_addr, to_addr = self._select_realistic_address_pair("nft")
        nft_contract = self._select_address_by_type(["contract"])

        if not nft_contract:
            return None

        value_eth = random.uniform(*pattern.value_range)
        nft_methods = ["safeTransferFrom", "transferFrom", "mint", "burn"]
        method = random.choice(nft_methods)

        tx.update({
            "from": from_addr,
            "to": nft_contract,
            "value": str(int(value_eth * 10**18)) if method == "mint" else "0",
            "method_name": method,
            "input_data": f"0x{random.randbytes(150).hex()}",
            "description": f"NFT operation: {method}",
            "token_id": random.randint(1, 10000),
            "collection": "0x" + random.randbytes(20).hex()
        })

        return tx

    def _generate_mev_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate MEV arbitrage transaction."""
        mev_bot = self._select_address_by_type(["mev_bot"])
        defi_addr = self._select_address_by_type(["defi"])

        if not mev_bot or not defi_addr:
            return None

        value_eth = random.uniform(*pattern.value_range)

        tx.update({
            "from": mev_bot,
            "to": defi_addr,
            "value": str(int(value_eth * 10**18)),
            "method_name": "arbitrage",
            "input_data": f"0x{random.randbytes(400).hex()}",
            "description": "MEV arbitrage opportunity",
            "gas_price": self._generate_realistic_gas_price() * 2,  # MEV bots pay more
            "mev_type": random.choice(["arbitrage", "liquidation", "sandwich"]),
            "profit_eth": random.uniform(0.01, 5.0)
        })

        return tx

    def _generate_governance_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate governance voting transaction."""
        voter = self._select_address_by_type(["eoa", "institutional", "whale"])
        governance_contract = self._select_address_by_type(["governance"])

        if not voter or not governance_contract:
            return None

        tx.update({
            "from": voter,
            "to": governance_contract,
            "value": "0",
            "method_name": "castVote",
            "input_data": f"0x{random.randbytes(100).hex()}",
            "description": "Governance vote",
            "proposal_id": random.randint(1, 100),
            "vote_type": random.choice(["for", "against", "abstain"]),
            "voting_power": random.randint(1, 1000000)
        })

        return tx

    def _generate_multi_send_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate multi-send batch transaction."""
        from_addr = self._select_address_by_type(["institutional", "exchange"])
        multisend_contract = self._select_address_by_type(["contract"])

        if not from_addr or not multisend_contract:
            return None

        value_eth = random.uniform(*pattern.value_range)
        recipient_count = random.randint(5, 50)

        tx.update({
            "from": from_addr,
            "to": multisend_contract,
            "value": str(int(value_eth * 10**18)),
            "method_name": "multiSend",
            "input_data": f"0x{random.randbytes(500).hex()}",
            "description": f"Batch send to {recipient_count} recipients",
            "recipient_count": recipient_count,
            "total_value": value_eth
        })

        return tx

    def _generate_generic_transaction(self, tx: Dict, pattern: TransactionPattern) -> Dict:
        """Generate generic transaction as fallback."""
        from_addr, to_addr = self._select_realistic_address_pair("generic")

        tx.update({
            "from": from_addr,
            "to": to_addr,
            "value": "0",
            "method_name": "unknown",
            "description": "Generic transaction"
        })

        return tx

    def _select_realistic_address_pair(self, interaction_type: str) -> Tuple[str, str]:
        """Select realistic address pair based on interaction type."""
        if interaction_type == "transfer":
            from_addr = self._select_address_by_type(["eoa", "exchange", "institutional"])
            to_addr = self._select_address_by_type(["eoa", "exchange"])
        elif interaction_type == "nft":
            from_addr = self._select_address_by_type(["eoa", "whale"])
            to_addr = self._select_address_by_type(["eoa", "whale"])
        else:
            from_addr = random.choice(self.addresses)["address"]
            to_addr = random.choice(self.addresses)["address"]

        return from_addr or self.addresses[0]["address"], to_addr or self.addresses[1]["address"]

    def _select_address_by_type(self, allowed_types: List[str]) -> Optional[str]:
        """Select address by type labels."""
        candidates = [
            addr for addr in self.addresses
            if any(label in addr.get("labels", []) for label in allowed_types)
        ]

        if candidates:
            return random.choice(candidates)["address"]
        return None

    def _generate_daily_timestamp(self, day_start: datetime) -> int:
        """Generate timestamp within a day with realistic distribution."""
        # More activity during certain hours
        hour = np.random.beta(2, 5) * 24  # Beta distribution for realistic timing
        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        day_time = day_start + timedelta(hours=hour, minutes=minute, seconds=second)
        return int(day_time.timestamp())

    def _generate_daily_blocks(self, day_start: datetime, transactions: List[Dict]) -> List[Dict]:
        """Generate blocks for a day's transactions."""
        blocks = []

        # Group transactions by approximate block time (12-15 seconds per block)
        block_time = 13  # Average block time
        day_seconds = 86400
        blocks_per_day = day_seconds // block_time

        for i in range(blocks_per_day):
            block_timestamp = int(day_start.timestamp()) + (i * block_time)
            block_number = self.current_block + i

            # Find transactions for this block
            block_transactions = [
                tx["hash"] for tx in transactions
                if abs(tx["timestamp"] - block_timestamp) < block_time // 2
            ]

            block = {
                "number": block_number,
                "hash": self._generate_block_hash(),
                "parent_hash": "0x" + random.randbytes(32).hex(),
                "timestamp": block_timestamp,
                "gas_limit": 30000000,
                "gas_used": sum(
                    tx["gas_used"] for tx in transactions
                    if tx["hash"] in block_transactions
                ),
                "transaction_count": len(block_transactions),
                "transactions": block_transactions,
                "miner": self._select_address_by_type(["mev_bot", "institutional"]) or self.addresses[0]["address"],
                "difficulty": random.randint(1000000, 10000000),
                "total_difficulty": random.randint(10**15, 10**16)
            }

            blocks.append(block)

        self.current_block += blocks_per_day
        return blocks

    def _analyze_generated_patterns(self, transactions: List[Dict]) -> Dict:
        """Analyze patterns in generated transactions for validation."""
        if not transactions:
            return {}

        # Calculate pattern statistics
        tx_types = [tx.get("type", "unknown") for tx in transactions]
        type_counts = {tx_type: tx_types.count(tx_type) for tx_type in set(tx_types)}

        gas_prices = [int(tx.get("gas_price", 0)) for tx in transactions]
        values = [int(tx.get("value", "0")) for tx in transactions]

        return {
            "total_transactions": len(transactions),
            "transaction_types": type_counts,
            "gas_price_stats": {
                "mean": np.mean(gas_prices) if gas_prices else 0,
                "median": np.median(gas_prices) if gas_prices else 0,
                "std": np.std(gas_prices) if gas_prices else 0
            },
            "value_stats": {
                "mean": np.mean(values) if values else 0,
                "median": np.median(values) if values else 0,
                "total": sum(values)
            },
            "success_rate": sum(1 for tx in transactions if tx.get("status") == 1) / len(transactions),
            "time_span": {
                "start": min(tx["timestamp"] for tx in transactions),
                "end": max(tx["timestamp"] for tx in transactions)
            }
        }
```

This implementation provides a comprehensive mock data generation system that creates realistic blockchain patterns with sophisticated transaction types, proper relationships between addresses, and statistically accurate distributions. The system supports three complexity levels and can generate everything from simple transfers to complex MEV operations and governance activities.
