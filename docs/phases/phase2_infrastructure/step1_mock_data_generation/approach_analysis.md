# Step 1: Mock Data Generation System - Approach Analysis

**Objective:** Create realistic blockchain mock data that simulates real Ethereum patterns
**Context:** No API access, need realistic data for agent development
**Estimated Duration:** 4-6 hours

---

## Approach Options

### Option 1: Static JSON Files (Simple)
```python
# Pre-generated static mock data
mock_data/
├── addresses/
│   ├── whale_address_1.json
│   ├── normal_user_1.json
│   └── suspicious_address_1.json
├── transactions/
│   ├── whale_txs.json
│   └── normal_txs.json
└── protocols/
    ├── uniswap_interactions.json
    └── aave_interactions.json
```

**Pros:**
- Fast to implement
- Consistent test data
- No runtime generation overhead
- Easy to version control specific scenarios

**Cons:**
- Limited variety
- Hard to test edge cases
- Requires manual curation
- Doesn't scale to many addresses

### Option 2: Dynamic Generation with Faker (Recommended)
```python
# Runtime generation with realistic patterns
class EthereumAddressFactory:
    def generate_whale_profile(self) -> MockAddressProfile
    def generate_normal_user(self) -> MockAddressProfile
    def generate_suspicious_address(self) -> MockAddressProfile

class TransactionFactory:
    def generate_realistic_history(self, profile, days=90) -> List[Transaction]
```

**Pros:**
- Infinite variety of test data
- Parameterizable scenarios
- Can simulate complex patterns
- Easy to add new address types

**Cons:**
- More complex to implement
- Runtime generation overhead
- Need to ensure consistency
- Requires blockchain domain knowledge

### Option 3: Hybrid Approach (Best for Prototype)
```python
# Combine static templates with dynamic variation
templates/
├── address_patterns.json    # Base patterns
├── protocol_templates.json  # Protocol interaction templates
└── risk_scenarios.json     # Risk flag scenarios

# Plus dynamic generation using templates
```

**Pros:**
- Best of both approaches
- Fast realistic generation
- Consistent base patterns
- Easy to extend

**Cons:**
- Slightly more complex
- Need to maintain templates

---

## Recommended Approach: Hybrid with Faker

### Implementation Strategy

#### 1. Address Profile Templates
```json
{
  "whale": {
    "balance_range": [1000, 10000],
    "tx_frequency": "high",
    "protocols": ["uniswap", "aave", "compound"],
    "counterparty_diversity": "high",
    "risk_level": "low"
  },
  "normal_user": {
    "balance_range": [1, 100],
    "tx_frequency": "medium",
    "protocols": ["uniswap", "opensea"],
    "counterparty_diversity": "medium",
    "risk_level": "low"
  },
  "suspicious": {
    "balance_range": [0.1, 10],
    "tx_frequency": "burst",
    "protocols": ["mixers", "bridges"],
    "counterparty_diversity": "low",
    "risk_level": "high"
  }
}
```

#### 2. Realistic Data Patterns
Based on real Ethereum behavior:

**Transaction Patterns:**
- Normal users: 1-10 transactions per day
- Whales: 10-100 transactions per day
- Bots: Consistent timing patterns
- Suspicious: Burst patterns, then dormancy

**Protocol Interactions:**
- DEX swaps: Uniswap, SushiSwap patterns
- Lending: AAVE, Compound deposit/withdraw cycles
- NFTs: OpenSea trading patterns
- DeFi farming: Complex multi-step transactions

**Risk Indicators:**
- Mixer interactions (Tornado Cash patterns)
- Phishing contract interactions
- Approval-for-all patterns
- Bridge exploitation patterns

---

## Technical Implementation

### Core Components

#### 1. Address Factory
```python
from faker import Faker
from faker.providers import BaseProvider
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

class EthereumProvider(BaseProvider):
    """Custom Faker provider for Ethereum-specific data."""

    def eth_address(self) -> str:
        """Generate a valid Ethereum address."""
        return "0x" + self.generator.hexify(text="^" * 40, upper=False)

    def tx_hash(self) -> str:
        """Generate a valid transaction hash."""
        return "0x" + self.generator.hexify(text="^" * 64, upper=False)

    def block_number(self, days_ago: int = 0) -> int:
        """Generate realistic block number."""
        current_block = 18_500_000  # Approximate current mainnet
        blocks_per_day = 7200  # ~12 second blocks
        return current_block - (days_ago * blocks_per_day) + random.randint(-1000, 1000)

class MockAddressProfile:
    """Profile for generating consistent address behavior."""

    def __init__(self, profile_type: str, faker: Faker):
        self.faker = faker
        self.address = faker.eth_address()
        self.profile_type = profile_type
        self.created_at = faker.date_time_between(start_date="-2y", end_date="now")

        # Load profile configuration
        self.config = self._load_profile_config(profile_type)

    def generate_transaction_history(self, days: int = 90) -> List[Dict[str, Any]]:
        """Generate realistic transaction history."""
        transactions = []
        current_date = datetime.now()

        for day in range(days):
            date = current_date - timedelta(days=day)

            # Generate transactions for this day based on profile
            daily_tx_count = self._get_daily_tx_count(date)

            for _ in range(daily_tx_count):
                tx = self._generate_transaction(date)
                transactions.append(tx)

        return sorted(transactions, key=lambda x: x["timestamp"])

    def _generate_transaction(self, date: datetime) -> Dict[str, Any]:
        """Generate a single realistic transaction."""
        tx_type = random.choices(
            ["transfer", "swap", "approval", "contract_call"],
            weights=self.config["tx_type_weights"]
        )[0]

        base_tx = {
            "hash": self.faker.tx_hash(),
            "from": self.address,
            "to": self._get_counterparty(tx_type),
            "value": self._get_transaction_value(tx_type),
            "gas_used": random.randint(21000, 500000),
            "gas_price": random.randint(20, 200) * 10**9,  # gwei
            "timestamp": date.timestamp(),
            "block_number": self.faker.block_number(
                days_ago=(datetime.now() - date).days
            ),
            "status": 1 if random.random() > 0.02 else 0,  # 2% failure rate
            "type": tx_type
        }

        # Add type-specific data
        if tx_type == "swap":
            base_tx.update(self._generate_swap_data())
        elif tx_type == "approval":
            base_tx.update(self._generate_approval_data())

        return base_tx
```

#### 2. Protocol Interaction Simulator
```python
class ProtocolInteractionFactory:
    """Generate realistic DeFi protocol interactions."""

    PROTOCOLS = {
        "uniswap": {
            "contract": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
            "functions": ["swapExactETHForTokens", "swapExactTokensForETH"],
            "typical_gas": [150000, 180000]
        },
        "aave": {
            "contract": "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
            "functions": ["deposit", "withdraw", "borrow", "repay"],
            "typical_gas": [200000, 250000]
        },
        "tornado_cash": {  # For risk scenarios
            "contract": "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936",
            "functions": ["deposit", "withdraw"],
            "typical_gas": [1000000, 1200000]
        }
    }

    def generate_protocol_interaction(self, protocol: str, address: str) -> Dict[str, Any]:
        """Generate realistic protocol interaction."""
        config = self.PROTOCOLS[protocol]

        return {
            "to": config["contract"],
            "input": self._generate_function_call(config["functions"]),
            "gas_used": random.choice(config["typical_gas"]),
            "protocol": protocol
        }
```

#### 3. Risk Scenario Generator
```python
class RiskScenarioFactory:
    """Generate specific risk scenarios for testing."""

    def generate_mixer_user(self) -> MockAddressProfile:
        """Generate address that uses mixers."""
        profile = MockAddressProfile("suspicious", self.faker)

        # Add tornado cash interactions
        mixer_txs = []
        for _ in range(random.randint(2, 8)):
            mixer_tx = self._generate_mixer_transaction(profile.address)
            mixer_txs.append(mixer_tx)

        profile.risk_transactions = mixer_txs
        return profile

    def generate_approval_farming_scenario(self) -> MockAddressProfile:
        """Generate address with suspicious approval patterns."""
        profile = MockAddressProfile("suspicious", self.faker)

        # Generate many approvals to unknown contracts
        approval_txs = []
        for _ in range(random.randint(10, 30)):
            approval_tx = self._generate_suspicious_approval(profile.address)
            approval_txs.append(approval_tx)

        profile.risk_transactions = approval_txs
        return profile
```

---

## Data Persistence Strategy

### Option 1: Runtime Generation Only
- Generate fresh data each time
- Fast startup, no storage requirements
- Good for development and testing

### Option 2: Cache Generated Data
```python
class MockDataCache:
    """Cache generated mock data for consistency."""

    def __init__(self, cache_dir: str = "./mock_data_cache"):
        self.cache_dir = cache_dir
        self.cache = diskcache.Cache(cache_dir)

    def get_or_generate_address(self, address: str, profile_type: str) -> MockAddressProfile:
        """Get cached address or generate new one."""
        cache_key = f"address:{address}:{profile_type}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        profile = MockAddressProfile(profile_type, self.faker)
        self.cache[cache_key] = profile
        return profile
```

### Option 3: Hybrid with Seed Control
```python
# Deterministic generation with seeds
def generate_deterministic_data(seed: int = 12345) -> Dict[str, Any]:
    """Generate deterministic mock data for testing."""
    faker = Faker()
    faker.seed_instance(seed)

    # Generate consistent test data
    return {
        "addresses": generate_test_addresses(faker, count=100),
        "scenarios": generate_test_scenarios(faker)
    }
```

---

## Validation and Testing

### Data Quality Checks
```python
def validate_mock_data(address_profile: MockAddressProfile) -> bool:
    """Validate generated mock data quality."""

    checks = [
        # Address format validation
        len(address_profile.address) == 42,
        address_profile.address.startswith("0x"),

        # Transaction consistency
        all(tx["from"] == address_profile.address for tx in address_profile.transactions),

        # Realistic patterns
        address_profile.daily_tx_count > 0,
        address_profile.counterparty_count > 0,

        # Gas usage reasonable
        all(21000 <= tx["gas_used"] <= 2000000 for tx in address_profile.transactions)
    ]

    return all(checks)
```

### Performance Benchmarks
```python
def benchmark_mock_generation():
    """Benchmark mock data generation performance."""
    import time

    start = time.time()

    # Generate 100 addresses with 90 days of history
    for _ in range(100):
        profile = MockAddressProfile("normal_user", faker)
        transactions = profile.generate_transaction_history(90)

    duration = time.time() - start
    print(f"Generated 100 address profiles in {duration:.2f}s")
```

---

## Success Criteria

**Step 1 is complete when:**

1. ✅ **Realistic Patterns** - Generated data resembles real Ethereum behavior
2. ✅ **Multiple Scenarios** - Can generate whale, normal, suspicious address types
3. ✅ **Protocol Interactions** - Simulates real DeFi protocol usage patterns
4. ✅ **Risk Scenarios** - Can generate specific risk patterns for testing
5. ✅ **Performance** - Can generate test data quickly enough for development
6. ✅ **Validation** - Generated data passes quality checks
7. ✅ **Deterministic** - Can generate consistent data for testing
8. ✅ **Extensible** - Easy to add new address types and scenarios

**Next Step Dependencies:**
- Provides realistic test data for schema validation
- Supplies varied scenarios for tool adapter testing
- Enables comprehensive agent behavior testing