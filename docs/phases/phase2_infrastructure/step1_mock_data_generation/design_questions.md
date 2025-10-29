# Step 1: Mock Data Generation System - Design Questions

**Context:** Creating realistic blockchain mock data without API access
**Decision Point:** Balancing realism, performance, and development complexity

---

## Critical Design Questions

### 1. Mock Data Realism Level
**Question:** How realistic should the mock data be vs development speed?

**Options:**
- **High Fidelity** - Exact mimicking of real blockchain patterns
- **Medium Fidelity** - Realistic enough for agent development
- **Low Fidelity** - Simple patterns that satisfy basic requirements

**Considerations:**
- **Agent Quality:** More realistic data = better agent training
- **Development Speed:** Complex mock data takes longer to implement
- **Debugging:** Overly complex data can hide bugs vs reveal them
- **Future Integration:** Mock patterns should match real API responses

**Decision Needed:**
- What level of blockchain knowledge should be encoded?
- How important is exact gas usage patterns vs approximate?
- Should we simulate MEV, flashloans, and complex DeFi interactions?

### 2. Data Generation Strategy
**Question:** When and how should mock data be generated?

**Options:**
- **Runtime Generation** - Generate fresh data each execution
- **Pre-generated Files** - Static JSON files with curated scenarios
- **Cached Generation** - Generate once, cache for consistency
- **Hybrid Approach** - Mix of static templates and dynamic generation

**Trade-offs:**
- **Consistency:** Static data is consistent, dynamic data varies
- **Performance:** Pre-generated is fastest, runtime generation is slowest
- **Flexibility:** Dynamic allows parameterization, static is fixed
- **Storage:** Pre-generated requires disk space, runtime needs memory

**Decision Needed:**
- How important is data consistency across test runs?
- Should different developers get the same mock data?
- How much variety is needed for comprehensive testing?

### 3. Address Profile Diversity
**Question:** What types of address profiles should we support?

**Profiles from Spec:**
- Normal users (various activity levels)
- Whales (high-value, high-activity)
- Bots (consistent patterns)
- Suspicious actors (risk indicators)
- Smart contracts (different types)

**Additional Considerations:**
- **DeFi Power Users** - Complex protocol interactions
- **NFT Traders** - OpenSea, marketplace patterns
- **Bridge Users** - Cross-chain activity simulation
- **DAO Members** - Governance token interactions

**Decision Needed:**
- How many distinct profile types to implement initially?
- Should profiles be composable (e.g., whale + NFT trader)?
- How granular should behavioral differences be?

### 4. Protocol Interaction Modeling
**Question:** Which DeFi protocols should be simulated and at what depth?

**Core Protocols (High Priority):**
- **Uniswap/DEXs** - Swapping patterns
- **AAVE/Compound** - Lending protocols
- **OpenSea** - NFT marketplace

**Risk Protocols (For Detection Testing):**
- **Tornado Cash** - Mixer interactions
- **Suspicious Contracts** - Phishing, scams
- **Bridge Exploits** - Cross-chain risk patterns

**Advanced Protocols (Lower Priority):**
- **Curve** - Stablecoin trading
- **Yearn** - Yield farming
- **Governance** - DAO voting patterns

**Decision Needed:**
- How deep should protocol simulation go (function calls vs just transfers)?
- Should we model protocol state changes?
- How important are gas cost patterns vs just transaction flows?

### 5. Risk Scenario Coverage
**Question:** What risk scenarios should be encoded for sybil/risk detection testing?

**Sybil Patterns:**
- **Shared Funding Source** - Multiple addresses funded from same source
- **Burst Activity** - Coordinated transaction bursts
- **Similar Patterns** - Identical behavioral patterns
- **Address Clustering** - Related address generation patterns

**Risk Patterns:**
- **Mixer Usage** - Various privacy protocol interactions
- **Approval Farming** - Suspicious approval-for-all patterns
- **Sandwich Attacks** - MEV exploitation patterns
- **Bridge Exploits** - Cross-chain manipulation

**Decision Needed:**
- How sophisticated should sybil detection test scenarios be?
- Should we model evolving attack patterns?
- How obvious should risk patterns be for initial testing?

---

## Secondary Design Questions

### 6. Performance Requirements
**Question:** What are the performance constraints for mock data generation?

**Considerations:**
- **Development Speed** - How fast should test data generate?
- **Memory Usage** - How much mock data can be held in memory?
- **Disk Storage** - Acceptable cache size for persistent data?

### 7. Data Consistency and Seeding
**Question:** How should deterministic generation work?

**Approaches:**
- **Global Seed** - Same seed generates identical datasets
- **Address-Based Seeds** - Consistent data per address
- **Scenario Seeds** - Consistent data per test scenario

### 8. Extensibility and Maintenance
**Question:** How should new patterns and protocols be added?

**Considerations:**
- **Configuration Files** - JSON/YAML pattern definitions
- **Plugin System** - Modular protocol simulators
- **Code Generation** - Template-based pattern creation

---

## Recommended Decisions

### High Confidence Recommendations

1. **Medium-High Fidelity Mock Data** ✅
   - **Rationale:** Realistic enough for agent development, not overly complex
   - **Implementation:** Focus on transaction patterns, basic gas usage, protocol interactions

2. **Hybrid Generation Strategy** ✅
   - **Rationale:** Best balance of consistency, performance, and flexibility
   - **Implementation:** Static templates + dynamic variation with caching

3. **Core Profile Types First** ✅
   - **Rationale:** Start with 4-5 essential profiles, expand later
   - **Implementation:** Normal, Whale, Suspicious, Bot, Contract profiles

4. **Essential Protocol Focus** ✅
   - **Rationale:** Implement Uniswap, AAVE, OpenSea, Tornado Cash patterns
   - **Implementation:** Function-level simulation for key protocols

### Decisions Requiring Input

1. **Data Consistency Requirements**
   - Should test runs be deterministic or allow variation?
   - How important is reproducible test scenarios?

2. **Risk Pattern Sophistication**
   - How obvious should risk patterns be initially?
   - Should we model evolving/subtle attack patterns?

3. **Performance vs Realism Trade-offs**
   - Acceptable generation time for development workflow?
   - Memory/storage constraints for mock data cache?

---

## Specific Questions for Stakeholder

### Development Workflow
1. **Test Consistency:** Do you prefer deterministic test data (same results each time) or varied data (different patterns each run)?

2. **Performance Priority:** What's more important - fast generation time or highly realistic data patterns?

3. **Risk Scenario Focus:** Should risk patterns be obvious (for easier debugging) or subtle (for realistic detection testing)?

### Technical Scope
1. **Protocol Depth:** Should we simulate complex DeFi interactions (flashloans, liquidations) or focus on basic patterns (swaps, transfers)?

2. **Blockchain Features:** How important is modeling gas optimization patterns, MEV, and advanced Ethereum features?

3. **Future Extensibility:** Priority on clean architecture for adding new protocols vs faster initial implementation?

---

## Impact on Later Phases

### Phase 3 (Agents)
- **Quality of mock data directly impacts agent development quality**
- **Risk scenarios determine how well agents can be tested**
- **Protocol patterns affect feature extraction accuracy**

### Phase 4 (Production)
- **Mock interfaces must match real API integration patterns**
- **Performance characteristics affect real-world scalability**
- **Data validation patterns carry forward to production**

---

## Next Steps

1. **Gather stakeholder input** on critical decisions
2. **Research real blockchain patterns** for fidelity baseline
3. **Prototype core generation logic** to validate approach
4. **Document final decisions** in implementation notes
5. **Begin implementation** with chosen strategy

**Key Principle:** Build mock data that's realistic enough to develop quality agents, but simple enough to implement quickly in a prototype timeline.