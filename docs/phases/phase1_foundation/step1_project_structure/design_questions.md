# Step 1: Project Structure Setup - Design Questions

**Context:** Setting up the foundational project structure for ReputeNet
**Decision Point:** Establishing conventions that will impact the entire development process

---

## Critical Design Questions

### 1. Package Management Strategy
**Question:** Should we use UV, pip, poetry, or another package manager?

**Context from Spec:**
- Specification suggests `uv pip install -r requirements.txt` or `pip install -r requirements.txt`
- Modern Python projects often use more sophisticated dependency management

**Options:**
- **UV** - Fast, modern Python package installer
- **Poetry** - Popular dependency management with lock files
- **pip + requirements.txt** - Simple, universally compatible
- **pip-tools** - Compile requirements with dependency resolution

**Decision Needed:**
- Which approach provides best developer experience?
- How important is lock file dependency pinning?
- What's the team's familiarity with different tools?

### 2. Module Organization Strategy
**Question:** How should we organize the agents and tools modules?

**Context from Spec:**
- Six distinct agents: DataHarvester, AddressProfiler, RiskScorer, SybilDetector, ReputationAggregator, Reporter
- Multiple tool adapters: EthProviderTool, EtherscanTool, DefiLlamaTool, LabelRegistry

**Options:**
- **Flat structure** - All agents in single `agents/` directory
- **Hierarchical** - Group by functionality (e.g., `agents/analysis/`, `agents/data/`)
- **Feature-based** - Group by business domain (e.g., `risk/`, `sybil/`, `data/`)

**Decision Needed:**
- How granular should the module separation be?
- Should tools and agents that work together be co-located?
- How will this scale as the system grows?

### 3. Configuration Architecture
**Question:** How should configuration and secrets be managed?

**Context from Spec:**
- Multiple API keys required (RPC_URL, ETHERSCAN_API_KEY, ALCHEMY_API_KEY, OPENAI_API_KEY)
- Different environments (development, testing, production)
- Dependency injection mentioned in spec

**Options:**
- **Simple .env** - Single environment file with python-dotenv
- **Hierarchical config** - Environment-specific configuration files
- **External config** - Configuration from environment/cloud config services
- **Mixed approach** - Local .env for development, external for production

**Decision Needed:**
- How should secrets be handled in different environments?
- Should configuration be validated at startup?
- How complex should environment-specific overrides be?

### 4. Testing Strategy Foundation
**Question:** What testing architecture should we establish?

**Context from Spec:**
- Unit tests mentioned for each agent
- Integration tests for full pipeline
- Golden cases for known good/bad addresses

**Options:**
- **Simple pytest** - Basic unit testing with minimal fixtures
- **Comprehensive fixtures** - Rich test data and mock objects
- **Integration-focused** - Emphasis on end-to-end testing
- **TDD approach** - Test-first development methodology

**Decision Needed:**
- How much test infrastructure should be built upfront?
- Should we mock external APIs or test against real data?
- What's the balance between unit and integration tests?

### 5. CLI and Execution Interface
**Question:** How should users interact with the ReputeNet system?

**Context from Spec:**
- Spec shows command-line execution: `python -m src.graph`
- System should be usable by other developers and potentially deployed

**Options:**
- **Module execution** - `python -m reputenet`
- **CLI script** - `reputenet` command via entry points
- **API server** - HTTP API for programmatic access
- **Multiple interfaces** - CLI, API, and library usage

**Decision Needed:**
- What's the primary usage pattern?
- Should we support both programmatic and command-line usage?
- How important is discoverability vs simplicity?

---

## Secondary Design Questions

### 6. Error Handling Strategy
**Question:** How should errors and retries be handled across the system?

**Considerations:**
- External API failures and rate limiting
- Blockchain data inconsistencies
- LLM API timeouts and errors
- Spec mentions "idempotent retries at each node"

### 7. Logging and Observability
**Question:** What level of logging and monitoring should be built in?

**Context from Spec:**
- Structured logging with structlog mentioned
- Duration tracking per node
- Address and node context needed

### 8. Extensibility Architecture
**Question:** How should the system be designed for future extensions?

**Context from Spec:**
- Extension ideas include multi-chain, entity aggregation, ZK proofs
- System should be modular and swappable

---

## Recommended Decisions

### High Confidence Recommendations

1. **Use src/ layout** - Industry standard, better testing isolation
2. **UV for package management** - Fast, modern, specified in original spec
3. **Structured logging from start** - Required for production monitoring
4. **Environment-based configuration** - Essential for multi-environment deployment

### Decisions Requiring Input

1. **Module organization granularity** - Depends on team preferences and growth plans
2. **Testing mock strategy** - Depends on available test data and API quotas
3. **CLI vs API priority** - Depends on primary use cases and users
4. **Configuration complexity** - Depends on deployment requirements

---

## Questions for Stakeholders

### Technical Lead Questions
1. What's the team's preferred package management approach?
2. How important is backwards compatibility vs using latest Python features?
3. What's the expected deployment model (container, serverless, bare metal)?

### Product Questions
1. Who are the primary users of this system?
2. Is this a library, service, or both?
3. What's the timeline for multi-chain expansion?

### Operations Questions
1. How will this be deployed and monitored in production?
2. What are the performance and availability requirements?
3. How should secrets and configuration be managed?

---

## Next Steps

1. **Gather stakeholder input** on critical decisions
2. **Review team preferences** for tooling and conventions
3. **Validate assumptions** about deployment and usage patterns
4. **Document final decisions** in implementation notes
5. **Proceed with structure creation** based on confirmed decisions