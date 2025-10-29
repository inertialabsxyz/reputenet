# ReputeNet — Multi‑Agent System for On‑Chain Reputation (LangChain + Python)

**Owner:** Head of Engineering, Movement Labs  
**Date:** 2025‑10‑29  
**Goal:** Prototype a **multi‑agent system** using **LangChain/LangGraph** to compute an **on‑chain reputation profile** for a wallet or contract address. Output is a structured JSON report plus a human‑readable summary.

---

## 1) Scope & Success Criteria

**In‑scope (MVP):**
- Input: one or more EVM addresses.
- Fetch recent activity + basic graph context (top counterparties, token interactions, contract calls).
- Score: activity quality, contract risk, social proximity, anti‑sybil heuristics.
- Produce: (a) JSON reputation object, (b) Markdown summary.

**Out‑of‑scope (MVP):** multi‑chain expansion, advanced ML community detection, ZK proofs.

**Success criteria:**
- End‑to‑end run completes under 90s for a single address with cached data.
- Deterministic orchestration (LangGraph) with idempotent retries at each node.
- Clear boundaries between data, reasoning, and scoring modules.

---

## 2) Architecture Overview

**Design principles:**
- **Graph‑orchestrated agents** using **LangGraph** (deterministic, resumable).
- **Separation of concerns:** data collection → feature extraction → analysis → scoring → reporting.
- **Tool adapters** isolate external APIs (Etherscan/Alchemy, DefiLlama, Dune optional).
- **Typed state** shared across nodes; immutable inputs, incremental enrichments.

```
┌───────────────────────────┐
│         Orchestrator      │  (LangGraph: controls flow, retries)
└─────────────┬─────────────┘
              │
   ┌──────────▼───────────┐
   │    DataHarvester     │  → RPC/Indexers (Etherscan/Alchemy)
   └──────────┬───────────┘
              │ txs, logs
   ┌──────────▼───────────┐
   │   AddressProfiler    │  → Features: age, diversity, dex use
   └──────────┬───────────┘
              │ features
   ┌──────────▼───────────┐
   │     RiskScorer       │  → Heuristics: mixers, scams, anomalies
   └──────────┬───────────┘
              │ risk
   ┌──────────▼───────────┐
   │    SybilDetector     │  → Graph: clustering, temporal bursts
   └──────────┬───────────┘
              │ sybil signals
   ┌──────────▼───────────┐
   │ ReputationAggregator │  → Weighted composite + explanation
   └──────────┬───────────┘
              │
   ┌──────────▼───────────┐
   │     Reporter         │  → JSON + Markdown
   └──────────────────────┘
```

---

## 3) Agents & Responsibilities

### 3.1 DataHarvester (Tool‑using Agent)
- **Inputs:** target addresses, chain id.
- **Outputs:** normalized transactions, logs, counterparties, token standards involved (ERC‑20/721/1155), first/last seen timestamps.
- **Tools:** `EthProviderTool`, `EtherscanTool` (rate‑limited), `DefiLlamaTool` (optional for protocol metadata).

### 3.2 AddressProfiler (LLM + Python)
- **Purpose:** derive features from raw on‑chain data.
- **Features (examples):**
  - Account age, inter‑transaction intervals, protocol diversity index.
  - Volume/size buckets, DEX/bridge interactions, contract deployment flag.
  - Counterparty uniqueness ratio (unique addrs / total txs).

### 3.3 RiskScorer (Rule‑first, LLM‑aided)
- **Heuristics:**
  - Interactions with flagged contracts (mixers, phishing lists).
  - High MEV likelihood patterns (sandwich signatures).
  - Bridge anomalies, unusually high revert rates, approval‑for‑all risks.
- **Output:** `risk_score` (0–100), evidence list.

### 3.4 SybilDetector (Graph + Stats)
- **Signals:**
  - Burstiness (many small txs/short windows), common funding source, shared spenders.
  - Address similarity via n‑gram / edit distance in ENS names (if present).
  - Covariance of activity time‑of‑day vs control set.

### 3.5 ReputationAggregator (LLM‑assisted weighting)
- Combine: profile features + risk + sybil signals.
- Produce: `reputation_score` (0–100) and rationale with transparent component weights.

### 3.6 Reporter (LLM)
- Create human summary + structured JSON per schema (see §4).

---

## 4) Data & State Schemas

### 4.1 Input
```json
{
  "chain_id": 1,
  "targets": ["0xabc...", "0xdef..."],
  "lookback_days": 90,
  "max_txs": 2000
}
```

### 4.2 Orchestrator State (typed)
```ts
State = {
  params: { chain_id: number; targets: string[]; lookback_days: number; max_txs: number },
  raw: { [addr: string]: { txs: Tx[]; logs: Log[]; tokens: TokenTouch[] } },
  features: { [addr: string]: AddressFeatures },
  risk: { [addr: string]: RiskOutput },
  sybil: { [addr: string]: SybilOutput },
  reputation: { [addr: string]: ReputationOutput },
  reports: { [addr: string]: { json: any; markdown: string } }
}
```

### 4.3 Output JSON (per address)
```json
{
  "address": "0x...",
  "reputation_score": 74,
  "components": {
    "activity_quality": 0.62,
    "protocol_diversity": 0.71,
    "counterparty_uniqueness": 0.68,
    "risk": 0.15,
    "sybil": 0.22
  },
  "flags": ["approval_for_all_to_unknown", "bridge_anomaly:hop"]
}
```

---

## 5) Prompts (System Messages)

> **DataHarvester**  
> “You are a data acquisition toolrunner. When asked for on‑chain activity, call the provided tools, return **concise normalized JSON**. Never speculate.”

> **AddressProfiler**  
> “You convert raw on‑chain activity into **interpretable features** that are stable across runs. Avoid labels; output typed fields only.”

> **RiskScorer**  
> “You evaluate operational and counterparty risk using explicit heuristics. Explain each penalty briefly. Output `risk_score` 0–100 and `evidence`.”

> **SybilDetector**  
> “You detect likely sybil behavior using graph and temporal statistics. Prefer interpretable metrics over black‑box predictions.”

> **ReputationAggregator**  
> “You combine component scores using **transparent weights** and produce a final score with a rationale. Output JSON + a short explanation.”

> **Reporter**  
> “You produce a **crisp Markdown report** for humans and a machine‑readable JSON block.”

---

## 6) Tools & Adapters (Interfaces)

> All adapters live in `src/tools/` and expose a minimal, swappable interface.

```python
# src/tools/eth_provider.py
from typing import List, Dict, Any

class EthProviderTool:
    def __init__(self, rpc_url: str, api_key: str | None = None): ...
    def get_normalized_history(self, address: str, lookback_days: int, max_txs: int) -> Dict[str, Any]: ...
    def get_logs(self, address: str, from_block: int | None = None, to_block: int | None = None) -> List[Dict[str, Any]]: ...
```

```python
# src/tools/defillama.py
class DefiLlamaTool:
    def get_protocol_metadata(self, contract: str) -> dict: ...
```

```python
# src/tools/labels.py  (optional; local lists)
class LabelRegistry:
    def is_flagged_contract(self, address: str) -> bool: ...
    def get_label(self, address: str) -> str | None: ...
```

---

## 7) LangGraph Orchestration (Code Skeleton)

```python
# src/graph.py
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class GraphState(TypedDict):
    params: Dict[str, Any]
    raw: Dict[str, Any]
    features: Dict[str, Any]
    risk: Dict[str, Any]
    sybil: Dict[str, Any]
    reputation: Dict[str, Any]
    reports: Dict[str, Any]

llm = ChatOpenAI(model="gpt-4o-mini")  # replace as needed

# --- Node functions ---

def node_data_harvest(state: GraphState) -> GraphState:
    params = state["params"]
    # TODO: instantiate EthProviderTool from DI container
    # For each address, fetch and normalize history
    raw = {}
    for addr in params["targets"]:
        raw[addr] = {
            "txs": [],  # provider.get_normalized_history(...)
            "logs": [],
            "tokens": []
        }
    state["raw"] = raw
    return state


def node_profile(state: GraphState) -> GraphState:
    features = {}
    for addr, bundle in state["raw"].items():
        features[addr] = {
            "account_age_days": 420,
            "protocol_diversity": 0.7,
            "counterparty_uniqueness": 0.65,
        }
    state["features"] = features
    return state


def node_risk(state: GraphState) -> GraphState:
    risk = {}
    for addr, feats in state["features"].items():
        risk[addr] = {"risk_score": 18, "evidence": ["low revoke approvals"]}
    state["risk"] = risk
    return state


def node_sybil(state: GraphState) -> GraphState:
    sybil = {}
    for addr in state["features"].keys():
        sybil[addr] = {"sybil_score": 20, "signals": ["no burstiness"]}
    state["sybil"] = sybil
    return state


def node_aggregate(state: GraphState) -> GraphState:
    reputation = {}
    for addr in state["features"].keys():
        activity = 0.6
        risk_component = max(0, 1 - state["risk"][addr]["risk_score"]/100)
        sybil_component = max(0, 1 - state["sybil"][addr]["sybil_score"]/100)
        final = round(100*(0.5*activity + 0.3*risk_component + 0.2*sybil_component))
        reputation[addr] = {
            "reputation_score": final,
            "components": {
                "activity_quality": activity,
                "risk": 1 - state["risk"][addr]["risk_score"]/100,
                "sybil": 1 - state["sybil"][addr]["sybil_score"]/100,
            }
        }
    state["reputation"] = reputation
    return state


def node_report(state: GraphState) -> GraphState:
    reports = {}
    for addr, rep in state["reputation"].items():
        md = f"""### Address {addr}\n\n- Reputation: **{rep['reputation_score']}**/100\n- Components: {rep['components']}\n\n"""
        reports[addr] = {"json": rep, "markdown": md}
    state["reports"] = reports
    return state


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("harvest", node_data_harvest)
    g.add_node("profile", node_profile)
    g.add_node("risk", node_risk)
    g.add_node("sybil", node_sybil)
    g.add_node("aggregate", node_aggregate)
    g.add_node("report", node_report)

    g.set_entry_point("harvest")
    g.add_edge("harvest", "profile")
    g.add_edge("profile", "risk")
    g.add_edge("risk", "sybil")
    g.add_edge("sybil", "aggregate")
    g.add_edge("aggregate", "report")
    g.add_edge("report", END)
    return g

if __name__ == "__main__":
    graph = build_graph()
    initial = {
        "params": {"chain_id": 1, "targets": ["0xabc..."], "lookback_days": 90, "max_txs": 2000},
        "raw": {}, "features": {}, "risk": {}, "sybil": {}, "reputation": {}, "reports": {}
    }
    final_state = graph.compile().invoke(initial)
    print(final_state["reports"]["0xabc..."]["markdown"])  # demo
```

**Notes:**
- Replace placeholder computations with real logic as adapters land.
- Each node should be **pure** (input state → output state) and retryable.

---

## 8) Evaluation Plan

- **Unit checks** per node (e.g., profile features monotonicity, no NaNs).
- **Golden cases:** known good/bad addresses to calibrate weights.
- **Robustness:** randomize lookback window; reputation score variance < ±5.
- **Latency budget:** report per address in < 5s with warm cache.

---

## 9) Configuration & Secrets

`.env` keys (example):
```
RPC_URL_MAINNET=
ETHERSCAN_API_KEY=
ALCHEMY_API_KEY=
OPENAI_API_KEY=
```

Use a small **dependency injector** (`src/di.py`) so tools are easily swapped.

---

## 10) Project Structure

```
reputenet/
  ├─ src/
  │   ├─ graph.py
  │   ├─ agents/
  │   │   ├─ profiler.py
  │   │   ├─ risk.py
  │   │   ├─ sybil.py
  │   │   └─ reporter.py
  │   ├─ tools/
  │   │   ├─ eth_provider.py
  │   │   ├─ defillama.py
  │   │   └─ labels.py
  │   ├─ di.py
  │   └─ schema.py
  ├─ tests/
  │   ├─ test_graph.py
  │   └─ test_features.py
  ├─ README.md
  └─ pyproject.toml
```

---

## 11) Heuristics (Initial Set)

- **Activity Quality (0–1):** scaled by tx frequency, regularity, DEX/bridge diversity, successful tx ratio.
- **Counterparty Uniqueness (0–1):** `unique_counterparties / total_txs` (clipped).
- **Risk:** +10 for any mixer interaction; +5 for flagged contract; +1 per high‑risk approval; −5 if timely revoke.
- **Sybil:** +10 if >N txs in 1m bursts; +5 if funded by a common hub; +5 if shared spenders across N targets.

Weights (MVP): `final = 100*(0.5*activity + 0.3*(1-risk) + 0.2*(1-sybil))`.

---

## 12) Runbook

**Local:**
```
uv pip install -r requirements.txt  # or: pip install -r requirements.txt
python -m src.graph
```

**Caching:**
- Use `diskcache` or `sqlite` layer in `EthProviderTool` for tx pages.

**Observability:**
- Structured logging with `structlog` per node; include address, node, duration.

---

## 13) Requirements (initial)

```
langchain>=0.2
langgraph>=0.2
langchain-openai>=0.1
web3>=6.0
httpx>=0.27
pydantic>=2.7
structlog>=24.1
diskcache>=5.6
python-dotenv>=1.0
```

---

## 14) Extension Ideas (Post‑MVP)

- **Multi‑chain:** add chain routers; dedupe identity via bridges + ENS.
- **Entity aggregation:** cluster addresses into entities via funding/spend flows.
- **Reputation attestations:** emit EAS/AttestationStation badges.
- **DAO governance weight:** integrate Snapshot/Compound histories.
- **ZK proofs of reputation components (selective disclosure).**

---

## 15) Open Questions

- Do we adopt a public label source (Chainabuse, TRM public lists) or only local labels in MVP?
- What is the right default lookback (30/90/180 days) for stable scores?
- Should reputation be **address‑level** or **entity‑level** from Day 1?

---

## 16) Acceptance Checklist (MVP)

- [ ] Single address run produces JSON + Markdown.
- [ ] Nodes are pure and retryable; graph compile works.
- [ ] Basic heuristics implemented and unit‑tested.
- [ ] `.env` driven tool configuration.
- [ ] README with setup and example output.

