# Project Breakdown Methodology

**Version:** 1.0
**Purpose:** Systematic approach for breaking down software projects from design documents into executable development phases
**Scope:** Reusable methodology for any complex software project with initial design specification

---

## Overview

This methodology transforms high-level design documents into structured, executable development plans through a systematic breakdown process. It ensures comprehensive planning, clear accountability, and reduces implementation risk by front-loading design decisions and research.

## Methodology Steps

### 1. Design Document Analysis
**Input:** Initial design specification or requirements document
**Output:** Understanding of scope, architecture, and success criteria

**Process:**
- Read and analyze the complete design document
- Identify key components, dependencies, and constraints
- Extract success criteria and non-functional requirements
- Note any ambiguities or areas requiring clarification

### 2. High-Level Phase Planning
**Input:** Design analysis
**Output:** `project_plan.md` with 3-6 major development phases

**Process:**
- Break project into logical development phases (typically 3-6 phases)
- Each phase should have clear deliverables and dependencies
- Phases should build incrementally toward final system
- Define phase completion criteria and success metrics

**Typical Phase Pattern:**
1. **Foundation** - Core infrastructure, dependencies, basic structure
2. **Infrastructure** - Data models, core services, orchestration
3. **Feature Implementation** - Core business logic and components
4. **Integration & Polish** - Testing, documentation, production readiness

### 3. Phase Step Breakdown
**Input:** High-level phases
**Output:** `phases/{phase_name}/steps_overview.md` for each phase

**Process:**
- For each phase, create a subfolder: `phases/{phase_name}/`
- Break phase into 3-8 concrete steps
- Each step should be independently implementable
- Steps should have clear inputs, outputs, and completion criteria
- Create `steps_overview.md` listing all steps with brief descriptions

### 4. Step Research & Approach Documentation
**Input:** Phase steps
**Output:** `phases/{phase_name}/step{N}_{step_name}/` folders with research documents

**Process:**
For each step, create a dedicated folder containing:

#### 4.1 `approach_analysis.md`
- Research different implementation approaches
- Compare pros/cons of each approach
- Recommend preferred approach with justification
- List any dependencies or prerequisites

#### 4.2 `design_questions.md`
- Document any unclear aspects from the original design
- List specific technical decisions that need clarification
- Identify potential risks or blockers
- Note areas where additional research is needed

#### 4.3 `implementation_notes.md`
- Technical implementation details
- Code structure recommendations
- Library/framework choices
- Performance considerations

#### 4.4 **CRITICAL: Design Question Resolution**
**Before proceeding to Phase Runbook creation:**

1. **Consolidate all design questions** from the phase into a summary document
2. **Categorize questions** by urgency (blocking vs deferrable)
3. **Gather stakeholder input** on critical decisions
4. **Document final decisions** with rationale
5. **Update implementation notes** to reflect confirmed choices

**⚠️ WARNING: No phase should proceed to implementation without resolving all critical design questions. Unresolved decisions create technical debt and potential rework.**

### 5. Phase Runbook Creation
**Input:** All step research documents for a phase
**Output:** `phases/{phase_name}/phase_{name}_runbook.md`

**Process:**
- Synthesize all step research into executable instructions
- Provide step-by-step implementation guide
- Include code examples, configuration templates
- Add troubleshooting and validation steps
- Reference external documentation and resources

---

## File Structure Template

```
docs/
├── project_breakdown_methodology.md (this document)
├── {project_name}_project_plan.md
└── phases/
    ├── phase1_{name}/
    │   ├── steps_overview.md
    │   ├── step1_{name}/
    │   │   ├── approach_analysis.md
    │   │   ├── design_questions.md
    │   │   └── implementation_notes.md
    │   ├── step2_{name}/
    │   │   └── ...
    │   └── phase1_{name}_runbook.md
    ├── phase2_{name}/
    │   └── ...
    └── phase{N}_{name}/
        └── ...
```

---

## Benefits

### Risk Reduction
- Front-loads design decisions and clarifications
- Identifies blockers before implementation begins
- Ensures comprehensive understanding of requirements

### Clear Accountability
- Each step has defined owners and completion criteria
- Progress can be tracked at granular level
- Dependencies are explicitly documented

### Knowledge Preservation
- Design decisions and rationale are documented
- Research findings are preserved for future reference
- Methodology can be reused across projects

### Implementation Efficiency
- Developers have clear, researched instructions
- Reduces context-switching and decision paralysis
- Enables parallel work on independent steps

---

## Usage Guidelines

### When to Apply
- Complex projects with multiple components/phases
- Projects with significant architectural decisions
- Projects involving multiple team members
- Projects with unclear or evolving requirements

### When to Skip
- Simple, well-understood implementations
- Proof-of-concept or throwaway prototypes
- Projects with extremely tight deadlines
- Single-developer projects with familiar technology

### Customization
- Adjust number of phases based on project complexity
- Modify step granularity based on team experience
- Add project-specific document templates as needed
- Include additional research areas (security, performance, etc.)

---

## Success Criteria

A successful application of this methodology should result in:

1. **Complete Understanding** - All team members understand the full scope and approach
2. **Clear Implementation Path** - Each phase has actionable, researched instructions
3. **Risk Mitigation** - Major technical risks and design questions are addressed upfront
4. **Efficient Execution** - Implementation proceeds smoothly with minimal blockers
5. **Knowledge Preservation** - Decisions and research are documented for future reference

---

## Meta-Application

This methodology can be applied to:
- Web applications and APIs
- Data processing pipelines
- Machine learning systems
- Infrastructure projects
- Integration projects
- Migration projects

The key is adapting the phase structure and research focus areas to match the specific domain and technology stack of the project.