# unfolding — Agent Instructions

Python utility to unfold band structures (phonons, electrons, magnons) from supercell calculations into the primitive-cell representation.

## Kosmic Integration

This project uses **Kosmic** for persistent memory and knowledge management.

### Configuration
- **Vault**: `/home/hexu/projects/memnotes/memnotes`
- **Target**: `unfolding` (vault Project folder)
- **Project ID**: `unfolding` (immutable)
- **Usage Skill**: `unfolding-usage`
- **Development Skill**: `unfolding-development`
- **Canonical Pair**: `/home/hexu/projects/memnotes/memnotes/Projects/unfolding/skills/`

### Skills

The `kosmic` skill is the workflow orchestrator. Contributor work loads `unfolding-development`, which in turn requires `unfolding-usage` for consumer and shared behavior. Consumers may invoke `unfolding-usage` directly. If named skills are unsupported, persistent instructions must point to both canonical packages.

### Project Knowledge Format

`/home/hexu/projects/memnotes/memnotes/Projects/unfolding/notes/` is the authoritative Obsidian-compatible OKF v0.2 bundle. Use `index.md` for discovery. Concept notes require YAML frontmatter on line 1, non-empty `type`, `status: draft | stable | deprecated`, and relative Markdown links. Do not create or hand-maintain a second OKF export.

### When to Use Kosmic Tasks

#### Use Task: Initialize
**When**: The vault Project, immutable Project Identifier, conformant Knowledge bundle, or canonical Skill Pair is missing
**Reference**: `references/task_init.md`
**Check**: Does `/home/hexu/projects/memnotes/memnotes/Projects/unfolding/ProjectEntry.md` define `unfolding`, does `notes/index.md` declare OKF v0.2, and does `skills/` contain both valid projections?

#### Use Task: Search
**When**: 
- BEFORE implementing new features
- BEFORE making architectural changes
- When encountering a problem
- When unsure how something works

**Reference**: `references/task_search.md`

#### Use Task: Quick Task
**When**:
- Small, low-risk implementation work
- Concrete fix, narrow change, or simple addition
- Scope and acceptance criteria are clear
- Full PRD → optional Research → Architecture → Epics → Stories workflow would be disproportionate

**Do not use when**: The task introduces architecture changes, data models, public APIs, security-sensitive behavior, broad user-facing workflows, or unclear requirements.

**Workflow**: Search → Quick Task record → Approval → Implement/verify → Review → Log/notes

**Reference**: `references/task_quick_task.md`

#### Use Task: Add Note
**When**:
- **Any time knowledge is found** (new OR existing - document what you learn)
- Reading code to understand a system, algorithm, or pattern
- Writing documentation (capture core concepts as notes too)
- Made an architectural decision
- Found a solution worth remembering
- Validating or re-verifying existing knowledge
- At end of session to log progress

**Key Principle**: Document knowledge whenever you FIND it, not just when you CREATE it.

**Reference**: `references/task_add_note.md`

#### Use Task: Review
**When**:
- End of work session (quick daily review)
- End of week (weekly review)
- Knowledge seems outdated
- Returning to project after absence

**Reference**: `references/task_review.md`

#### Use Task R: Research
**When**:
- Optional discovery is needed after PRD and before Architecture
- Task Search found that existing notes/specs are not enough
- Technical feasibility, library choice, or architecture direction is uncertain

**Workflow**: Search → Research memo → Findings/recommendation → Link from Architecture

**Gate**: No new approval gate. Research feeds PRD or Architecture review.

**Reference**: `references/task_research.md`

#### Use Task: Create Spec (Spec-Driven Development)
**When**: Building a new feature or planning a project

**Workflow** (agent MUST follow in order, pausing at each gate for user approval):
1. Create PRD → **WAIT for user approval**
2. Optional Task R: Research → no new gate; feed findings into Architecture
3. Create Architecture → **WAIT for user approval**
4. Create Epics + Stories → **WAIT for user approval**
5. Implement approved stories (TDD) → **Code review MANDATORY**

**Reference**: `references/task_create_spec.md`

### Planning Workflow

When planning a task, always integrate knowledge retrieval: search notes first, research the codebase for unknowns, add notes BEFORE finalizing a plan, and reference the notes that informed decisions.

### Typical Session

1. **Start**: Search for relevant knowledge
2. **Plan**: Check notes, research if needed, add new notes, include in plan
3. **During**: Add notes whenever you understand something
4. **End**: Review and log progress
