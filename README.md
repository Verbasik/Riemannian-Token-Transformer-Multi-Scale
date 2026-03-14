<div align="center">

# 🧠 Memora

**Production memory architecture for AI coding agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Standard: AGENTS.md](https://img.shields.io/badge/Standard-AGENTS.md-black?logo=markdown&logoColor=white)](./AGENTS.md)
[![Status](https://img.shields.io/badge/Status-InProgress-0f766e?style=flat-square)](.)
[![Built with](https://img.shields.io/badge/Built_with-Markdown-blue?logo=markdown&logoColor=white)](.)

<p>
  <strong>Progressive context loading</strong> ·
  <strong>Canonical knowledge ownership</strong> ·
  <strong>Cross-tool compatibility</strong>
</p>

### Supported AI Toolchains

[![Claude Code](https://img.shields.io/badge/Claude_Code-✓-7c3aed?style=flat)](.)
[![Codex CLI](https://img.shields.io/badge/Codex_CLI-✓-0f172a?style=flat)](.)
[![Qwen Code](https://img.shields.io/badge/Qwen_Code-✓-0369a1?style=flat)](.)
[![OpenCode](https://img.shields.io/badge/OpenCode-✓-065f46?style=flat)](.)

---

</div>

## 📌 Why Memora?

**Memora** transforms chaotic project context into a **managed, routable, and verifiable knowledge architecture** for AI agents.

Instead of loading the entire memory bank at startup, Memora uses **progressive disclosure**:

- **Agent** reads minimal required context
- **No duplicate facts** across files — canonical ownership
- **Predictable behavior** across different tools
- **Sustainable engineering memory**, not noise

> When AI agents work on long-lived projects, they need structured memory, not just context windows.

---

## 🎯 What is Memora?

A **cross-tool production memory layer** for engineering repositories where AI agents work as **participants in long-lived development**, not stateless helpers.

### How It Works

```
Agent starts
    ↓
Reads: AGENTS.md
    ↓
Checks: memory-bank/INDEX.md
    ↓
Loads: Only relevant files
    ↓
Executes: Task
    ↓
Updates: CURRENT.md, HANDOFF.md
    ↓
Promotes: Durable insights → Stable files
```

### Three outcomes

- ⬇️ **Fewer tokens** in context
- ⬆️ **Consistent** responses
- ✨ **Better** long-term agent work

---

## 🔑 Core Principles

| Principle | Meaning |
|-----------|---------|
| 🎯 **Canonical ownership** | One fact lives in one place |
| 📦 **Progressive loading** | Load only what you need |
| ⏰ **Temporal metadata** | Every fact has a verification date |
| 🔌 **Thin adapters** | Tool-specific files are adapters, not duplicates |
| 🔒 **Session isolation** | Session context lives in `.local/`, not mixed with stable knowledge |
| 📋 **Explicit decisions** | Architecture decisions go to `DECISIONS.md` + `ADR/` |
| 🛡️ **Security-first** | No secrets, tokens, or PII in memory |
| ⚖️ **Constitution-first** | All changes must respect `CONSTITUTION.md` |

---

## 🏗️ Architecture

```
AGENTS.md (canonical instructions)
    ↓
memory-bank/INDEX.md (routing layer)
    ├→ PROJECT.md (identity & scope)
    ├→ ARCHITECTURE.md (system design)
    ├→ CONVENTIONS.md (engineering rules)
    ├→ TESTING.md (validation flow)
    ├→ DECISIONS.md + ADR/ (why decisions exist)
    ├→ AREAS/ (subsystem knowledge)
    ├→ PATTERNS/ (reusable techniques)
    ├→ .local/CURRENT.md (session state)
    └→ .local/HANDOFF.md (next-session briefing)
         ↓
    Promotion pipeline → stable files
```

---

## 📂 Repository Structure

```
memora/
├── AGENTS.md                 # ⭐ Entry point for all agents
├── CLAUDE.md                 # Claude Code adapter
├── MANIFESTO.md              # Standard specification
├── memory-bank/
│   ├── INDEX.md              # Routing table (what to read when)
│   ├── CONSTITUTION.md       # Inviolable principles
│   ├── PROJECT.md            # Project identity & scope
│   ├── ARCHITECTURE.md       # System design
│   ├── CONVENTIONS.md        # Code style & naming
│   ├── TESTING.md            # Test strategy
│   ├── DECISIONS.md          # Decision registry
│   ├── CHANGELOG.md          # Significant milestones
│   ├── LIFECYCLE.md          # Memory operations flow
│   ├── OPEN_QUESTIONS.md     # Unresolved issues
│   ├── ADR/                  # Architecture Decision Records
│   ├── PATTERNS/             # Reusable techniques
│   ├── AREAS/                # Subsystem knowledge
│   ├── .local/               # 🚫 gitignored
│   │   ├── CURRENT.md        # Current session state
│   │   ├── HANDOFF.md        # Next session briefing
│   │   └── SESSIONS/         # Session logs
│   └── ARCHIVE/              # Retired sessions
├── .claude/                  # Claude Code integration
├── .codex/                   # Codex CLI integration
├── .qwen/                    # Qwen Code integration
├── .opencode/                # OpenCode integration
└── bin/memora.js             # CLI tool
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js `>=16`
- `bash` (macOS/Linux; Windows: Git Bash/WSL)

### 1️⃣ Initialize CLI

```bash
npm install -g ./memora-cli-X.X.X.tgz
# or for development
npm link
```

### 2️⃣ Bootstrap a New Project

```bash
memora init ./my-project
cd my-project
```

### 3️⃣ Fill Core Context

Update these files with your project details:

- `memory-bank/PROJECT.md` — What is this project?
- `memory-bank/ARCHITECTURE.md` — How does it work?
- `memory-bank/TESTING.md` — How do we validate?
- `memory-bank/CONVENTIONS.md` — How do we write code?

### 4️⃣ Connect Your AI Tool

**Claude Code:**
```bash
claude .
```

**Codex CLI:**
```bash
codex --trust-project
```

**Qwen Code:**
Update `.qwen/settings.json` with `AGENTS.md` in context files.

---

## 🛠️ Memory Skills

Memora includes **6 operational skills** to maintain memory integrity:

### 🔧 `memory-bootstrap`
**First-run initialization** — Explores your project and fills `PROJECT.md`, `ARCHITECTURE.md`, etc.
- Run once after `memora init`
- Auto-detects stack, modules, tests
- Proposes `CONSTITUTION.md` principles for human review

### 📝 `update-memory`
**Session finalization** — Updates after task completion
- Refreshes `CURRENT.md` (session state)
- Refreshes `HANDOFF.md` (next-session briefing)
- Promotes durable insights to `DECISIONS.md`, `PATTERNS/`

### 🔍 `memory-audit`
**Integrity check** — Weekly or before major tasks
- Detects stale verifications
- Finds architectural drift
- Checks for orphaned decisions
- Scans for credential leaks

### 🔗 `memory-consolidate`
**Multi-session promotion** — Weekly or after several sessions
- Moves session notes → stable files
- Reduces duplicates and drift

### 🧹 `memory-gc`
**Cleanup** — Monthly or when `SESSIONS/` > 20 files
- Archives old sessions
- Compacts `CURRENT.md`
- Cleans replaced solutions

### ❓ `memory-clarify`
**Gap analysis** — When audit finds issues or before major features
- Analyzes missing knowledge
- Checks decision consistency
- Generates targeted questions for `OPEN_QUESTIONS.md`

---

## 📊 Compatibility Matrix

| Feature | Claude Code | Codex CLI | Qwen Code | OpenCode |
|---------|:----------:|:---------:|:---------:|:--------:|
| Instructions | `CLAUDE.md` | Native | `settings.json` | Native |
| Skills/Commands | `.claude/skills/` | `.codex/skills/` | `.qwen/agents/` | `.opencode/commands/` |
| Memory files | ✅ Shared | ✅ Shared | ✅ Shared | ✅ Shared |
| Security mode | `permissions.deny` | Sandbox | `.qwenignore` | Patterns |

---

## ⚡ Recommended Workflow

```
1. Start task → 2. Read AGENTS.md → 3. Check INDEX.md
    ↓
4. Load minimal files → 5. Execute work
    ↓
6. Update CURRENT.md → 7. Write HANDOFF.md
    ↓
8. Durable insight? → YES: Promote to DECISIONS/ADR/PATTERNS
    ↓
9. Finish session
```

**Golden rule:** Don't load all memory bank. Load only what you need. Promote only durable knowledge.

---

## 🔐 Security First

Memora is designed with **memory hygiene by default**.

### Never stored

- 🚫 API keys
- 🚫 Access tokens
- 🚫 Passwords
- 🚫 Raw credentials
- 🚫 PII

### Instead, reference by name

```bash
$DATABASE_URL
$OPENAI_API_KEY
$JWT_SECRET
```

### Built-in safeguards

- `.claudeignore` blocks sensitive files
- `audit` skill scans for credential patterns
- `CONSTITUTION.md` governs what can be recorded

---

## 🎓 Why This Works

### Predictability
Every knowledge class has an owner and clear read path

### Scalability
Handles growth in sessions, decisions, subsystems, and agents

### Compatibility
Same knowledge architecture works across AI toolchains

### Observability
`CURRENT.md`, `HANDOFF.md`, `SESSIONS/`, `DECISIONS.md` form an audit trail

### Context hygiene
Memora reduces noise and prevents instruction dilution

---

## 📚 Documentation

- **[MANIFESTO.md](./MANIFESTO.md)** — Open standard for agent markdown instructions
- **[AGENTS.md](./AGENTS.md)** — Bootstrap and operating rules
- **[memory-bank/INDEX.md](./memory-bank/INDEX.md)** — Routing guide
- **[memory-bank/LIFECYCLE.md](./memory-bank/LIFECYCLE.md)** — Memory operations flow

---

## 🗺️ Roadmap

- [ ] Monorepo and multi-service templates
- [ ] Starter packs (TypeScript, Python, Go, Rust)
- [ ] Auto-generator from existing codebases
- [ ] CI checks for temporal freshness and drift
- [ ] Visual dashboard for memory health

---

## 📄 License

MIT — Use freely. See [LICENSE](./LICENSE) for details.

---

<div align="center">

**Memora** — Made with love in Russia ❤️ 

[Star on GitHub](https://github.com/) · [Read Manifesto](./MANIFESTO.md) · [Quick Start](#-quick-start)

</div>
