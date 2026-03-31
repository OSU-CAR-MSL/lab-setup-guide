---
tags:
  - Resources
  - AI
---
<!-- last-reviewed: 2026-03-30 -->
# Concept Map & Knowledge Graph

This page provides two visual overviews: a **concept map** showing how topics in this documentation connect (so you can plan your learning path), and a **knowledge graph** demonstrating how AI coding assistants persist structured knowledge across sessions.

## How These Docs Are Organized

The site has eight sections that form a progression:

| Section | Purpose |
|---------|---------|
| **Getting Started** | Local development setup: WSL, VS Code, Python, AI tools |
| **OSC Basics** | Connecting to OSC: accounts, SSH, remote development |
| **Working on OSC** | Day-to-day work: environments, job submission, pipelines |
| **ML Workflows** | Machine learning stack: PyTorch, PyG, experiment tracking |
| **GitHub** | Git fundamentals, repo management, SSH auth, CI/CD, troubleshooting |
| **Contributing** | How to contribute to lab docs, issues, PRs, and project management |
| **Assignments** | Course assignments and project templates |
| **Resources** | Troubleshooting, useful links, and this concept map |
| **Tags** | Browse all pages by topic tag |

The concept map below shows the key prerequisite relationships between pages.

## Docs Concept Map

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#e8f4fd', 'primaryTextColor': '#1a1a1a', 'lineColor': '#555'}}}%%
flowchart TB
  subgraph GS["fa:fa-rocket Getting Started"]
    wsl["fa:fa-linux WSL2 Setup"]:::gs
    vscode["fa:fa-code VS Code Setup"]:::gs
    pyenv["fa:fa-box Python Env Setup"]:::gs
    ai["fa:fa-robot AI Coding Assistants"]:::gs
    agent["fa:fa-wand-magic-sparkles Agent Workflows"]:::gs
    wsl --> vscode --> pyenv
    vscode --> ai --> agent
  end

  subgraph OB["fa:fa-server OSC Basics"]
    account["fa:fa-user-plus Account Setup"]:::osc
    ssh["fa:fa-key SSH Connection"]:::osc
    remote["fa:fa-desktop Remote Development"]:::osc
    account --> ssh --> remote
  end

  subgraph WO["fa:fa-terminal Working on OSC"]
    envmgmt["fa:fa-cubes Environment Mgmt"]:::osc
    jobs["fa:fa-list-check Job Submission"]:::osc
    pipeline["fa:fa-arrows-rotate Pipeline Orchestration"]:::osc
    envmgmt --> jobs
    jobs --> pipeline
  end

  subgraph ML["fa:fa-brain ML Workflows"]
    pytorch["fa:fa-fire PyTorch & GPU Setup"]:::ml
    pyg["fa:fa-diagram-project PyG Setup"]:::ml
    rapids["fa:fa-bolt RAPIDS GPU"]:::ml
    mlwf["fa:fa-folder-open ML Project Template"]:::ml
    tracking["fa:fa-chart-line Data & Experiment Tracking"]:::ml
    duckdb["fa:fa-database DuckDB Analytics"]:::ml
    hf["fa:fa-cloud Hugging Face Spaces"]:::ml
    pytorch --> pyg
    pytorch --> rapids
    mlwf --> tracking --> duckdb
    tracking --> hf
  end

  subgraph GH["fa:fa-code-branch GitHub"]
    gitfund["fa:fa-code-branch Git Fundamentals"]:::gh
    reposetup["fa:fa-folder-plus Repository Setup"]:::gh
    sshauth["fa:fa-key SSH & Authentication"]:::gh
    actions["fa:fa-gears GitHub Actions & CI/CD"]:::gh
    gittrouble["fa:fa-wrench Git Troubleshooting"]:::gh
    gitfund --> reposetup
    gitfund --> gittrouble
    sshauth --> reposetup
    reposetup --> actions
  end

  %% Cross-section prerequisite edges
  pyenv --> envmgmt
  remote --> envmgmt
  jobs --> pytorch
  pytorch --> mlwf
  sshauth --> ssh
  gitfund --> jobs
  actions --> hf

  classDef gs fill:#e8f4fd,stroke:#3b82f6,color:#1a1a1a,stroke-width:2px
  classDef osc fill:#fef3c7,stroke:#d97706,color:#1a1a1a,stroke-width:2px
  classDef ml fill:#ede9fe,stroke:#7c3aed,color:#1a1a1a,stroke-width:2px
  classDef gh fill:#d1fae5,stroke:#059669,color:#1a1a1a,stroke-width:2px
```

## Hub Pages

These five pages are referenced most often across the documentation. If you're looking for something, there's a good chance one of these is the right starting point.

| Page | Role |
|------|------|
| [Job Submission](../working-on-osc/osc-job-submission.md) | Central reference for SLURM scripts, partitions, and job arrays |
| [PyTorch & GPU Setup](../ml-workflows/pytorch-setup.md) | GPU environment setup, CUDA troubleshooting, performance tuning |
| [Environment Management](../working-on-osc/osc-environment-management.md) | Modules, virtual environments, uv, and dependency management |
| [Data & Experiment Tracking](../ml-workflows/data-experiment-tracking.md) | W&B, DVC, and the Parquet datalake pattern |
| [SSH Connection](../osc-basics/osc-ssh-connection.md) | SSH keys, config, ProxyJump, and connection troubleshooting |
| [Git Fundamentals](../github/git-fundamentals.md) | Git mental model, branching, collaborative workflows, worktrees |

## What Is a Knowledge Graph?

A **knowledge graph** stores information as a network of **entities** (things) connected by **relations** (how they relate). Each entity can have **observations** — facts or notes attached to it.

This is different from flat notes or documents:

- **Entities** are named objects with a type (e.g., "PyTorch Setup" of type `infrastructure`)
- **Relations** connect entities directionally (e.g., "MapViz Build" *enables* "DuckDB-WASM Plan")
- **Observations** are free-text facts attached to an entity, like log entries

Knowledge graphs are used in AI systems to give agents **persistent, structured memory**. Claude Code's [MCP memory server](../getting-started/agent-workflows.md) maintains a knowledge graph in `~/.claude/knowledge-graph.json` that persists across coding sessions. When you use the `/save` skill or Claude learns something about your project, it stores that knowledge as entities and relations — so it can recall context in future sessions without re-reading every file.

### NDJSON Format

The knowledge graph is stored as newline-delimited JSON (NDJSON). Each line is either an entity or a relation:

```json
{"type": "entity", "name": "Database Architecture", "entityType": "architecture_decision", "observations": ["KD-GAT and Map-Viz converge on DuckDB + Parquet pattern"]}
{"type": "relation", "from": "Mosaic vgplot API", "to": "KD-GAT", "relationType": "used_by"}
```

## Lab Knowledge Graph

The diagram below is auto-generated from `~/.claude/knowledge-graph.json` using `scripts/generate-kg-mermaid.py`. It shows the 13 entities and their relationships from our lab's Claude Code memory server.

```mermaid
graph TB
    Database_Architecture["Database Architecture"]:::architecture_decision
    Context_Management_Rules["Context Management Rules"]:::convention
    Modular_Context_Architecture["Modular Context Architecture"]:::architecture_decision
    Context_Update_Log["Context Update Log"]:::changelog
    KD_GAT_Shell_Environment["KD-GAT Shell Environment"]:::canonical_answer
    Claude_Config_Best_Practices["Claude Config Best Practices"]:::architecture_decision
    KD_GAT_CI_CD_Pipeline["KD-GAT CI/CD Pipeline"]:::infrastructure
    KD_GAT_Quarto_Site["KD-GAT Quarto Site"]:::infrastructure
    Mosaic_vgplot_API["Mosaic vgplot API"]:::technology
    MapViz_Observable_Build["MapViz-Observable-Build"]:::milestone
    MapViz_DuckDB_WASM_Plan["MapViz-DuckDB-WASM-Plan"]:::architecture_decision
    pyarrow_parquet_stdout["pyarrow-parquet-stdout"]:::learning
    Mosaic_vgplot["Mosaic vgplot"]:::library

    MapViz_Observable_Build -->|enables| MapViz_DuckDB_WASM_Plan
    MapViz_DuckDB_WASM_Plan -->|depends_on| MapViz_Observable_Build

    classDef architecture_decision fill:#4A90D9,stroke:#2A5F9E,color:#fff,stroke-width:2px
    classDef canonical_answer fill:#27AE60,stroke:#1E8449,color:#fff,stroke-width:2px
    classDef changelog fill:#95A5A6,stroke:#717D7E,color:#fff,stroke-width:2px
    classDef convention fill:#9B59B6,stroke:#6C3483,color:#fff,stroke-width:2px
    classDef infrastructure fill:#E67E22,stroke:#BA4A00,color:#fff,stroke-width:2px
    classDef learning fill:#E91E90,stroke:#B3166E,color:#fff,stroke-width:2px
    classDef library fill:#1ABC9C,stroke:#0E8C73,color:#fff,stroke-width:2px
    classDef milestone fill:#F1C40F,stroke:#B7950B,color:#fff,stroke-width:2px
    classDef technology fill:#1ABC9C,stroke:#0E8C73,color:#fff,stroke-width:2px
```

**Legend:**

| Color | Entity Type | Example |
|-------|-------------|---------|
| :blue_square: Blue | Architecture Decision | Database Architecture, Modular Context Architecture |
| :purple_square: Purple | Convention | Context Management Rules |
| :orange_square: Orange | Infrastructure | KD-GAT CI/CD Pipeline, Quarto Site |
| Teal | Technology / Library | Mosaic vgplot |
| Pink | Learning | pyarrow-parquet-stdout |
| Yellow | Milestone | MapViz-Observable-Build |
| Grey | Changelog | Context Update Log |
| :green_square: Green | Canonical Answer | KD-GAT Shell Environment |

!!! note "Regenerating this diagram"
    To update the knowledge graph diagram after adding new entities:
    ```bash
    python scripts/generate-kg-mermaid.py
    ```
    Copy the output into the Mermaid code fence above.

## Contributing

When you add a new page to the docs:

1. Consider whether it should appear in the concept map above (skip minor/niche pages)
2. If it's a key learning page, add a node in the appropriate `subgraph` and connect it with prerequisite edges
3. Run `mkdocs build --strict` to verify no broken links

For the full guide to editing this site, see [How This Site Works](../contributing/how-this-site-works.md).
