# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MkDocs Material documentation site for the OSU CAR Mobility Systems Lab. All content is Markdown in `docs/`. The site is deployed automatically to GitHub Pages on push to `main`.

**Live site:** https://osu-car-msl.github.io/lab-setup-guide/

## Commands

```bash
# Install dependencies (pinned in requirements-docs.txt; [imaging] extra pulls in Pillow + CairoSVG)
pip install -r requirements-docs.txt

# Local development server (auto-reloads at http://127.0.0.1:8000)
mkdocs serve

# Production build (used in CI)
mkdocs build --strict

# Content freshness check
python scripts/check-freshness.py --max-age-days 180

# SSOT duplication check (advisory)
python scripts/check-duplication.py
```

There are no tests or linters. The `--strict` flag in CI catches broken links and configuration errors.

## Architecture

```
lab-setup-guide/
├── mkdocs.yml                          # Nav, theme, extensions, plugins
├── requirements-docs.txt               # Python deps for MkDocs build (CI cache key)
├── CLAUDE.md                           # This file
├── overrides/                          # MkDocs Material theme overrides
│   └── 404.html                        # Custom 404 page
├── docs/                               # All Markdown content (31 pages)
│   ├── index.md                        # Homepage with grid cards, tool matrix, tips
│   ├── tags.md                         # Auto-generated tag index
│   ├── includes/
│   │   └── abbreviations.md            # Glossary for abbreviation tooltips
│   ├── assets/                         # Logo, favicon (placeholder)
│   ├── getting-started/                # 5 pages — WSL2, VS Code, extensions, Python env, AI assistants
│   ├── osc-basics/                     # 6 pages — clusters, account, SSH, remote dev, file transfer, OnDemand
│   ├── working-on-osc/                 # 5 pages — jobs, envs, orchestration, CARLA, MATLAB
│   ├── ml-workflows/                   # 5 pages — PyTorch, PyG, project template, notebook-to-script, tracking
│   ├── contributing/                   # 2 pages — contributing guide, GitHub Pages setup
│   ├── assignments/                    # 4 pages — index + 3 assignments
│   ├── resources/                      # 2 pages — troubleshooting, useful links
│   └── stylesheets/extra.css           # Custom OSU scarlet branding + print styles
├── scripts/
│   ├── check-freshness.py              # Flags pages with stale last-reviewed dates (>6 months)
│   ├── check-duplication.py            # Advisory SSOT duplication detector
│   └── mcp_lab_docs.py                 # MCP server — exposes docs as tools for Claude Code
├── .github/workflows/
│   ├── deploy-docs.yml                 # Build + deploy on push to main (docs/**, mkdocs.yml, overrides/**)
│   └── link-check.yml                  # Lychee link checker + freshness + duplication (push + weekly cron)
└── site/                               # Auto-generated build output (gitignored)
```

## Content Architecture — Canonical Locations

Each topic has exactly one canonical page. Other pages cross-link to it instead of repeating the content.

| Topic | Canonical Page |
|-------|---------------|
| WSL2 installation and setup | `getting-started/wsl-setup.md` |
| Cluster specs, partitions, storage quotas | `osc-basics/osc-clusters-overview.md` |
| SSH keys and SSH config | `osc-basics/osc-ssh-connection.md` |
| File transfer (SCP, rsync, SFTP) | `osc-basics/osc-file-transfer.md` |
| OSC OnDemand portal and interactive apps | `osc-basics/osc-ondemand.md` |
| VS Code extensions | `getting-started/vscode-extensions.md` |
| Modules, venvs, conda | `working-on-osc/osc-environment-management.md` |
| SLURM jobs, job arrays, job scripts | `working-on-osc/osc-job-submission.md` |
| Pipeline orchestration (Nextflow, Prefect) | `working-on-osc/pipeline-orchestration.md` |
| PyTorch install, GPU requesting, GPU perf, multi-GPU, memory mgmt | `ml-workflows/pytorch-setup.md` |
| PyTorch Geometric (PyG) setup and GNN workflows | `ml-workflows/pyg-setup.md` |
| Experiment tracking (DVC, SQLite, MLflow, W&B, TensorBoard, Parquet) | `ml-workflows/data-experiment-tracking.md` |
| Notebook-to-script conversion | `ml-workflows/notebook-to-script.md` |
| OSC portals and support contacts | `resources/useful-links.md` |

## Content Conventions

**Single source of truth.** Every topic lives on one page. If another page needs that information, it links to the canonical page — never copy-pastes the content.

**Lean pages.** Each page should cover its own topic and nothing more. Avoid "here's everything you might need" mega-pages. If a section is growing beyond the page's scope, it belongs on its canonical page with a cross-link.

**Cross-links over duplication.** When referencing another topic, use a one- or two-line mention plus a Markdown link. Example:
```markdown
For partition details and GPU types, see the [Clusters Overview](../osc-basics/osc-clusters-overview.md).
```

**Content freshness.** Every `.md` page in `docs/` must have `<!-- last-reviewed: YYYY-MM-DD -->` as its first line (or immediately after YAML front matter if tags are used). Update the date when you meaningfully review or edit a page. The CI freshness check flags pages older than 6 months.

**No link dumps.** The useful-links page is intentionally minimal — only OSC-specific portals, lab resources, and genuinely hard-to-find references. Don't add generic links (Python docs, ML courses, framework homepages) that any search engine would find.

**Practical, not encyclopedic.** Include concrete commands, code snippets, and job script templates that people will actually copy. Skip generic explanations that the official docs already cover.

## Adding a New Page

1. Create a `.md` file in the appropriate `docs/` subfolder.
2. Add `<!-- last-reviewed: YYYY-MM-DD -->` as the first line (after optional YAML front matter for tags).
3. Add the page to the `nav:` section in `mkdocs.yml`.
4. Check the canonical locations table above — if the new page overlaps with an existing topic, cross-link instead of duplicating.
5. Verify:
   ```bash
   mkdocs build --strict                            # broken links, missing nav
   python scripts/check-freshness.py --max-age-days 180   # stale pages
   python scripts/check-duplication.py              # SSOT violations (advisory)
   ```

## Markdown Features Available

The site has these extensions configured in `mkdocs.yml` — use them in content pages:

- **Admonitions:** `!!! note`, `!!! tip`, `!!! warning`, `!!! danger` (and collapsible with `???`)
- **Tabbed content:** `=== "Tab 1"` / `=== "Tab 2"`
- **Code blocks:** syntax highlighting, copy buttons, line numbers via `anchor_linenums`
- **Mermaid diagrams:** fenced with ` ```mermaid `
- **Keyboard keys:** `++ctrl+c++`
- **Task lists:** `- [x] Done` / `- [ ] Todo`
- **Abbreviation tooltips:** HPC/ML terms auto-defined via `includes/abbreviations.md`
- **Tags:** Pages can have YAML front matter `tags:` for the auto-generated tag index at `tags.md`
- **Image lightbox:** Images are click-to-zoom via `glightbox` plugin
- **Social cards:** Auto-generated Open Graph preview images via `social` plugin
- **Navigation footer:** Prev/next page links at bottom of every page
- **Breadcrumbs:** Navigation path shown above page titles
- **Git revision dates:** "Last updated" shown on every page via `git-revision-date-localized` plugin

## Assignments

Assignment pages live in `docs/assignments/`. Each assignment follows a consistent pattern established by `personal-website.md`:

- Metadata table (author, estimated time, prerequisites)
- "What You'll Build/Learn" intro paragraph
- Multi-part structure with numbered tasks
- Admonitions for tips/warnings, tabbed content for OS-specific instructions
- Task-list checklists (`- [ ]`) at the end of each part
- Final Deliverables checklist
- Troubleshooting table

When adding new assignments, also update `docs/assignments/index.md` (the table) and the `nav:` section in `mkdocs.yml`.

## Contributing

The contributing section has two pages:

- `contributing/how-this-site-works.md` — Combined contributing guide: architecture, adding pages, local preview, deployment, and markdown reference
- `contributing/github-pages-setup.md` — Setting up a separate documentation site with MkDocs or Quarto

## CI / Deployment

Two GitHub Actions workflows, both triggered on push to `main` when `docs/` or `mkdocs.yml` change:

| Workflow | Trigger | Purpose | Blocking? |
|----------|---------|---------|-----------|
| `deploy-docs.yml` | Push to `main` + manual | `mkdocs build --strict` → GitHub Pages deploy | Yes — broken links fail the build |
| `link-check.yml` | Push to `main` + weekly Monday cron | Lychee external link check, freshness audit, SSOT duplication check | Yes — `fail: true` |

The link-check workflow excludes authenticated portals (`ondemand.osc.edu`, `my.osc.edu`) and localhost URLs.

## MCP Server

`scripts/mcp_lab_docs.py` is a FastMCP server that exposes the documentation as queryable tools for any Claude Code session on this machine. It reads raw markdown files directly from `docs/` — no build step required.

**Tools provided:**

| Tool | Purpose |
|------|---------|
| `search_docs(query, limit=5)` | Case-insensitive search across all pages; returns matching lines with context |
| `read_page(page_path)` | Returns full markdown content of a page (e.g. `ml-workflows/pytorch-setup.md`) |
| `list_pages()` | Navigation tree parsed from `mkdocs.yml` — discover what pages exist |

**Configuration** is in `~/.claude/settings.json` under the `lab-docs` key. It uses `uv run --with` to pull `mcp[cli]` and `pyyaml` at runtime, so no virtualenv setup is needed.

```bash
# Test the server starts cleanly
uv run --with "mcp[cli]" --with pyyaml scripts/mcp_lab_docs.py
# Ctrl+C to stop — it blocks on stdio input

# Verify in Claude Code
# Run /mcp — should show "lab-docs" with 3 tools
```
