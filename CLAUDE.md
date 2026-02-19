# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MkDocs Material documentation site for the OSU CAR Mobility Systems Lab. All content is Markdown in `docs/`. The site is deployed automatically to GitHub Pages on push to `main`.

**Live site:** https://osu-car-msl.github.io/lab-setup-guide/

## Commands

```bash
# Install dependencies (pinned in requirements-docs.txt)
pip install -r requirements-docs.txt

# Local development server (auto-reloads at http://127.0.0.1:8000)
mkdocs serve

# Production build (used in CI)
mkdocs build --strict
```

There are no tests or linters. The `--strict` flag in CI catches broken links and configuration errors.

## Architecture

- **`mkdocs.yml`** — Central configuration: site metadata, theme settings, markdown extensions, plugins, and the `nav:` section that defines all page navigation. Every new page must be added here.
- **`docs/`** — All Markdown content, organized into section subfolders (`getting-started/`, `osc-basics/`, `working-on-osc/`, `ml-workflows/`, `contributing/`, `assignments/`, `resources/`).
- **`docs/stylesheets/extra.css`** — Custom OSU branding (scarlet `#bb0000` color scheme) applied on top of Material theme.
- **`.github/workflows/deploy-docs.yml`** — GitHub Actions pipeline. Triggers on pushes to `main` that touch `docs/` or `mkdocs.yml`. Builds with `mkdocs build --strict` and deploys to GitHub Pages.
- **`requirements-docs.txt`** — Python dependencies for MkDocs build. Referenced by CI cache key in the deploy workflow.
- **`site/`** — Auto-generated build output (gitignored).

## Content Architecture — Canonical Locations

Each topic has exactly one canonical page. Other pages cross-link to it instead of repeating the content.

| Topic | Canonical Page |
|-------|---------------|
| Cluster specs, partitions, storage quotas | `osc-basics/osc-clusters-overview.md` |
| SSH keys and SSH config | `osc-basics/osc-ssh-connection.md` |
| File transfer (SCP, rsync, SFTP) | `osc-basics/osc-file-transfer.md` |
| VS Code extensions | `getting-started/vscode-extensions.md` |
| Modules, venvs, conda | `working-on-osc/osc-environment-management.md` |
| SLURM jobs, job arrays, job scripts | `working-on-osc/osc-job-submission.md` |
| PyTorch install, GPU requesting, GPU perf, multi-GPU, memory mgmt | `ml-workflows/pytorch-setup.md` |
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

**No link dumps.** The useful-links page is intentionally minimal — only OSC-specific portals, lab resources, and genuinely hard-to-find references. Don't add generic links (Python docs, ML courses, framework homepages) that any search engine would find.

**Practical, not encyclopedic.** Include concrete commands, code snippets, and job script templates that people will actually copy. Skip generic explanations that the official docs already cover.

## Adding a New Page

1. Create a `.md` file in the appropriate `docs/` subfolder.
2. Add the page to the `nav:` section in `mkdocs.yml`.
3. Check the canonical locations table above — if the new page overlaps with an existing topic, cross-link instead of duplicating.
4. Run `mkdocs build --strict` to verify no broken links.

## Markdown Features Available

The site has these extensions configured in `mkdocs.yml` — use them in content pages:

- **Admonitions:** `!!! note`, `!!! tip`, `!!! warning`, `!!! danger` (and collapsible with `???`)
- **Tabbed content:** `=== "Tab 1"` / `=== "Tab 2"`
- **Code blocks:** syntax highlighting, copy buttons, line numbers via `anchor_linenums`
- **Mermaid diagrams:** fenced with ` ```mermaid `
- **Keyboard keys:** `++ctrl+c++`
- **Task lists:** `- [x] Done` / `- [ ] Todo`

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

## Deployment

Push to `main` triggers automatic deployment via GitHub Actions. Only changes to `docs/**`, `mkdocs.yml`, or the workflow file itself trigger builds. Manual builds can be triggered from the Actions tab.
