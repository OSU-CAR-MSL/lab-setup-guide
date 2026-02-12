# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MkDocs Material documentation site for the OSU CAR Mobility Systems Lab. All content is Markdown in `docs/`. The site is deployed automatically to GitHub Pages on push to `main`.

**Live site:** https://osu-car-msl.github.io/lab-setup-guide/

## Commands

```bash
# Install dependencies
pip install mkdocs-material mkdocs-minify-plugin

# Local development server (auto-reloads at http://127.0.0.1:8000)
mkdocs serve

# Production build (used in CI)
mkdocs build --strict
```

There are no tests or linters. The `--strict` flag in CI catches broken links and configuration errors.

## Architecture

- **`mkdocs.yml`** — Central configuration: site metadata, theme settings, markdown extensions, plugins, and the `nav:` section that defines all page navigation. Every new page must be added here.
- **`docs/`** — All Markdown content, organized into section subfolders (`getting-started/`, `osc-basics/`, `working-on-osc/`, `ml-workflows/`, `contributing/`, `resources/`).
- **`docs/stylesheets/extra.css`** — Custom OSU branding (scarlet `#bb0000` color scheme) applied on top of Material theme.
- **`.github/workflows/deploy-docs.yml`** — GitHub Actions pipeline. Triggers on pushes to `main` that touch `docs/` or `mkdocs.yml`. Builds with `mkdocs build --strict` and deploys to GitHub Pages.
- **`site/`** — Auto-generated build output (gitignored).

## Adding a New Page

1. Create a `.md` file in the appropriate `docs/` subfolder.
2. Add the page to the `nav:` section in `mkdocs.yml`.
3. Run `mkdocs serve` to preview locally before pushing.

## Markdown Features Available

The site has these extensions configured in `mkdocs.yml` — use them in content pages:

- **Admonitions:** `!!! note`, `!!! tip`, `!!! warning`, `!!! danger` (and collapsible with `???`)
- **Tabbed content:** `=== "Tab 1"` / `=== "Tab 2"`
- **Code blocks:** syntax highlighting, copy buttons, line numbers via `anchor_linenums`
- **Mermaid diagrams:** fenced with ` ```mermaid `
- **Keyboard keys:** `++ctrl+c++`
- **Task lists:** `- [x] Done` / `- [ ] Todo`

## Deployment

Push to `main` triggers automatic deployment via GitHub Actions. Only changes to `docs/**`, `mkdocs.yml`, or the workflow file itself trigger builds. Manual builds can be triggered from the Actions tab.
