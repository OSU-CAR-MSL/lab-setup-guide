# How This Website Works

This site is built with [MkDocs](https://www.mkdocs.org/) using the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. It turns plain markdown files into a professional documentation website, hosted for free on GitHub Pages.

## Architecture Overview

```
lab-setup-guide/
├── mkdocs.yml                  # Site configuration (theme, nav, plugins)
├── docs/                       # All markdown content lives here
│   ├── index.md                # Homepage
│   ├── vscode-setup.md         # Each .md file becomes a page
│   ├── osc-account-setup.md
│   ├── stylesheets/
│   │   └── extra.css           # Custom OSU scarlet branding
│   └── contributing/
│       ├── how-this-site-works.md
│       └── adding-a-page.md
└── .github/
    └── workflows/
        └── deploy-docs.yml     # Automatic build & deploy pipeline
```

## How Deployment Works

Every time you push changes to the `main` branch, the following happens automatically:

```
git push origin main
       │
       ▼
GitHub detects the push
       │
       ▼
GitHub Actions workflow triggers
(only if docs/ or mkdocs.yml changed)
       │
       ▼
A cloud server spins up and:
  1. Checks out the repo
  2. Installs Python + mkdocs-material
  3. Runs "mkdocs build --strict"
  4. Uploads the built site
       │
       ▼
GitHub Pages serves the site at:
osu-car-msl.github.io/lab-setup-guide/
```

You never need to manually build or upload anything. Just push markdown and the site updates in about 45 seconds.

## Key Files

### `mkdocs.yml`

The main configuration file at the repo root. It controls:

- **`site_name`** — The title shown in the header
- **`theme`** — Material theme settings, color palette, features like dark mode
- **`nav`** — The sidebar navigation structure (which pages appear and in what order)
- **`markdown_extensions`** — Extra features like code highlighting, admonitions, tabs
- **`plugins`** — Search, HTML minification

### `docs/` folder

Every `.md` file in this folder becomes a page on the site. Subfolders are supported and help organize related content. The file path determines the URL:

| File path | URL |
|-----------|-----|
| `docs/index.md` | `/lab-setup-guide/` |
| `docs/vscode-setup.md` | `/lab-setup-guide/vscode-setup/` |
| `docs/contributing/adding-a-page.md` | `/lab-setup-guide/contributing/adding-a-page/` |

### `.github/workflows/deploy-docs.yml`

The GitHub Actions workflow that automates deployment. It triggers on:

- Pushes to `main` that change `docs/` or `mkdocs.yml`
- Manual trigger from the Actions tab on GitHub

## Theme Features

The Material theme gives us these features out of the box:

- **Search** — Press ++s++ or ++slash++ to search all pages
- **Dark/light mode** — Toggle in the header
- **Code copy buttons** — One-click copy on all code blocks
- **Navigation tabs** — Top-level sections appear as tabs
- **Edit on GitHub** — Pencil icon links to the source file on GitHub
- **Table of contents** — Auto-generated sidebar from headings
- **Admonitions** — Callout boxes for tips, warnings, notes

## Markdown Extensions Available

You can use these in any markdown file:

### Admonitions

```markdown
!!! tip "Helpful hint"
    This is a tip callout box.

!!! warning
    This is a warning.
```

### Code blocks with syntax highlighting

````markdown
```python
import torch
print(torch.cuda.is_available())
```
````

### Tabbed content

```markdown
=== "Linux"
    ```bash
    ssh user@pitzer.osc.edu
    ```

=== "Windows"
    ```bash
    ssh user@pitzer.osc.edu
    ```
```

### Keyboard keys

```markdown
Press ++ctrl+c++ to copy.
```

### Task lists

```markdown
- [x] Install VS Code
- [ ] Set up SSH keys
```

## Local Development

To preview changes before pushing:

```bash
# Install dependencies (one time)
pip install mkdocs-material mkdocs-minify-plugin

# Start local server with live reload
mkdocs serve

# Open http://127.0.0.1:8000
```

The local server auto-reloads when you save a file, so you can see changes instantly.
