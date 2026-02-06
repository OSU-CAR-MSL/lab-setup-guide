# How to Add a New Page

This guide walks through adding a new markdown page to the site, step by step.

## Step 1: Create the Markdown File

Create a new `.md` file inside the `docs/` folder:

```bash
# In the repo root
touch docs/my-new-guide.md
```

You can also put it in a subfolder:

```bash
mkdir -p docs/my-section
touch docs/my-section/my-new-guide.md
```

## Step 2: Write Your Content

Open the file and add your content. Every page should start with a level-1 heading:

```markdown
# My New Guide

A brief introduction to what this page covers.

## First Section

Your content here. You can use:

- Bullet lists
- **Bold** and *italic* text
- [Links](https://example.com)
- Code blocks, tables, images, etc.

## Second Section

More content.
```

!!! tip "Use the existing pages as examples"
    Look at any file in `docs/` to see real examples of formatting, code blocks, admonitions, and other features.

## Step 3: Add It to the Navigation

Open `mkdocs.yml` and add your page to the `nav:` section. Find the section where your page fits:

```yaml
nav:
  - Home: index.md

  - Getting Started:
    - VS Code Setup: vscode-setup.md
    - VS Code Extensions: vscode-extensions.md

  - OSC Basics:
    - Account Setup: osc-account-setup.md
    # ... other pages ...

  - Resources:
    - Troubleshooting: troubleshooting.md
    - Useful Links: useful-links.md
    - My New Guide: my-new-guide.md        # <-- add your page here
```

For pages in subfolders, include the subfolder path:

```yaml
  - My Section:
    - My New Guide: my-section/my-new-guide.md
```

!!! warning "File paths are relative to `docs/`"
    The paths in `nav:` are relative to the `docs/` folder, not the repo root. So `docs/vscode-setup.md` is listed as just `vscode-setup.md`.

## Step 4: Preview Locally (Optional)

If you have MkDocs installed locally, you can preview before pushing:

```bash
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) to see your changes with live reload.

If you don't have it installed:

```bash
pip install mkdocs-material mkdocs-minify-plugin
```

## Step 5: Commit and Push

```bash
git add docs/my-new-guide.md mkdocs.yml
git commit -m "Add guide for my-new-guide"
git push origin main
```

The site will automatically rebuild and deploy in about 45 seconds. You can monitor the build at the [Actions tab](https://github.com/OSU-CAR-MSL/lab-setup-guide/actions).

## Quick Reference: Formatting

Here are the most useful formatting options available on this site.

### Headings

```markdown
# Page Title (H1 — use only once per page)
## Section (H2)
### Subsection (H3)
```

### Code Blocks

Use triple backticks with a language name for syntax highlighting:

````markdown
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```bash
squeue -u $USER
```
````

### Admonitions (Callout Boxes)

```markdown
!!! note
    A note callout.

!!! tip "Custom Title"
    A tip with a custom title.

!!! warning
    A warning callout.

!!! danger
    A danger callout.
```

Which renders as:

!!! note
    A note callout.

!!! tip "Custom Title"
    A tip with a custom title.

!!! warning
    A warning callout.

### Tables

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Row 1    | Data     | Data     |
| Row 2    | Data     | Data     |
```

### Links

```markdown
Link to an external site: [OSC Documentation](https://www.osc.edu/resources)

Link to another page on this site: [VS Code Setup](../vscode-setup.md)
```

!!! tip "Linking between pages"
    Use relative paths when linking to other pages on the site. The `../` goes up one folder level.

### Images

Place images in `docs/assets/` and reference them:

```markdown
![Alt text](assets/my-screenshot.png)
```

## Checklist

Before pushing a new page, make sure:

- [x] File is in the `docs/` folder (or a subfolder)
- [x] File starts with a `# Title` heading
- [x] Page is added to `nav:` in `mkdocs.yml`
- [x] Links to other pages use correct relative paths
- [x] Code blocks specify the language for syntax highlighting
