# Quarto Academic Website Template

A minimal [Quarto](https://quarto.org/) template for personal academic websites, designed for the CAR Mobility Systems Lab onboarding workflow.

## What's Included

| File | Purpose |
|------|---------|
| `index.qmd` | Home page with photo, bio, and links |
| `research.qmd` | Auto-listing of research projects |
| `cv.qmd` | Curriculum vitae |
| `posts.qmd` | Blog post listing |
| `_quarto.yml` | Site configuration (navbar, theme, URLs) |
| `.github/workflows/publish.yml` | Auto-deploy to GitHub Pages on push |

## Quick Start

1. **Use this template** → click the green "Use this template" button above
2. **Name your repo** `YOURUSERNAME.github.io`
3. **Clone locally:** `git clone git@github.com:YOURUSERNAME/YOURUSERNAME.github.io.git`
4. **Edit `_quarto.yml`** — update name, description, URLs
5. **Edit `index.qmd`** — add your bio and photo
6. **Preview locally:** `quarto preview`
7. **Push to GitHub:** `git add . && git commit -m "personalize site" && git push`
8. **Enable GitHub Pages:** Settings → Pages → Source: **GitHub Actions**

Your site will be live at `https://YOURUSERNAME.github.io` within a few minutes.

## Prerequisites

- [Git](https://git-scm.com/downloads)
- [VS Code](https://code.visualstudio.com/) (recommended) with the [Quarto extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto)
- [Quarto CLI](https://quarto.org/docs/get-started/)

## Adding Content

**New blog post:** Create a `.qmd` file in `posts/` with YAML frontmatter:
```yaml
---
title: "My Post Title"
description: "A short summary."
date: "2025-02-12"
categories: [topic1, topic2]
---
```

**New research project:** Create a `.qmd` file in `research/` with the same pattern.

## Deployment

The included GitHub Action automatically renders and deploys your site whenever you push to `main`. No need to render locally before pushing (though `quarto preview` is useful for checking your work).

## License

MIT — use freely for your own academic site.
