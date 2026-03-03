<!-- last-reviewed: 2026-03-03 -->
# Hugging Face Spaces Deployment

Deploy interactive ML dashboards, Quarto reports, and demo apps to [Hugging Face Spaces](https://huggingface.co/spaces) — free hosting with GitHub Actions CI/CD.

---

## What HF Spaces Are

Hugging Face Spaces provides free hosting for:

- **Static sites** — Quarto reports, HTML dashboards, paper supplements
- **Gradio apps** — Interactive ML demos with auto-generated UIs
- **Streamlit apps** — Data apps with Python
- **Docker containers** — Anything else

For research, the most common use case is deploying a Quarto report site with interactive figures (Mosaic/vgplot, Observable Plot) backed by DuckDB-WASM for in-browser SQL queries over your experiment data.

### Why HF Spaces for Research

| Alternative | Limitation |
|-------------|-----------|
| GitHub Pages | No custom headers (breaks DuckDB-WASM CORS), public repos only for free |
| Netlify/Vercel | Requires separate account, more complex setup |
| **HF Spaces** | Free, ML-native, custom CORS headers, private spaces, API integration |

HF Spaces supports custom HTTP headers — critical for DuckDB-WASM, which requires Cross-Origin Isolation headers (`cross-origin-embedder-policy: require-corp`).

---

## Account & Token Setup

### 1. Create a Hugging Face Account

Sign up at [huggingface.co](https://huggingface.co/) — free accounts work fine.

### 2. Create a Write Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name it (e.g., `github-actions-deploy`)
4. Select **Write** permission
5. Copy the token (`hf_...`)

### 3. Add as GitHub Secret

In your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `HF_TOKEN`
4. Value: paste your `hf_...` token

---

## Deploying a Static Quarto Site

### Step 1: Create the HF Space

You can create the Space manually on huggingface.co, or let the first deploy create it automatically. If creating manually:

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Static** as the SDK
3. Set visibility (public or private)

### Step 2: Prepare Your Quarto Project

Ensure your Quarto project builds to a `_site/` directory:

```yaml
# _quarto.yml
project:
  type: website
  output-dir: _site
```

### Step 3: Add the GitHub Actions Workflow

Add a deploy job to your CI workflow (`.github/workflows/ci.yml`):

```yaml
deploy-hf:
  if: github.ref == 'refs/heads/main'
  needs: [build]  # Run after your build/test jobs
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - uses: quarto-dev/quarto-actions/setup@v2

    - uses: quarto-dev/quarto-actions/render@v2
      with:
        path: reports/    # Path to your Quarto project

    - name: Create Space README
      run: |
        cat > reports/_site/README.md << 'EOF'
        ---
        title: "My Research Dashboard"
        sdk: static
        app_file: index.html
        pinned: true
        custom_headers:
          cross-origin-embedder-policy: require-corp
          cross-origin-opener-policy: same-origin
          cross-origin-resource-policy: cross-origin
        ---
        EOF

    - name: Deploy to HF Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        pip install huggingface-hub
        python -c "
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            repo_id='YOUR-USERNAME/YOUR-SPACE-NAME',
            folder_path='reports/_site',
            repo_type='space',
            commit_message='Deploy from ${{ github.sha }}',
        )
        print('HF Space deploy complete')
        "
```

!!! warning "Replace placeholders"
    Change `YOUR-USERNAME/YOUR-SPACE-NAME` to your actual HF username and Space name (e.g., `buckeyeguy/my-dashboard`). Change `reports/` to the path of your Quarto project.

---

## Space Configuration

The Space README (`README.md` in the Space root) controls Space behavior via YAML front matter:

```yaml
---
title: "My Research Dashboard"
sdk: static                    # static | gradio | streamlit | docker
app_file: index.html           # Entry point for static sites
pinned: true                   # Pin to your profile
private: true                  # Private visibility (optional)
colorFrom: green               # Gradient color (cosmetic)
colorTo: blue
custom_headers:                # Custom HTTP headers
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---
```

### CORS Headers for DuckDB-WASM

DuckDB-WASM requires Cross-Origin Isolation to use `SharedArrayBuffer`. Without these headers, DuckDB-WASM falls back to a slower single-threaded mode or fails entirely:

| Header | Value | Why |
|--------|-------|-----|
| `cross-origin-embedder-policy` | `require-corp` | Enables `SharedArrayBuffer` |
| `cross-origin-opener-policy` | `same-origin` | Required companion to COEP |
| `cross-origin-resource-policy` | `cross-origin` | Allows loading resources cross-origin |

!!! note
    GitHub Pages does not support custom headers, which is one reason HF Spaces is better suited for DuckDB-WASM dashboards.

---

## Use Cases in Research

### Interactive Paper Supplements

Deploy a Quarto website alongside your paper with interactive figures that let reviewers explore your data:

```
reports/
├── _quarto.yml
├── index.qmd              # Landing page
├── dashboard.qmd          # Interactive dashboard
├── paper/                 # Paper chapters with embedded figures
│   ├── 01-introduction.qmd
│   ├── 02-methods.qmd
│   └── ...
└── data/                  # Parquet files for DuckDB-WASM
    ├── metrics.parquet
    └── training_curves.parquet
```

### Sharing Experiment Results

The Parquet datalake pattern from [Data & Experiment Tracking](data-experiment-tracking.md) pairs naturally with HF Spaces:

1. **Train** — Run experiments, log to Parquet datalake
2. **Export** — Export datalake Parquet to `reports/data/`
3. **Build** — Quarto renders interactive dashboards with DuckDB-WASM
4. **Deploy** — GitHub Actions pushes to HF Spaces on every push to main

This gives you a live dashboard that updates automatically with every new experiment.

### Gradio Apps for Model Demos

For interactive model inference demos, use Gradio instead of static:

```python
# app.py
import gradio as gr
import torch

model = torch.load("model.pt")

def predict(input_data):
    with torch.no_grad():
        return model(input_data)

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
demo.launch()
```

Set `sdk: gradio` in the Space README and push `app.py` + `requirements.txt`.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Deploy fails with 401 | Invalid or expired HF token | Regenerate token, update GitHub secret |
| DuckDB-WASM errors in browser | Missing CORS headers | Verify `custom_headers` in Space README |
| Space shows "Building" forever | Large files or build errors | Check Space logs on HF; reduce `_site/` size |
| Charts don't render | OJS/JS runtime errors | Open browser console; `quarto render` success doesn't guarantee JS works |
| Space is blank | Wrong `app_file` | Verify `app_file: index.html` matches your build output |

---

## Next Steps

- [Data & Experiment Tracking](data-experiment-tracking.md) — Parquet datalake that feeds HF Spaces dashboards
- [DuckDB Analytics Layer](duckdb-analytics.md) — SQL queries over experiment results
- [Pipeline Orchestration](../working-on-osc/pipeline-orchestration.md) — Automating the train → export → deploy pipeline
