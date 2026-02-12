# CAR Mobility Systems Lab Setup Guide

Welcome to the OSU CAR Mobility Systems Lab documentation! This guide helps lab members set up their development environment and work effectively on the Ohio Supercomputer Center (OSC).

---

## :rocket: Quick Start

New to the lab? Follow these steps to get up and running:

| Step | Guide | Description |
|:----:|-------|-------------|
| **1** | **[Set up VS Code](getting-started/vscode-setup.md)** | Install and configure Visual Studio Code with essential extensions |
| **2** | **[Get OSC Access](osc-basics/osc-account-setup.md)** | Request your account on the Ohio Supercomputer Center |
| **3** | **[Connect via SSH](osc-basics/osc-ssh-connection.md)** | Set up SSH keys and connect to OSC from your machine |
| **4** | **[Start Developing](osc-basics/osc-remote-development.md)** | Use VS Code's Remote-SSH for seamless development on the cluster |

---

## :wrench: Why These Tools?

Each tool in our stack addresses a specific challenge in the ML research workflow — from writing code locally to running large-scale experiments on OSC's supercomputers.

| | VS Code | Git / GitHub | SSH | SLURM | Conda / uv | Snakemake | PyTorch | Copilot / Claude |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| **Edit & debug code** | Primary IDE and debugger | | | | | | | AI-powered suggestions |
| **Collaborate on code** | | Version control & pull requests | | | Shared environment specs | | | |
| **Access HPC clusters** | Remote-SSH extension | | Secure cluster login | | | | | |
| **Manage Python envs** | | | | | Reproducible environments | | | |
| **Run compute jobs** | | | | Schedule CPUs & GPUs | | Auto-submit SLURM jobs | | |
| **Automate pipelines** | | | | Execute each step | | Define DAG workflows | | |
| **Train ML models** | Monitor & debug | | | GPU allocation | Install ML stack | Orchestrate experiments | Neural network framework | Help write & fix code |

---

## :books: Documentation Sections

### Getting Started

Set up your local development environment before connecting to OSC.

| Guide | Description |
|-------|-------------|
| [VS Code Setup](getting-started/vscode-setup.md) | Install and configure Visual Studio Code |
| [VS Code Extensions](getting-started/vscode-extensions.md) | Required and recommended extensions |
| [Python Environment Setup](getting-started/python-environment-setup.md) | WSL filesystem, `uv`, and local Python environments |
| [AI Coding Assistants](getting-started/ai-coding-assistants.md) | Set up GitHub Copilot and Claude Code |

### OSC Basics

Learn how to access and navigate the Ohio Supercomputer Center.

| Guide | Description |
|-------|-------------|
| [Clusters Overview](osc-basics/osc-clusters-overview.md) | HPC terminology, Pitzer and Owens specs, partitions |
| [Account Setup](osc-basics/osc-account-setup.md) | Request and configure your OSC account |
| [SSH Connection](osc-basics/osc-ssh-connection.md) | Connect from your local machine |
| [Remote Development](osc-basics/osc-remote-development.md) | Use VS Code with OSC |
| [File Transfer](osc-basics/osc-file-transfer.md) | Move files to and from OSC |

### Working on OSC

Best practices and guides for running computational work.

| Guide | Description |
|-------|-------------|
| [Best Practices](working-on-osc/osc-best-practices.md) | Tips for efficient work on OSC |
| [Job Submission](working-on-osc/osc-job-submission.md) | Submit and manage SLURM jobs |
| [Environment Management](working-on-osc/osc-environment-management.md) | Modules, Conda, and virtual environments |
| [Snakemake Pipelines](working-on-osc/snakemake-orchestration.md) | Automate multi-step workflows with SLURM |

### Machine Learning Workflows

Guides specific to ML/DL research on OSC.

| Guide | Description |
|-------|-------------|
| [PyTorch Setup](ml-workflows/pytorch-setup.md) | Install and configure PyTorch on OSC |
| [ML Workflow Guide](ml-workflows/ml-workflow.md) | Best practices for ML projects |
| [GPU Computing](ml-workflows/gpu-computing.md) | Using GPUs for training |
| [Data & Experiment Tracking](ml-workflows/data-experiment-tracking.md) | DVC, SQLite, MLflow, and Parquet for ML projects |

### Contributing

Learn how the site works and how to add new content.

| Guide | Description |
|-------|-------------|
| [How This Site Works](contributing/how-this-site-works.md) | Architecture, deployment, and theme features |
| [Adding a Page](contributing/adding-a-page.md) | Step-by-step guide to adding new pages |
| [GitHub Pages Setup](contributing/github-pages-setup.md) | Set up a documentation site with MkDocs or Quarto |

### Resources

| Guide | Description |
|-------|-------------|
| [Troubleshooting](resources/troubleshooting.md) | Common issues and solutions |
| [Useful Links](resources/useful-links.md) | External resources and documentation |

---

## :bulb: Tips

!!! tip "Use the search"
    Press ++s++ or ++slash++ to search the documentation. The search indexes all pages and code blocks.

!!! tip "Dark mode"
    Click the :material-brightness-7: icon in the header to toggle dark mode.

!!! tip "Edit on GitHub"
    Found an error? Click the :material-pencil: icon on any page to suggest an edit.

---

## :link: Quick Links

| Resource | Link |
|----------|------|
| OSC Documentation | [osc.edu/resources](https://www.osc.edu/resources) |
| OSC OnDemand Portal | [ondemand.osc.edu](https://ondemand.osc.edu) |
| OSC Status Page | [osc.edu/status](https://www.osc.edu/services/status) |
| Lab GitHub | [github.com/OSU-CAR-MSL](https://github.com/OSU-CAR-MSL) |
