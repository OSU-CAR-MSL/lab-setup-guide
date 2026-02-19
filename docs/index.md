<!-- last-reviewed: 2026-02-19 -->
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

| | VS Code | Git / GitHub | SSH | SLURM | Conda / uv | Nextflow / Prefect | PyTorch | Copilot / Claude |
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
| [Job Submission](working-on-osc/osc-job-submission.md) | Submit and manage SLURM jobs |
| [Environment Management](working-on-osc/osc-environment-management.md) | Modules, Conda, and virtual environments |
| [Pipeline Orchestration](working-on-osc/pipeline-orchestration.md) | Automate multi-step workflows with Nextflow or Prefect |
| [CARLA Simulator](working-on-osc/carla-simulator.md) | Run CARLA for autonomous driving research |
| [MATLAB & Simulink](working-on-osc/matlab-simulink.md) | Use MATLAB and Simulink on OSC |

### Machine Learning Workflows

Guides specific to ML/DL research on OSC.

| Guide | Description |
|-------|-------------|
| [PyTorch & GPU Setup](ml-workflows/pytorch-setup.md) | Install PyTorch, request GPUs, optimize training |
| [ML Project Template](ml-workflows/ml-workflow.md) | Project structure and data organization for ML on OSC |
| [Notebook-to-Script Workflow](ml-workflows/notebook-to-script.md) | Convert Jupyter prototypes to production scripts |
| [Data & Experiment Tracking](ml-workflows/data-experiment-tracking.md) | DVC, SQLite, MLflow, W&B, and Parquet for ML projects |

### Contributing

Learn how the site works and how to add new content.

| Guide | Description |
|-------|-------------|
| [Contributing Guide](contributing/how-this-site-works.md) | Architecture, adding pages, deployment |
| [GitHub Pages Setup](contributing/github-pages-setup.md) | Set up a documentation site with MkDocs or Quarto |

### Assignments

Hands-on tasks for new undergraduates joining the lab.

| Assignment | Description |
|------------|-------------|
| [Assignment 1: Personal Website](assignments/personal-website.md) | Build and deploy your academic website with Git, Quarto, and GitHub Pages |
| [Assignment 2a: Exploratory Data Analysis](assignments/exploratory-data-analysis.md) | EDA, visualizations, and baseline ML models with pandas and scikit-learn |
| [Assignment 2b: Concepts, Admin & AI Setup](assignments/concepts-admin-setup.md) | OSC access, W&B, SQL, ML pipelines, Copilot, and Claude Code setup |

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
| OSC Status Page | [osc.edu/status](https://www.osc.edu/supercomputing/system-status) |
| Lab GitHub | [github.com/OSU-CAR-MSL](https://github.com/OSU-CAR-MSL) |
