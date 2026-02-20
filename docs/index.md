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

<div class="grid cards" markdown>

-   :material-laptop:{ .lg .middle } **Getting Started**

    ---

    WSL2, VS Code, Python environments, and AI coding assistants for local development.

    [:octicons-arrow-right-24: Get started](getting-started/wsl-setup.md)

-   :material-server-network:{ .lg .middle } **OSC Basics**

    ---

    Cluster specs, account setup, SSH, remote development, file transfer, and OnDemand.

    [:octicons-arrow-right-24: Learn OSC basics](osc-basics/osc-clusters-overview.md)

-   :material-cog-play:{ .lg .middle } **Working on OSC**

    ---

    SLURM job submission, environment management, pipeline orchestration, and simulators.

    [:octicons-arrow-right-24: Start working](working-on-osc/osc-job-submission.md)

-   :material-brain:{ .lg .middle } **ML Workflows**

    ---

    PyTorch & GPU setup, PyG, project templates, notebook-to-script, and experiment tracking.

    [:octicons-arrow-right-24: ML guides](ml-workflows/pytorch-setup.md)

-   :material-file-document-edit:{ .lg .middle } **Contributing**

    ---

    How this site works, adding pages, and setting up your own documentation site.

    [:octicons-arrow-right-24: Contribute](contributing/how-this-site-works.md)

-   :material-school:{ .lg .middle } **Assignments**

    ---

    Hands-on tasks for new undergraduates: personal websites, EDA, and OSC onboarding.

    [:octicons-arrow-right-24: View assignments](assignments/index.md)

-   :material-lifebuoy:{ .lg .middle } **Resources**

    ---

    Troubleshooting guides, useful links, and OSC support contacts.

    [:octicons-arrow-right-24: Get help](resources/troubleshooting.md)

</div>

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
| OSC Status Page | [osc.edu/status](https://www.osc.edu/resources/system-status) |
| Lab GitHub | [github.com/OSU-CAR-MSL](https://github.com/OSU-CAR-MSL) |
