# Snakemake Pipeline Orchestration

Learn how to use Snakemake to build reproducible, automated computational pipelines on OSC with SLURM integration.

**On this page:**

- [Installation on OSC](#installation-on-osc) — pip or conda setup
- [Snakefile Basics](#snakefile-basics) — rules, wildcards, config files
- [SLURM Profile Configuration](#slurm-profile-configuration) — submit rules as SLURM jobs
- [Running Pipelines on OSC](#running-pipelines-on-osc) — dry run, local, cluster execution
- [Practical Examples](#practical-examples) — ML training, hyperparameter sweep, data processing
- [Monitoring and Debugging](#monitoring-and-debugging) — logs, recovery, common errors

## Overview

Snakemake is a workflow management system that uses Python-based rule definitions to create reproducible and scalable data analysis pipelines. It automatically determines which steps need to run based on file dependencies and can submit each step as a separate SLURM job.

### When to Use Snakemake

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **Single `sbatch` script** | One-off jobs, simple tasks | No dependency tracking, manual reruns |
| **Job arrays** | Many identical jobs with different parameters | All jobs must use same resources |
| **Dependency chains** (`--dependency`) | Sequential multi-stage pipelines | Manual setup, fragile, no partial reruns |
| **Snakemake** | Multi-step pipelines with complex dependencies | Learning curve, initial setup time |

Use Snakemake when your workflow has:

- ✅ Multiple steps with input/output dependencies
- ✅ Steps that need different resources (CPU vs GPU)
- ✅ Need to rerun only failed or changed steps
- ✅ Reproducibility requirements
- ✅ Parameter sweeps combined with processing pipelines

## Installation on OSC

### Option 1: pip in a Virtual Environment (Recommended)

```bash
# Load Python
module load python/3.9-2022.05

# Create a dedicated venv
python -m venv ~/venvs/snakemake
source ~/venvs/snakemake/bin/activate

# Install Snakemake with SLURM support
pip install snakemake snakemake-executor-plugin-slurm

# Verify installation
snakemake --version
```

### Option 2: Conda

```bash
module load miniconda3
conda create -n snakemake -c conda-forge -c bioconda snakemake
conda activate snakemake

# Verify
snakemake --version
```

!!! note "Activate before running"
    Always activate your Snakemake environment before running pipelines:
    ```bash
    module load python/3.9-2022.05
    source ~/venvs/snakemake/bin/activate
    ```

## Snakefile Basics

### Your First Snakefile

Create a file named `Snakefile` in your project directory:

```python
# Snakefile

rule all:
    input:
        "results/analysis_complete.txt"

rule preprocess:
    input:
        "data/raw_data.csv"
    output:
        "data/processed_data.csv"
    shell:
        "python scripts/preprocess.py --input {input} --output {output}"

rule analyze:
    input:
        "data/processed_data.csv"
    output:
        "results/analysis_complete.txt"
    shell:
        "python scripts/analyze.py --input {input} --output {output}"
```

Key concepts:

- **`rule all`** — The default target rule. Its `input` lists all final output files.
- **`input` / `output`** — File dependencies. Snakemake builds a DAG (directed acyclic graph) from these.
- **`shell`** — The command to run. Use `{input}` and `{output}` placeholders.

### Rules, Inputs, and Outputs

Every rule defines a transformation from input files to output files:

```python
rule train_model:
    input:
        data="data/processed/{dataset}.csv",
        config="configs/train_config.yaml"
    output:
        model="models/{dataset}/model.pt",
        metrics="results/{dataset}/metrics.json"
    shell:
        """
        python scripts/train.py \
            --data {input.data} \
            --config {input.config} \
            --model-output {output.model} \
            --metrics-output {output.metrics}
        """
```

!!! tip "Named inputs and outputs"
    Use named inputs (`data=...`, `config=...`) when a rule has multiple files. Reference them as `{input.data}`, `{input.config}`, etc.

### Wildcards and `expand()`

Wildcards let you write generic rules that apply to many files:

```python
DATASETS = ["train", "val", "test"]

rule all:
    input:
        expand("results/{dataset}/metrics.json", dataset=DATASETS)

rule preprocess:
    input:
        "data/raw/{dataset}.csv"
    output:
        "data/processed/{dataset}.csv"
    shell:
        "python scripts/preprocess.py --input {input} --output {output}"
```

`expand()` generates all combinations of the wildcard values, so the `all` rule requests:

- `results/train/metrics.json`
- `results/val/metrics.json`
- `results/test/metrics.json`

### Configuration Files

Use a config file to separate parameters from workflow logic:

```yaml
# config.yaml
datasets:
  - train
  - val
  - test

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

Reference in your Snakefile:

```python
configfile: "config.yaml"

rule all:
    input:
        expand("results/{dataset}/metrics.json", dataset=config["datasets"])

rule train:
    input:
        "data/processed/{dataset}.csv"
    output:
        "results/{dataset}/metrics.json"
    params:
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
        bs=config["training"]["batch_size"]
    shell:
        """
        python scripts/train.py \
            --data {input} \
            --epochs {params.epochs} \
            --lr {params.lr} \
            --batch-size {params.bs} \
            --output {output}
        """
```

## SLURM Profile Configuration

A Snakemake SLURM profile tells Snakemake how to submit each rule as a SLURM job.

### Creating the Profile

```bash
# Create profile directory
mkdir -p ~/.config/snakemake/slurm
```

Create the profile configuration file:

```yaml
# ~/.config/snakemake/slurm/config.yaml

# Use the SLURM executor
executor: slurm

# Default resources for all rules
default-resources:
  slurm_account: "PAS1234"
  slurm_partition: "serial"
  mem_mb: 16000          # 16 GB
  runtime: 120           # 2 hours (in minutes)
  cpus_per_task: 4

# Job name format
jobs: 50                 # Max concurrent jobs
latency-wait: 60         # Wait 60s for output files to appear (NFS lag)
```

### Per-Rule Resource Overrides

Override resources for specific rules in your Snakefile:

```python
rule train_gpu:
    input:
        "data/processed/{dataset}.csv"
    output:
        "models/{dataset}/model.pt"
    resources:
        slurm_partition="gpu",
        mem_mb=64000,        # 64 GB
        runtime=480,         # 8 hours
        cpus_per_task=8,
        slurm_extra="--gpus-per-node=1"
    shell:
        """
        module load cuda/11.8.0
        python scripts/train.py --data {input} --output {output} --device cuda
        """
```

!!! warning "GPU rules need explicit partition"
    Always set `slurm_partition="gpu"` for rules that need GPUs. The default partition typically does not have GPU access.

### Module Loading in Rules

Load modules inside the `shell` block of each rule:

```python
rule train:
    input: "data/processed.csv"
    output: "models/model.pt"
    resources:
        slurm_partition="gpu",
        slurm_extra="--gpus-per-node=1"
    shell:
        """
        module load python/3.9-2022.05
        module load cuda/11.8.0
        source ~/venvs/pytorch/bin/activate
        python scripts/train.py --data {input} --output {output}
        """
```

## Running Pipelines on OSC

### Dry Run

Always start with a dry run to verify the workflow logic:

```bash
# Show what would be executed
snakemake -n

# With detailed reasons for each rule
snakemake -n -r
```

### Local Execution (Debug Queue)

For quick tests, request an interactive debug session and run locally:

```bash
# Get an interactive session
srun -p debug -c 4 --time=00:30:00 --pty bash

# Run the full pipeline locally (no SLURM submission)
snakemake --cores 4
```

### Cluster Execution with SLURM Profile

Submit each rule as a separate SLURM job:

```bash
# Run using the SLURM profile
snakemake --profile slurm

# Or with additional options
snakemake --profile slurm --jobs 20 --latency-wait 120
```

!!! tip "Run Snakemake from a login node"
    Unlike individual jobs, the Snakemake controller process should run on a **login node** (or in a `screen`/`tmux` session). It submits and monitors SLURM jobs on your behalf.

### Using `screen` or `tmux` for Long Pipelines

```bash
# Start a tmux session
tmux new -s pipeline

# Run your pipeline
source ~/venvs/snakemake/bin/activate
snakemake --profile slurm

# Detach: press Ctrl+B, then D
# Reattach later:
tmux attach -t pipeline
```

### Visualizing the DAG

Generate a visual representation of your pipeline:

```bash
# Generate DAG image
snakemake --dag | dot -Tpng > dag.png

# Generate a simplified rule graph
snakemake --rulegraph | dot -Tpng > rulegraph.png
```

!!! note "Graphviz required"
    The `dot` command requires Graphviz. Load it with `module load graphviz` on OSC.

## Practical Examples

### ML Training Pipeline

A complete pipeline that preprocesses data, trains a model, and evaluates it:

```python
# Snakefile
configfile: "config.yaml"

rule all:
    input:
        "results/evaluation_report.json"

rule preprocess:
    input:
        "data/raw_data.csv"
    output:
        train="data/processed/train.csv",
        val="data/processed/val.csv",
        test="data/processed/test.csv"
    resources:
        cpus_per_task=8,
        mem_mb=32000,
        runtime=60
    shell:
        """
        module load python/3.9-2022.05
        source ~/venvs/pytorch/bin/activate
        python scripts/preprocess.py \
            --input {input} \
            --train-output {output.train} \
            --val-output {output.val} \
            --test-output {output.test}
        """

rule train:
    input:
        train="data/processed/train.csv",
        val="data/processed/val.csv"
    output:
        "models/best_model.pt"
    resources:
        slurm_partition="gpu",
        cpus_per_task=8,
        mem_mb=64000,
        runtime=480,
        slurm_extra="--gpus-per-node=1"
    shell:
        """
        module load python/3.9-2022.05
        module load cuda/11.8.0
        source ~/venvs/pytorch/bin/activate
        python scripts/train.py \
            --train-data {input.train} \
            --val-data {input.val} \
            --output {output} \
            --epochs {config[training][epochs]}
        """

rule evaluate:
    input:
        model="models/best_model.pt",
        test="data/processed/test.csv"
    output:
        "results/evaluation_report.json"
    resources:
        slurm_partition="gpu",
        cpus_per_task=4,
        mem_mb=32000,
        runtime=60,
        slurm_extra="--gpus-per-node=1"
    shell:
        """
        module load python/3.9-2022.05
        module load cuda/11.8.0
        source ~/venvs/pytorch/bin/activate
        python scripts/evaluate.py \
            --model {input.model} \
            --test-data {input.test} \
            --output {output}
        """
```

### Hyperparameter Sweep

Use wildcards to sweep over hyperparameters:

```python
# Snakefile
LEARNING_RATES = ["0.001", "0.01", "0.1"]
BATCH_SIZES = ["16", "32", "64"]

rule all:
    input:
        "results/best_params.json"

rule train_variant:
    input:
        "data/processed/train.csv"
    output:
        "results/lr_{lr}_bs_{bs}/metrics.json"
    resources:
        slurm_partition="gpu",
        cpus_per_task=4,
        mem_mb=32000,
        runtime=240,
        slurm_extra="--gpus-per-node=1"
    shell:
        """
        module load python/3.9-2022.05
        module load cuda/11.8.0
        source ~/venvs/pytorch/bin/activate
        python scripts/train.py \
            --data {input} \
            --lr {wildcards.lr} \
            --batch-size {wildcards.bs} \
            --output-dir results/lr_{wildcards.lr}_bs_{wildcards.bs}/
        """

rule select_best:
    input:
        expand("results/lr_{lr}_bs_{bs}/metrics.json",
               lr=LEARNING_RATES, bs=BATCH_SIZES)
    output:
        "results/best_params.json"
    shell:
        """
        python scripts/select_best.py --results-dir results/ --output {output}
        """
```

### Data Processing Pipeline

Process multiple datasets through a shared pipeline:

```python
# Snakefile
SUBJECTS = glob_wildcards("data/raw/{subject}.csv").subject

rule all:
    input:
        "results/combined_report.csv"

rule clean:
    input:
        "data/raw/{subject}.csv"
    output:
        "data/cleaned/{subject}.csv"
    resources:
        cpus_per_task=2,
        mem_mb=8000,
        runtime=30
    shell:
        "python scripts/clean.py --input {input} --output {output}"

rule extract_features:
    input:
        "data/cleaned/{subject}.csv"
    output:
        "data/features/{subject}.csv"
    resources:
        cpus_per_task=4,
        mem_mb=16000,
        runtime=60
    shell:
        "python scripts/extract_features.py --input {input} --output {output}"

rule combine:
    input:
        expand("data/features/{subject}.csv", subject=SUBJECTS)
    output:
        "results/combined_report.csv"
    shell:
        "python scripts/combine.py --input-dir data/features/ --output {output}"
```

## Monitoring and Debugging

### Pipeline Summary

```bash
# See which rules need to run
snakemake --summary

# List all output files and their status
snakemake --list
```

### Log Files

Configure per-rule log files to capture output:

```python
rule train:
    input: "data/processed.csv"
    output: "models/model.pt"
    log:
        "logs/train.log"
    shell:
        """
        python scripts/train.py --data {input} --output {output} 2>&1 | tee {log}
        """
```

### Recovering from Failures

```bash
# Rerun only failed or incomplete jobs
snakemake --profile slurm --rerun-incomplete

# If Snakemake was interrupted and left a lock
snakemake --unlock

# Force rerun a specific rule
snakemake --profile slurm --forcerun train
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `IncompleteFilesException` | Previous run was interrupted | Run with `--rerun-incomplete` |
| `LockException` | Another Snakemake instance is running (or crashed) | Run `snakemake --unlock` |
| `MissingInputException` | An input file does not exist | Check file paths, ensure upstream rules ran |
| `AmbiguousRuleException` | Multiple rules can produce the same output | Make output patterns more specific |
| `CalledProcessError` | A shell command failed | Check the rule's log file for details |
| Output file not found after job completes | NFS filesystem lag | Increase `--latency-wait` to 120+ seconds |

## Best Practices

### 1. Keep Snakefiles Modular

Split large pipelines into separate Snakefiles:

```python
# Snakefile (main)
include: "rules/preprocess.smk"
include: "rules/training.smk"
include: "rules/evaluation.smk"

rule all:
    input:
        "results/final_report.json"
```

### 2. Use Config Files for Parameters

Never hardcode paths or hyperparameters in the Snakefile:

```python
configfile: "config.yaml"

# Access config values
config["training"]["epochs"]  # ✅ Good
```

### 3. Set Resource Defaults in the Profile

Use the SLURM profile for default resources and override only when needed:

```python
# Only specify resources that differ from the profile defaults
rule gpu_task:
    resources:
        slurm_partition="gpu",
        slurm_extra="--gpus-per-node=1"
```

### 4. Test Locally First

Run your pipeline on a small subset before submitting to SLURM:

```bash
# Test with a dry run
snakemake -n

# Run locally with limited cores
snakemake --cores 4
```

### 5. Version Control Your Pipelines

Commit these files to your repository:

- ✅ `Snakefile` and any `rules/*.smk` files
- ✅ `config.yaml`
- ✅ `~/.config/snakemake/slurm/config.yaml` (copy to project as `profiles/slurm/config.yaml`)
- ❌ Do not commit output data or log files

??? note "Per-Rule Conda Environments"

    For complex pipelines with conflicting dependencies, use per-rule conda environments:

    ```python
    rule special_analysis:
        input: "data/processed.csv"
        output: "results/special.csv"
        conda:
            "envs/special.yaml"
        shell:
            "python scripts/special_analysis.py --input {input} --output {output}"
    ```

## Troubleshooting

### Pipeline Hangs After Submitting Jobs

The Snakemake controller may lose connection. Use `tmux` or `screen` (see [Running Pipelines](#using-screen-or-tmux-for-long-pipelines)) and increase `--latency-wait`.

### Jobs Fail but Snakemake Shows No Error

Check SLURM logs directly:

```bash
# Find your recent jobs
sacct -u $USER --starttime=$(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,ExitCode

# Check the SLURM output for a specific job
cat slurm-<jobid>.out
```

### "Directory cannot be locked" Error

```bash
# Unlock the working directory
snakemake --unlock
```

## Next Steps

- Review [Job Submission](osc-job-submission.md) for SLURM fundamentals
- See the [ML Workflow Guide](../ml-workflows/ml-workflow.md) for organizing ML projects

## Resources

- [Snakemake Documentation](https://snakemake.readthedocs.io/)
- [Snakemake SLURM Executor Plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
