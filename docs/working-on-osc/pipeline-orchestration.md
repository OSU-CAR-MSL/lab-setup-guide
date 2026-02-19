<!-- last-reviewed: 2026-02-19 -->
# Pipeline Orchestration

Automate multi-step computational pipelines on OSC with proper dependency tracking, resource allocation, and failure recovery.

## When You Need an Orchestrator

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **Single `sbatch` script** | One-off jobs, simple tasks | No dependency tracking, manual reruns |
| **Job arrays** | Many identical jobs with different parameters | All jobs must use same resources |
| **Dependency chains** (`--dependency`) | Sequential multi-stage pipelines | Manual setup, fragile, no partial reruns |
| **Pipeline orchestrator** | Multi-step pipelines with complex dependencies | Initial setup time |

Use an orchestrator when your workflow has:

- Multiple steps with input/output dependencies
- Steps that need different resources (CPU vs GPU)
- Need to rerun only failed or changed steps
- Parameter sweeps combined with processing pipelines

For simpler job patterns (single jobs, job arrays, dependency chains), see [Job Submission](osc-job-submission.md).

## Nextflow (Recommended)

Nextflow is available as an OSC module — no installation needed. It integrates natively with SLURM and handles job submission, dependency tracking, and failure recovery.

### Setup

```bash
module load nextflow
nextflow -version
```

### Pipeline Structure

A Nextflow pipeline is a `.nf` file defining processes (steps) and a workflow that connects them. Create `main.nf`:

```groovy
#!/usr/bin/env nextflow

params.raw_data = "data/raw_data.csv"
params.outdir   = "results"

process PREPROCESS {
    cpus 4
    memory '16 GB'
    time '1h'

    input:
    path raw_data

    output:
    path "processed.csv"

    script:
    """
    module load python/3.9-2022.05
    source ~/venvs/ml_project/bin/activate
    python scripts/preprocess.py --input ${raw_data} --output processed.csv
    """
}

process TRAIN {
    cpus 8
    memory '64 GB'
    time '8h'
    clusterOptions '--gpus-per-node=1 --partition=gpu'

    input:
    path data

    output:
    path "model.pt"

    script:
    """
    module load python/3.9-2022.05
    module load cuda/11.8.0
    source ~/venvs/ml_project/bin/activate
    python scripts/train.py --data ${data} --output model.pt
    """
}

process EVALUATE {
    cpus 4
    memory '32 GB'
    time '1h'
    clusterOptions '--gpus-per-node=1 --partition=gpu'

    input:
    path model
    path data

    output:
    path "metrics.json"

    publishDir params.outdir, mode: 'copy'

    script:
    """
    module load python/3.9-2022.05
    module load cuda/11.8.0
    source ~/venvs/ml_project/bin/activate
    python scripts/evaluate.py --model ${model} --data ${data} --output metrics.json
    """
}

workflow {
    raw = Channel.fromPath(params.raw_data)
    processed = PREPROCESS(raw)
    model = TRAIN(processed)
    EVALUATE(model, processed)
}
```

### SLURM Configuration

Create `nextflow.config` in your project root:

```groovy
process {
    executor = 'slurm'
    queue    = 'serial'              // default partition
    account  = 'PAS1234'            // your OSC project
    time     = '2h'                 // default walltime
    memory   = '16 GB'
    cpus     = 4
}

executor {
    queueSize   = 50                // max concurrent SLURM jobs
    pollInterval = '30 sec'
}

// Where Nextflow stores intermediate files
workDir = '/fs/scratch/PAS1234/$USER/nf-work'
```

Per-process resources in `main.nf` override these defaults, so GPU processes automatically get the right partition and resources.

### Running Pipelines

```bash
# Dry run — show what would execute
nextflow main.nf -preview

# Run the pipeline (Nextflow submits SLURM jobs for you)
nextflow main.nf

# Resume after a failure (only reruns failed/changed steps)
nextflow main.nf -resume

# Run with different parameters
nextflow main.nf --raw_data data/experiment2.csv --outdir results/exp2
```

!!! tip "Run Nextflow from a login node or tmux session"
    The Nextflow controller process runs on the login node and submits SLURM jobs on your behalf. Use `tmux` for long pipelines so the controller survives SSH disconnects:
    ```bash
    tmux new -s pipeline
    nextflow main.nf
    # Detach: Ctrl+B, then D
    # Reattach: tmux attach -t pipeline
    ```

### Monitoring

```bash
# Watch pipeline progress (built-in)
# Nextflow prints a live progress table to the terminal

# View execution report after completion
nextflow log last

# Generate HTML execution report
nextflow main.nf -with-report report.html
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Process fails but pipeline continues | Default error strategy | Add `errorStrategy 'retry'` with `maxRetries 2` to process |
| Output files not found | NFS lag after SLURM job completes | Add `beforeScript 'sleep 10'` or increase poll interval |
| Stale work directory | Previous run left partial results | Delete `work/` or use `-resume` |
| `No such variable` | Bash variable conflicts with Nextflow | Escape shell variables: `\$USER` instead of `$USER` |

## Alternative: Prefect

[Prefect](https://www.prefect.io/) is a Python-native orchestrator — useful if your entire pipeline is Python and you want programmatic control over task dependencies. It requires more setup than Nextflow but integrates well with Python ML tooling.

### Setup

```bash
module load python/3.9-2022.05
source ~/venvs/ml_project/bin/activate
pip install prefect dask-jobqueue
```

### Minimal Example

```python
from prefect import flow, task
from prefect_dask import DaskTaskRunner
from dask_jobqueue import SLURMCluster

@task
def preprocess(raw_path: str) -> str:
    # ... preprocessing logic ...
    return "data/processed.csv"

@task
def train(data_path: str) -> str:
    # ... training logic ...
    return "models/best_model.pt"

@task
def evaluate(model_path: str, data_path: str) -> dict:
    # ... evaluation logic ...
    return {"accuracy": 0.95}

@flow(task_runner=DaskTaskRunner(
    cluster_class=SLURMCluster,
    cluster_kwargs={
        "account": "PAS1234",
        "cores": 4,
        "memory": "16GB",
        "walltime": "02:00:00",
    },
    adapt_kwargs={"minimum": 1, "maximum": 10},
))
def ml_pipeline(raw_data: str = "data/raw.csv"):
    processed = preprocess(raw_data)
    model = train(processed)
    evaluate(model, processed)

if __name__ == "__main__":
    ml_pipeline()
```

!!! note "Prefect server"
    For the Prefect UI and run history, start a local server in a tmux session: `prefect server start`. For simple pipelines, you can skip the server and run flows directly.

## Next Steps

- [Job Submission](osc-job-submission.md) — SLURM fundamentals, job arrays, dependency chains
- [ML Project Template](../ml-workflows/ml-workflow.md) — project structure for ML experiments
- [Data & Experiment Tracking](../ml-workflows/data-experiment-tracking.md) — MLflow, W&B, DVC
