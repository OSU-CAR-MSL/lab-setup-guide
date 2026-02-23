<!-- last-reviewed: 2026-02-22 -->
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
    module load python/3.12
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
    module load python/3.12
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
    module load python/3.12
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

## Alternative: Ray

[Ray](https://docs.ray.io/) is a Python-native distributed compute framework — ideal when your entire pipeline is Python and you need GPU resource management, distributed HPO, and fault tolerance. It integrates with SLURM via `ray symmetric-run` (Ray 2.49+) for multi-node clusters.

**Why Ray for ML workloads:**

- Native SLURM integration — `ray symmetric-run` bootstraps a Ray cluster across SLURM-allocated nodes
- GPU resource management per task/actor — no manual `CUDA_VISIBLE_DEVICES` juggling
- Ray Tune for distributed hyperparameter optimization with early stopping
- Fault tolerance — tasks retry automatically on GPU failures
- Zero code change to scale from single-GPU to multi-node

### Setup

```bash
module load python/3.12
source ~/venvs/ml_project/bin/activate
pip install "ray[tune]" optuna
```

No OSC module needed — Ray runs entirely from your Python environment.

### Minimal Example

A training pipeline using `ray.remote` tasks submitted from a SLURM job:

```python
import ray

@ray.remote(num_gpus=1)
def train(data_path: str, config: dict) -> str:
    import torch
    # ... training logic using config ...
    torch.save(model.state_dict(), "model.pt")
    return "model.pt"

@ray.remote(num_gpus=1)
def evaluate(model_path: str, data_path: str) -> dict:
    # ... evaluation logic ...
    return {"f1": 0.95, "accuracy": 0.97}

if __name__ == "__main__":
    ray.init()  # Connects to Ray cluster (started by SLURM bootstrap)

    data = "data/processed.csv"
    config = {"lr": 1e-3, "hidden_dim": 128, "epochs": 50}

    model_ref = train.remote(data, config)
    metrics = ray.get(evaluate.remote(ray.get(model_ref), data))
    print(metrics)
```

To run on SLURM, submit via `sbatch` with Ray cluster bootstrap:

```bash
#!/bin/bash
#SBATCH --nodes=1 --gpus-per-node=1 --cpus-per-task=8
#SBATCH --time=04:00:00 --account=PAS1234

module load python/3.12
source ~/venvs/ml_project/bin/activate

python train_pipeline.py
```

For multi-node jobs, replace the Python call with `ray symmetric-run`:

```bash
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks=$SLURM_JOB_NUM_NODES \
    ray symmetric-run -- python train_pipeline.py
```

### Ray Tune for Hyperparameter Optimization

```python
from ray import tune
from ray.tune.search.optuna import OptunaSearch

def train_fn(config):
    import ray.train
    # ... build model from config, train one epoch at a time ...
    for epoch in range(100):
        loss, f1 = train_one_epoch(config)
        ray.train.report({"val_f1": f1, "val_loss": loss})

tuner = tune.Tuner(
    tune.with_resources(train_fn, {"gpu": 1, "cpu": 4}),
    tune_config=tune.TuneConfig(
        search_alg=OptunaSearch(metric="val_f1", mode="max"),
        scheduler=tune.schedulers.ASHAScheduler(
            max_t=100, grace_period=10, reduction_factor=3,
        ),
        num_samples=50,
        max_concurrent_trials=4,  # match available GPUs
    ),
    param_space={
        "lr": tune.loguniform(1e-5, 1e-2),
        "hidden_dim": tune.choice([64, 128, 256]),
        "dropout": tune.uniform(0.1, 0.5),
    },
)
results = tuner.fit()
print(f"Best F1: {results.get_best_result('val_f1', 'max').metrics['val_f1']}")
```

!!! tip "Single-node vs multi-node"
    For single-node jobs (1 GPU), `ray.init()` works out of the box — no special setup. For multi-node clusters, use `ray symmetric-run` in your SLURM script to bootstrap Ray across all allocated nodes automatically.

## Next Steps

- [Job Submission](osc-job-submission.md) — SLURM fundamentals, job arrays, dependency chains
- [ML Project Template](../ml-workflows/ml-workflow.md) — project structure for ML experiments
- [Data & Experiment Tracking](../ml-workflows/data-experiment-tracking.md) — MLflow, W&B, DVC
