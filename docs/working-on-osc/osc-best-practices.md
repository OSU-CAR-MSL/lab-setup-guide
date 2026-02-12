# OSC Best Practices

Follow these guidelines to work efficiently and be a good citizen on OSC's shared resources.

## Don't Run Jobs on Login Nodes

**Login nodes** are shared by all users and are only for:

- Editing code and submitting jobs
- Light file operations and quick tests (< 30 seconds)

**Never** run training, data processing, or anything using more than 2 CPU cores on login nodes. Use compute nodes instead:

```bash
# Request interactive session
srun -p debug -c 4 --time=01:00:00 --pty bash

# Or submit batch job
sbatch job_script.sh
```

See the [Job Submission Guide](osc-job-submission.md) for full details on interactive and batch jobs.

## Be Resource-Conscious

1. **Request only what you need** — don't ask for 48 cores for single-threaded code
2. **Set realistic time limits** — request slightly more than expected, not the partition maximum
3. **Use the right partition** — `debug` for quick tests, `serial` for CPU work, `gpu` for GPU work
4. **Check job efficiency** after completion:
   ```bash
   seff <job_id>
   # CPU efficiency should be > 80%
   ```

For cluster specs and partition details, see the [Clusters Overview](../osc-basics/osc-clusters-overview.md).

## Check Your Resources

```bash
# View your project allocations
sbalance

# Check disk quota
quota -s

# See your running jobs
squeue -u $USER
```

## File Management

| Location | Path | Backed Up | Use For |
|----------|------|-----------|---------|
| Home | `$HOME` | Yes | Code, scripts, small files |
| Scratch | `/fs/scratch/<project>/` | No | Large datasets, temp files |
| Project | `/fs/project/<project>/` | Yes | Shared data, important results |

!!! warning "Scratch files are purged"
    Files on scratch may be deleted after 30-90 days of inactivity. Copy important results back to home or project space.

For transferring files to/from OSC, see [File Transfer](../osc-basics/osc-file-transfer.md). For storage quotas and cluster details, see [Clusters Overview](../osc-basics/osc-clusters-overview.md).

## Security

- **Set appropriate permissions** on sensitive data: `chmod 700 ~/projects/sensitive_data`
- **Never put passwords or API keys in code** — use environment variables
- **Don't commit secrets to Git** — add credentials to `.gitignore`

```bash
# Use environment variables for secrets
export API_KEY="your_key_here"
```

```python
import os
api_key = os.environ.get('API_KEY')
```

## Collaboration

```
/fs/project/PAS1234/
├── datasets/           # Shared datasets
├── username1/          # Your work
└── username2/          # Collaborator's work
```

- Use shared project directories for datasets everyone needs
- Create shared conda environments in project space when collaborating:
  ```bash
  conda create -p /fs/project/PAS1234/envs/shared_env python=3.9
  ```
- Keep a README in each project and document environment setup

For environment setup details, see [Environment Management](osc-environment-management.md).

## Essential Commands Quick Reference

| Task | Command |
|------|---------|
| Submit batch job | `sbatch job.sh` |
| Interactive session | `srun -p debug --pty bash` |
| Cancel job | `scancel <job_id>` |
| Check your jobs | `squeue -u $USER` |
| Load software | `module load python/3.9` |
| List loaded modules | `module list` |
| Unload all modules | `module purge` |
| Check allocations | `sbalance` |
| Check disk quota | `quota -s` |
| Job efficiency | `seff <job_id>` |

## Next Steps

- Learn about [Job Submission](osc-job-submission.md)
- Set up [Environment Management](osc-environment-management.md)
- Read [PyTorch & GPU Setup](../ml-workflows/pytorch-setup.md)
- Explore [ML Workflow Guide](../ml-workflows/ml-workflow.md)
