# OSC Best Practices

Follow these best practices to work efficiently and be a good citizen on OSC shared resources.

## General Guidelines

### 1. Don't Run Jobs on Login Nodes

**Login nodes** are for:
- ✅ Editing code
- ✅ Compiling small programs
- ✅ Submitting jobs
- ✅ Light file operations
- ✅ Quick testing (< 30 seconds)

**Login nodes are NOT for**:
- ❌ Training ML models
- ❌ Running long computations
- ❌ Data processing jobs
- ❌ Anything using > 2 CPU cores
- ❌ Anything taking > 30 seconds

### 2. Use Compute Nodes for Real Work

Instead of running on login nodes:

```bash
# Request interactive session
srun -p debug -c 4 --time=01:00:00 --pty bash

# Or submit batch job
sbatch job_script.sh
```

### 3. Check System Status

Before starting work:
- Check [OSC System Status](https://www.osc.edu/supercomputing/system-status)
- Look for maintenance windows
- Check cluster load: `squeue`

## Resource Management

### Check Your Allocations

```bash
# View your project allocations
sbalance

# Check disk quota
quota -s

# See current usage
df -h $HOME
du -sh $HOME/*
```

### Monitor Your Jobs

```bash
# List your jobs
squeue -u $USER

# Detailed job info
scontrol show job <job_id>

# Job efficiency after completion
seff <job_id>
```

### Be Resource Conscious

1. **Request only what you need**
   ```bash
   # Bad: Requesting 48 cores for single-threaded code
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=48
   
   # Good: Request appropriate resources
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   ```

2. **Set realistic time limits**
   ```bash
   # Request slightly more than you need
   #SBATCH --time=02:00:00  # For ~1.5 hour job
   ```

3. **Use appropriate partition/queue**
   - `debug`: Quick testing (< 30 min)
   - `serial`: Single-node jobs
   - `parallel`: Multi-node jobs
   - `gpu`: GPU-accelerated jobs

## File Management

### Home Directory

- **Path**: `$HOME` or `~`
- **Quota**: 500 GB (check with `quota -s`)
- **Backed up**: Yes
- **Use for**: Code, scripts, small data
- **Don't use for**: Large datasets, temporary files

### Scratch Space

- **Path**: `$TMPDIR` (temporary) or `/fs/scratch/<project>/`
- **Quota**: Much larger
- **Backed up**: No
- **Use for**: Large datasets, temporary files, job outputs
- **Note**: Files may be deleted after 30-90 days of inactivity

### Project Space

- **Path**: `/fs/project/<project>/`
- **Quota**: Varies by project
- **Backed up**: Yes
- **Use for**: Shared data, important results
- **Access**: Shared with lab members

### Best Practices

```bash
# Use scratch for temporary large files
cd $TMPDIR
# Or
cd /fs/scratch/PAS1234/$USER

# Copy important results back to home
cp results.tar.gz $HOME/projects/

# Clean up scratch regularly
find /fs/scratch/PAS1234/$USER -mtime +30 -delete
```

## Environment Management

### Module System

OSC uses modules to manage software:

```bash
# List available modules
module avail

# Search for specific software
module spider python
module spider pytorch

# Load module
module load python/3.9

# List loaded modules
module list

# Unload module
module unload python

# Unload all modules
module purge
```

### Common Module Commands

```bash
# Python with conda
module load python/3.9-2022.05

# PyTorch with GPU
module load python/3.9-2022.05
module load cuda/11.8.0

# Check module info
module show python/3.9-2022.05
```

### Create Module Loading Scripts

Create `~/load_modules.sh`:

```bash
#!/bin/bash
module purge
module load python/3.9-2022.05
module load cuda/11.8.0
module load git
```

Use in job scripts:
```bash
source ~/load_modules.sh
```

## Python Best Practices

### Use Virtual Environments

```bash
# Load Python module
module load python/3.9-2022.05

# Create virtual environment
python -m venv ~/venvs/myproject

# Activate
source ~/venvs/myproject/bin/activate

# Install packages
pip install torch torchvision numpy

# Deactivate
deactivate
```

### Use Conda Environments (Alternative)

```bash
# Load conda
module load python/3.9-2022.05

# Create environment
conda create -n myproject python=3.9

# Activate
conda activate myproject

# Install packages
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch

# Deactivate
conda deactivate
```

### Install Packages to Your Home

```bash
# Install to user directory
pip install --user package_name

# Or use virtual environment (preferred)
source ~/venvs/myproject/bin/activate
pip install package_name
```

## Job Submission Best Practices

### 1. Test Interactively First

```bash
# Request interactive session
srun -p debug -c 4 --time=30:00 --pty bash

# Test your code
python train.py --epochs 1

# Exit when done
exit
```

### 2. Use Descriptive Job Names

```bash
#SBATCH --job-name=mnist_training
#SBATCH --output=logs/mnist_training_%j.out
```

### 3. Organize Output Files

```bash
# Create logs directory
mkdir -p logs checkpoints results

# In job script
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
```

### 4. Email Notifications

```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@osu.edu
```

### 5. Job Arrays for Multiple Runs

```bash
#SBATCH --array=1-10

# In script
python train.py --seed $SLURM_ARRAY_TASK_ID
```

## GPU Best Practices

### Request GPUs Appropriately

```bash
# Request specific number of GPUs
#SBATCH --gpus-per-node=1

# Request specific GPU type
#SBATCH --gpus-per-node=v100:1
#SBATCH --gpus-per-node=a100:1
```

### Monitor GPU Usage

```bash
# Check GPU availability
sinfo -p gpu

# Monitor GPU during job
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Verify GPU in Code

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
```

## Data Management

### 1. Compress Large Files

```bash
# Compress datasets
tar -czf dataset.tar.gz dataset/

# Compress with pigz (parallel)
tar -I pigz -cf dataset.tar.gz dataset/
```

### 2. Use Symbolic Links

```bash
# Link to shared dataset
ln -s /fs/project/PAS1234/datasets/imagenet ~/datasets/imagenet
```

### 3. Clean Up Regularly

```bash
# Find large files
du -sh ~/*/  | sort -hr | head -10

# Find old files
find ~/scratch -mtime +30 -ls

# Remove old checkpoints
find ~/checkpoints -name "epoch_*.pth" -mtime +7 -delete
```

### 4. Archive Old Projects

```bash
# Archive completed project
tar -czf project_backup.tar.gz project/
mv project_backup.tar.gz /fs/project/PAS1234/archives/
rm -rf project/
```

## Security Best Practices

### 1. Protect Your Data

```bash
# Set appropriate permissions
chmod 700 ~/projects/sensitive_data
chmod 600 ~/projects/sensitive_data/*
```

### 2. Don't Share Credentials

- Never put passwords in code
- Don't commit API keys to Git
- Use environment variables:
  ```bash
  # In .bashrc
  export API_KEY="your_key_here"
  
  # In code
  import os
  api_key = os.environ.get('API_KEY')
  ```

### 3. Use Git Carefully

```bash
# Check before committing
git status
git diff

# Don't commit sensitive files
# Add to .gitignore
echo "credentials.json" >> .gitignore
echo "*.key" >> .gitignore
```

## Collaboration Best Practices

### Share Code, Not Data

```bash
# Project directory structure
/fs/project/PAS1234/
├── datasets/           # Shared datasets
├── username1/          # Your work
└── username2/          # Collaborator's work
```

### Use Shared Environments

```bash
# Create shared conda environment
conda create -p /fs/project/PAS1234/envs/shared_env python=3.9

# All lab members can use it
conda activate /fs/project/PAS1234/envs/shared_env
```

### Document Your Work

- README.md in each project
- Comment your code
- Document environment setup
- Keep lab wiki updated

## Performance Tips

### 1. Use Appropriate Data Loaders

```python
# PyTorch: Use multiple workers
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Match CPU cores
    pin_memory=True  # For GPU training
)
```

### 2. Monitor Job Efficiency

```bash
# After job completes
seff <job_id>

# Look for:
# - CPU efficiency (should be > 80%)
# - Memory efficiency (should use what you requested)
```

### 3. Profile Your Code

```python
# Use Python profiler
import cProfile
cProfile.run('my_function()')

# Or PyTorch profiler
with torch.profiler.profile() as prof:
    train_one_epoch()
print(prof.key_averages().table())
```

## Troubleshooting Common Issues

### Job Pending Forever

```bash
# Check job details
squeue -u $USER
scontrol show job <job_id>

# Common reasons:
# - Requesting too many resources
# - Queue is full
# - Maintenance window approaching
```

### Out of Disk Space

```bash
# Check quota
quota -s

# Clean up
rm -rf ~/.cache/pip
rm -rf ~/.cache/torch
conda clean --all
```

### Module Not Found

```bash
# Search for module
module spider <name>

# Check dependencies
module spider <name>/<version>
```

## Quick Reference

### Essential Commands

```bash
# Job submission
sbatch job.sh              # Submit batch job
srun -p debug --pty bash   # Interactive session
scancel <job_id>           # Cancel job
squeue -u $USER            # Check your jobs

# Modules
module load python/3.9     # Load software
module list                # List loaded modules
module purge               # Unload all

# Resources
sbalance                   # Check allocations
quota -s                   # Check disk quota
seff <job_id>              # Job efficiency
```

## Next Steps

- Learn about [Job Submission](osc-job-submission.md)
- Set up [Environment Management](osc-environment-management.md)
- Read [PyTorch Setup Guide](../ml-workflows/pytorch-setup.md)
- Explore [ML Workflow Guide](../ml-workflows/ml-workflow.md)

## Resources

- [OSC Best Practices Documentation](https://www.osc.edu/resources/technical_support/supercomputers)
- [Batch System Documentation](https://www.osc.edu/resources/technical_support/supercomputers/slurm_migration)
- [Troubleshooting Guide](../resources/troubleshooting.md)
