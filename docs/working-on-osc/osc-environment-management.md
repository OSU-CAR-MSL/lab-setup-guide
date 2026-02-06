# Environment Management

Learn how to manage software environments on OSC using modules and virtual environments.

## Overview

OSC uses two main systems for environment management:
1. **Module System**: Load pre-installed software
2. **Virtual Environments**: Isolated Python environments with custom packages

## Module System

### What are Modules?

Modules are pre-installed software packages maintained by OSC. They handle:
- Software dependencies
- Environment variables
- Library paths
- Version management

### Basic Module Commands

```bash
# List available modules
module avail

# Search for specific software
module spider python
module spider pytorch
module spider cuda

# Load a module
module load python/3.9-2022.05

# Load specific version
module load cuda/11.8.0

# List loaded modules
module list

# Show module information
module show python/3.9-2022.05

# Unload a module
module unload python

# Unload all modules
module purge

# Swap modules
module swap python/3.8 python/3.9
```

### Commonly Used Modules

#### Python
```bash
# Python 3.9 with conda
module load python/3.9-2022.05

# Python 3.10
module load python/3.10-2022.05
```

#### CUDA (for GPU work)
```bash
# CUDA 11.8
module load cuda/11.8.0

# CUDA 12.1
module load cuda/12.1.0
```

#### Git
```bash
module load git
```

#### Other Useful Modules
```bash
module load cmake
module load gcc
module load intel
```

### Module Dependencies

Some modules load dependencies automatically:

```bash
# Load Python (includes conda)
module load python/3.9-2022.05

# Check what was loaded
module list
```

### Finding Module Information

```bash
# Search for software
module spider tensorflow

# Get detailed info
module spider tensorflow/2.10.0

# Example output shows:
# - Available versions
# - Dependencies required
# - How to load it
```

## Python Virtual Environments

### Why Use Virtual Environments?

- ✅ Isolate project dependencies
- ✅ Avoid package conflicts
- ✅ Reproducible environments
- ✅ Easy to recreate on different systems

### Creating Virtual Environments

#### Method 1: venv (Recommended)

```bash
# Load Python module
module load python/3.9-2022.05

# Create virtual environment
python -m venv ~/venvs/myproject

# Activate environment
source ~/venvs/myproject/bin/activate

# Verify activation (prompt changes)
which python
# Should show: /home/username/venvs/myproject/bin/python

# Install packages
pip install torch torchvision numpy pandas matplotlib

# Deactivate when done
deactivate
```

#### Method 2: conda

```bash
# Load conda
module load python/3.9-2022.05

# Create conda environment
conda create -n myproject python=3.9

# Activate
conda activate myproject

# Install packages
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch

# Deactivate
conda deactivate

# Remove environment
conda remove -n myproject --all
```

### Managing Environments

#### List Environments

```bash
# venv: Check directory
ls ~/venvs/

# conda: List environments
conda env list
```

#### Export/Import Environments

**venv:**
```bash
# Export requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

**conda:**
```bash
# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml
```

### Organizing Virtual Environments

Create a standard structure:

```bash
# Create venvs directory
mkdir -p ~/venvs

# Create project-specific environments
python -m venv ~/venvs/project1
python -m venv ~/venvs/project2
python -m venv ~/venvs/ml_research
```

## Using Environments in Job Scripts

### venv in SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --time=02:00:00

# Load module
module load python/3.9-2022.05

# Activate virtual environment
source ~/venvs/myproject/bin/activate

# Verify environment
which python
pip list

# Run code
python train.py

# Deactivate (optional, job ends anyway)
deactivate
```

### conda in SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --time=02:00:00

# Load conda
module load python/3.9-2022.05

# Activate conda environment
source activate myproject

# Or use conda activate
conda activate myproject

# Run code
python train.py
```

### Multiple Module Loading

```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --gpus-per-node=1

# Load all required modules
module purge                    # Start clean
module load python/3.9-2022.05
module load cuda/11.8.0
module load git

# Activate environment
source ~/venvs/pytorch_gpu/bin/activate

# Run code
python train_gpu.py
```

## Creating Reusable Module Scripts

### Load Modules Script

Create `~/scripts/load_ml_modules.sh`:

```bash
#!/bin/bash
# Load modules for ML work

module purge
module load python/3.9-2022.05
module load cuda/11.8.0
module load git

echo "Modules loaded for ML work"
module list
```

Use in job scripts:
```bash
source ~/scripts/load_ml_modules.sh
source ~/venvs/pytorch/bin/activate
python train.py
```

### Activation Script

Create `~/scripts/activate_ml_env.sh`:

```bash
#!/bin/bash
# Complete environment setup for ML

# Load modules
module purge
module load python/3.9-2022.05
module load cuda/11.8.0

# Activate virtual environment
source ~/venvs/ml_project/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "Environment activated"
```

Use it:
```bash
source ~/scripts/activate_ml_env.sh
```

## Advanced: Shared Environments

For lab collaboration, create shared environments:

### Shared Conda Environment

```bash
# Create in project space
conda create -p /fs/project/PAS1234/envs/lab_shared python=3.9

# All lab members can activate
conda activate /fs/project/PAS1234/envs/lab_shared

# Install packages (requires write access)
conda install pytorch torchvision cudatoolkit=11.8 -c pytorch
```

### Shared venv (Alternative)

```bash
# Create in project space
python -m venv /fs/project/PAS1234/envs/lab_shared

# Set permissions (if needed)
chmod -R g+rwX /fs/project/PAS1234/envs/lab_shared

# Activate
source /fs/project/PAS1234/envs/lab_shared/bin/activate
```

## PyTorch Environment Setup

### Complete PyTorch Setup

```bash
# 1. Load modules
module load python/3.9-2022.05
module load cuda/11.8.0

# 2. Create environment
python -m venv ~/venvs/pytorch

# 3. Activate
source ~/venvs/pytorch/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Install additional packages
pip install numpy pandas matplotlib scikit-learn
pip install jupyter tensorboard wandb

# 7. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Test PyTorch GPU

Create `test_gpu.py`:
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor on GPU
    x = torch.rand(5, 3).cuda()
    print(f"Tensor on GPU: {x.device}")
```

Run:
```bash
# Interactive GPU session
srun -p gpu --gpus-per-node=1 --pty bash

# Activate environment
source ~/venvs/pytorch/bin/activate

# Test
python test_gpu.py
```

## Managing Disk Space

### Check Environment Sizes

```bash
# Check venv sizes
du -sh ~/venvs/*

# Check conda environments
conda clean --all --dry-run
```

### Clean Up

```bash
# Remove unused venv
rm -rf ~/venvs/old_project

# Clean conda cache
conda clean --all

# Clean pip cache
pip cache purge

# Remove old packages
pip uninstall <package>
```

### Disk Quota

```bash
# Check your quota
quota -s

# Find large directories
du -sh ~/*/  | sort -hr | head -10
```

## .bashrc Configuration

Add to `~/.bashrc` for automatic setup:

```bash
# Load commonly used modules
module load python/3.9-2022.05

# Alias for activating environments
alias activate-ml='source ~/venvs/ml_project/bin/activate'
alias activate-pytorch='source ~/venvs/pytorch/bin/activate'

# Environment variables
export PYTHONUNBUFFERED=1
```

**Warning**: Don't load modules that conflict or slow down login.

## Best Practices

### 1. Use Descriptive Environment Names

```bash
# Good
~/venvs/pytorch_gpu_project
~/venvs/transformers_research
~/venvs/computer_vision

# Avoid
~/venvs/env1
~/venvs/test
~/venvs/myenv
```

### 2. Document Your Environment

Create `environment_setup.md` in your project:

```markdown
# Environment Setup

## Modules
- python/3.9-2022.05
- cuda/11.8.0

## Virtual Environment
Location: ~/venvs/myproject

## Installation
\`\`\`bash
module load python/3.9-2022.05 cuda/11.8.0
python -m venv ~/venvs/myproject
source ~/venvs/myproject/bin/activate
pip install -r requirements.txt
\`\`\`
```

### 3. Keep requirements.txt Updated

```bash
# Update requirements file
pip freeze > requirements.txt

# Commit to git
git add requirements.txt
git commit -m "Update dependencies"
```

### 4. Use Separate Environments per Project

Don't share environments between unrelated projects:

```bash
# Per-project environments
~/venvs/
  ├── project_A/
  ├── project_B/
  └── project_C/
```

### 5. Test Before Submitting Jobs

```bash
# Test interactively first
srun -p debug --pty bash
source ~/venvs/myproject/bin/activate
python train.py --epochs 1

# Then submit batch job
```

## Troubleshooting

### Module Not Found

**Problem**: `module: command not found`

**Solution**: Module system not initialized. Check `/etc/profile.d/modules.sh` is sourced.

### Module Load Fails

**Problem**: `Module xyz not found`

**Solutions**:
```bash
# Search for correct name
module spider xyz

# Check available versions
module avail xyz
```

### Virtual Environment Not Activating

**Problem**: Environment won't activate

**Solutions**:
```bash
# Verify environment exists
ls ~/venvs/myproject/bin/activate

# Recreate if corrupted
rm -rf ~/venvs/myproject
python -m venv ~/venvs/myproject
```

### Package Installation Fails

**Problem**: `pip install` fails

**Solutions**:
```bash
# Update pip
pip install --upgrade pip

# Install with user flag
pip install --user package_name

# Check disk quota
quota -s
```

### CUDA Not Available in PyTorch

**Problem**: `torch.cuda.is_available()` returns False

**Solutions**:
```bash
# Verify CUDA module loaded
module list | grep cuda

# Load CUDA module
module load cuda/11.8.0

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Test on GPU node
srun -p gpu --gpus-per-node=1 --pty bash
```

### Conflicting Modules

**Problem**: Modules conflict with each other

**Solution**:
```bash
# Start fresh
module purge

# Load modules in correct order
module load python/3.9-2022.05
module load cuda/11.8.0
```

## Quick Reference

### Essential Commands

```bash
# Modules
module load python/3.9-2022.05   # Load Python
module load cuda/11.8.0          # Load CUDA
module list                      # Show loaded
module purge                     # Unload all

# venv
python -m venv ~/venvs/name      # Create
source ~/venvs/name/bin/activate # Activate
deactivate                       # Deactivate
pip freeze > requirements.txt    # Export

# conda
conda create -n name python=3.9  # Create
conda activate name              # Activate
conda deactivate                 # Deactivate
conda env export > env.yml       # Export
```

## Next Steps

- Set up [PyTorch on OSC](../ml-workflows/pytorch-setup.md)
- Learn [Job Submission](osc-job-submission.md)
- Read [ML Workflow Guide](../ml-workflows/ml-workflow.md)
- Review [Best Practices](osc-best-practices.md)

## Resources

- [OSC Module System](https://www.osc.edu/resources/technical_support/supercomputers/modules)
- [Python venv Documentation](https://docs.python.org/3/library/venv.html)
- [Conda Documentation](https://docs.conda.io/)
- [Troubleshooting Guide](../resources/troubleshooting.md)
