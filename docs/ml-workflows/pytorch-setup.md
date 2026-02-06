# PyTorch Setup on OSC

This guide walks you through setting up PyTorch with GPU support on OSC clusters.

## Prerequisites

- OSC account with GPU access
- SSH connection configured
- Basic familiarity with Python and PyTorch

## Quick Setup

For those who want to get started quickly:

```bash
# 1. Load modules
module load python/3.9-2022.05 cuda/11.8.0

# 2. Create virtual environment
python -m venv ~/venvs/pytorch

# 3. Activate
source ~/venvs/pytorch/bin/activate

# 4. Install PyTorch with CUDA 11.8
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install common ML packages
pip install numpy pandas matplotlib scikit-learn jupyter tensorboard
```

## Detailed Setup

### Step 1: Choose CUDA Version

Check available CUDA versions:

```bash
module spider cuda
```

Common options:
- **CUDA 11.8**: Stable, widely supported
- **CUDA 12.1**: Newer, for latest features

For PyTorch, **CUDA 11.8** is recommended (as of 2024).

### Step 2: Load Required Modules

```bash
# Start with clean environment
module purge

# Load Python and CUDA
module load python/3.9-2022.05
module load cuda/11.8.0

# Verify
module list
```

### Step 3: Create Virtual Environment

```bash
# Create environment
python -m venv ~/venvs/pytorch

# Activate
source ~/venvs/pytorch/bin/activate

# Verify Python location
which python
# Should show: /home/username/venvs/pytorch/bin/python
```

### Step 4: Install PyTorch

#### For CUDA 11.8

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.1

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU-Only Version (Not Recommended)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install Additional Packages

```bash
# Core scientific packages
pip install numpy pandas matplotlib seaborn

# Machine learning utilities
pip install scikit-learn scipy

# Deep learning tools
pip install tensorboard wandb

# Jupyter for notebooks
pip install jupyter ipykernel

# Computer vision
pip install opencv-python pillow

# NLP (if needed)
pip install transformers datasets tokenizers

# Optimization and monitoring
pip install tqdm pytorch-lightning
```

### Step 6: Verify Installation

Create `test_pytorch.py`:

```python
import torch
import torchvision

print("=" * 50)
print("PyTorch Installation Test")
print("=" * 50)

print(f"\nPyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations
    print("\nTesting GPU operations...")
    x = torch.rand(5, 3)
    print(f"CPU tensor shape: {x.shape}")
    
    x_gpu = x.cuda()
    print(f"GPU tensor device: {x_gpu.device}")
    print("✓ GPU operations working!")
else:
    print("\n⚠ CUDA not available. Running on CPU.")

print("\n" + "=" * 50)
```

Test on GPU node:

```bash
# Request GPU node
srun -p gpu --gpus-per-node=1 --time=00:10:00 --pty bash

# Activate environment
module load python/3.9-2022.05 cuda/11.8.0
source ~/venvs/pytorch/bin/activate

# Run test
python test_pytorch.py

# Exit
exit
```

## Alternative: Conda Setup

If you prefer conda:

```bash
# Load conda
module load python/3.9-2022.05

# Create conda environment
conda create -n pytorch python=3.9 -y

# Activate
conda activate pytorch

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install additional packages
conda install numpy pandas matplotlib scikit-learn jupyter -y
```

## Creating a Requirements File

After installation, save your environment:

```bash
# Activate environment
source ~/venvs/pytorch/bin/activate

# Create requirements file
pip freeze > requirements.txt

# Review and clean up (remove unnecessary packages)
nano requirements.txt
```

Example cleaned `requirements.txt`:

```txt
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
scikit-learn==1.3.0
tensorboard==2.14.0
jupyter==1.0.0
tqdm==4.65.0
```

## Using PyTorch in Jobs

### Interactive GPU Session

```bash
# Request GPU for interactive work
srun -p gpu --gpus-per-node=1 --time=01:00:00 --pty bash

# Load modules and activate environment
module load python/3.9-2022.05 cuda/11.8.0
source ~/venvs/pytorch/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Work interactively
python
>>> import torch
>>> x = torch.rand(5, 3).cuda()
>>> print(x)
```

### Batch Job Script

Create `pytorch_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_train
#SBATCH --account=PAS1234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@osu.edu

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load python/3.9-2022.05
module load cuda/11.8.0

# Activate environment
source ~/venvs/pytorch/bin/activate

# Verify GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run training
python train.py \
    --data-path /fs/scratch/PAS1234/$USER/data \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --device cuda

echo "Job completed at: $(date)"
```

Submit:
```bash
mkdir -p logs
sbatch pytorch_job.sh
```

## Multi-GPU Training

### DataParallel (Single Node)

```python
import torch
import torch.nn as nn

model = MyModel()

# Use all available GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.cuda()
```

Job script:
```bash
#SBATCH --gpus-per-node=4  # Request 4 GPUs
```

### DistributedDataParallel (Recommended)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Training code here
    
    cleanup()
```

Job script:
```bash
#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_ddp.py
```

## GPU Selection and Management

### Set Specific GPU

```bash
# Environment variable
export CUDA_VISIBLE_DEVICES=0

# In Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### Check GPU Usage

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or in Python
import torch
torch.cuda.memory_summary(device=0)
```

### Memory Management

```python
import torch

# Clear cache
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8, device=0)

# Mixed precision training (saves memory)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Optimizing Performance

### Data Loading

```python
# Use multiple workers (match CPU cores)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,      # Match --cpus-per-task
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            output = model(data.cuda())
            loss = criterion(output, target.cuda())
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Gradient Accumulation

```python
# Effective batch size = batch_size * accumulation_steps
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Checkpointing

### Save Checkpoints

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# Save every N epochs
if epoch % 10 == 0:
    save_checkpoint(
        model, optimizer, epoch, loss,
        f'checkpoints/epoch_{epoch}.pth'
    )

# Save best model
if loss < best_loss:
    save_checkpoint(
        model, optimizer, epoch, loss,
        'checkpoints/best_model.pth'
    )
```

### Load Checkpoints

```python
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

# Resume training
if os.path.exists('checkpoints/best_model.pth'):
    epoch, loss = load_checkpoint(model, optimizer, 'checkpoints/best_model.pth')
    print(f"Resumed from epoch {epoch}")
```

## Troubleshooting

### CUDA Out of Memory

**Solutions**:
```python
# 1. Reduce batch size
batch_size = 32  # Instead of 64

# 2. Use gradient accumulation
accumulation_steps = 2

# 3. Clear cache
torch.cuda.empty_cache()

# 4. Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)

# 5. Use gradient checkpointing
from torch.utils.checkpoint import checkpoint
```

### CUDA Not Available

**Checks**:
```bash
# 1. Verify CUDA module loaded
module list | grep cuda

# 2. Check GPU requested in job
squeue -u $USER

# 3. Verify on GPU node
nvidia-smi

# 4. Reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Slow Training

**Solutions**:
```python
# 1. Use more workers
num_workers=4

# 2. Pin memory
pin_memory=True

# 3. Use mixed precision
from torch.cuda.amp import autocast, GradScaler

# 4. Profile code
import torch.profiler
with torch.profiler.profile() as prof:
    train_one_epoch()
print(prof.key_averages().table())
```

### Module Import Errors

```bash
# Verify environment activated
which python  # Should point to venv

# Reinstall package
pip install --force-reinstall torch

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

## Best Practices

1. **Always use virtual environments**
2. **Test on GPU node before batch submission**
3. **Save checkpoints regularly**
4. **Use mixed precision for faster training**
5. **Monitor GPU usage** with `nvidia-smi`
6. **Clear GPU cache** when needed
7. **Use appropriate batch size** for your GPU memory
8. **Pin memory** for faster data loading
9. **Use multiple workers** for data loading
10. **Document your environment** in requirements.txt

## Next Steps

- Read [ML Workflow Guide](ml-workflow.md)
- Learn [GPU Computing Guide](gpu-computing.md)
- Review [Job Submission](../working-on-osc/osc-job-submission.md)
- Check [Best Practices](../working-on-osc/osc-best-practices.md)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Troubleshooting Guide](../resources/troubleshooting.md)
