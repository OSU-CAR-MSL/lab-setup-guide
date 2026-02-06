# GPU Computing Guide

Guide for effectively using GPUs on OSC for machine learning and scientific computing.

## GPU Resources on OSC

### Available GPU Types

#### Pitzer Cluster

| GPU Model | Memory | CUDA Cores | Best For | Quantity |
|-----------|--------|------------|----------|----------|
| NVIDIA V100 | 32 GB | 5120 | Training large models | Limited |
| NVIDIA A100 | 40 GB | 6912 | Latest ML workloads | Limited |

#### Owens Cluster (Older)

| GPU Model | Memory | CUDA Cores | Best For | Quantity |
|-----------|--------|------------|----------|----------|
| NVIDIA P100 | 16 GB | 3584 | General GPU work | Many |

### Which GPU to Use?

- **A100**: Latest architectures (Transformers, large models)
- **V100**: Most ML workloads, good balance
- **P100**: Older but widely available, good for testing

## Requesting GPUs

### Interactive GPU Session

```bash
# Request any available GPU
srun -p gpu --gpus-per-node=1 --time=01:00:00 --pty bash

# Request specific GPU type
srun -p gpu --gpus-per-node=v100:1 --time=01:00:00 --pty bash
srun -p gpu --gpus-per-node=a100:1 --time=01:00:00 --pty bash

# Request multiple GPUs
srun -p gpu --gpus-per-node=2 --time=01:00:00 --pty bash

# With more CPUs and memory
srun -p gpu --gpus-per-node=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

### Batch Job

```bash
#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1           # Number of GPUs
#SBATCH --gpus-per-node=v100:1      # Specific GPU type
#SBATCH --cpus-per-task=4           # CPUs (for data loading)
#SBATCH --mem=32G                   # Memory
#SBATCH --time=08:00:00

# Your GPU job commands
```

## GPU Management

### Check GPU Availability

```bash
# Check GPU nodes and availability
sinfo -p gpu

# See GPU jobs in queue
squeue -p gpu

# Your GPU jobs
squeue -u $USER -p gpu
```

### Monitor GPU Usage

#### Using nvidia-smi

```bash
# Basic GPU info
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Specific GPU
nvidia-smi -i 0

# Show processes
nvidia-smi pmon

# Detailed query
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

#### Using Python

```python
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# GPU information
print(f"Number of GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

# Memory usage
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

### GPU Selection

#### Using Environment Variable

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1

# In job script
#SBATCH --gpus-per-node=2
export CUDA_VISIBLE_DEVICES=0,1
```

#### In Python Code

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Or use device parameter
device = torch.device('cuda:0')
```

## Efficient GPU Usage

### 1. Data Loading Optimization

```python
from torch.utils.data import DataLoader

# Use multiple workers (match CPUs requested)
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,        # Match --cpus-per-task
    pin_memory=True,      # Faster GPU transfer
    prefetch_factor=2,    # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

### 2. Mixed Precision Training

Mixed precision uses FP16 where possible, saving memory and speeding up training.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        
        # Forward pass in mixed precision
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### 3. Gradient Accumulation

Simulate larger batch sizes without more GPU memory:

```python
accumulation_steps = 4  # Effective batch size = batch_size * 4

for i, (data, target) in enumerate(train_loader):
    data, target = data.cuda(), target.cuda()
    
    # Forward pass
    output = model(data)
    loss = criterion(output, target)
    
    # Scale loss and backward
    loss = loss / accumulation_steps
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Gradient Checkpointing

Save memory by recomputing activations during backward pass:

```python
import torch.utils.checkpoint as checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        return x
```

## Multi-GPU Training

### DataParallel (Simple, Single Node)

```python
import torch.nn as nn

model = MyModel()

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.cuda()

# Train as usual
for data, target in train_loader:
    output = model(data.cuda())  # Automatically distributed
    loss = criterion(output, target.cuda())
    loss.backward()
    optimizer.step()
```

Job script:
```bash
#SBATCH --gpus-per-node=4
```

### DistributedDataParallel (Recommended)

More efficient than DataParallel:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    # Create model and move to GPU
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Use DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=64)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for data, target in dataloader:
            data, target = data.to(rank), target.to(rank)
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
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

## Memory Management

### Check Memory Usage

```python
import torch

# Current GPU memory usage
print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Peak memory usage
print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

# Detailed memory summary
print(torch.cuda.memory_summary(device=0, abbreviated=False))
```

### Clear GPU Memory

```python
# Clear cache
torch.cuda.empty_cache()

# Delete tensors explicitly
del large_tensor
torch.cuda.empty_cache()

# Move to CPU and delete
large_tensor = large_tensor.cpu()
del large_tensor
torch.cuda.empty_cache()
```

### Set Memory Fraction

```python
# Use only 80% of GPU memory
torch.cuda.set_per_process_memory_fraction(0.8, device=0)
```

### Memory-Efficient Practices

```python
# 1. Use in-place operations
x.add_(y)  # Instead of x = x + y

# 2. Use del for large tensors
result = large_computation()
# Use result
del result
torch.cuda.empty_cache()

# 3. Use torch.no_grad() for inference
with torch.no_grad():
    output = model(input)

# 4. Clear gradients properly
optimizer.zero_grad(set_to_none=True)  # More memory efficient

# 5. Use gradient accumulation for large models
```

## Performance Optimization

### Profile Your Code

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i in range(10):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Save trace for visualization
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

### Benchmark Your Model

```python
import time
import torch

def benchmark(model, input_shape, num_iterations=100):
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        _ = model(dummy_input)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {1/avg_time:.2f} samples/sec")

# Usage
benchmark(model, (1, 3, 224, 224))
```

### Optimize Data Loading

```python
# Poor data loading (bottleneck)
for data, target in dataloader:  # Waits for CPU
    output = model(data.cuda())  # GPU sits idle
    
# Optimized data loading
for data, target in dataloader:
    data = data.cuda(non_blocking=True)  # Async transfer
    target = target.cuda(non_blocking=True)
    output = model(data)
```

## Common GPU Issues

### Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size**
   ```python
   batch_size = 32  # Instead of 64
   ```

2. **Use gradient accumulation**
   ```python
   accumulation_steps = 2  # Effective batch size = 64
   ```

3. **Use mixed precision**
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       output = model(input)
   ```

4. **Clear cache**
   ```python
   torch.cuda.empty_cache()
   ```

5. **Use gradient checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

6. **Reduce model size**
   ```python
   # Use smaller model or fewer layers
   ```

### GPU Not Detected

**Checks:**

```bash
# 1. Verify GPU requested
squeue -u $USER

# 2. Check node has GPU
nvidia-smi

# 3. Check CUDA module loaded
module list | grep cuda

# 4. Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Training

**Diagnose:**

```python
# Profile to find bottlenecks
import torch.profiler
with torch.profiler.profile() as prof:
    train_one_epoch()
print(prof.key_averages().table())
```

**Common issues:**
- Too few data loader workers
- Not using pin_memory
- Not using mixed precision
- Inefficient model architecture
- CPU-GPU transfer bottleneck

## Best Practices

1. **Always request appropriate GPU resources**
   - Don't over-request GPUs you won't use
   - Use debug queue for testing

2. **Monitor GPU usage during training**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Use mixed precision** when possible
   - Faster training
   - Less memory usage

4. **Optimize data loading**
   - Multiple workers
   - Pin memory
   - Prefetching

5. **Save checkpoints regularly**
   - GPU nodes can fail
   - Jobs have time limits

6. **Clear GPU cache** when switching tasks
   ```python
   torch.cuda.empty_cache()
   ```

7. **Profile before optimizing**
   - Find actual bottlenecks
   - Don't optimize blindly

8. **Test interactively before batch jobs**
   ```bash
   srun -p gpu --gpus-per-node=1 --pty bash
   ```

## Next Steps

- Review [ML Workflow Guide](ml-workflow.md)
- Learn [PyTorch Setup](pytorch-setup.md)
- Check [Job Submission Guide](../working-on-osc/osc-job-submission.md)
- Read [Best Practices](../working-on-osc/osc-best-practices.md)

## Resources

- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA GPU Optimization](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [Troubleshooting Guide](../resources/troubleshooting.md)
