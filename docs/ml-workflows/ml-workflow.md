# Machine Learning Workflow Guide

Best practices and workflows for running machine learning experiments on OSC.

## Overview

This guide covers the complete ML workflow on OSC:
1. Environment setup
2. Data management
3. Code organization
4. Training and experimentation
5. Results tracking and analysis

## Project Structure

### Recommended Directory Layout

```
~/projects/my_ml_project/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── environment_setup.md      # Environment setup instructions
├── .gitignore               # Git ignore file
├── data/                    # Small data files, data scripts
│   ├── download_data.sh
│   └── preprocess.py
├── src/                     # Source code
│   ├── __init__.py
│   ├── models/              # Model definitions
│   │   ├── __init__.py
│   │   └── resnet.py
│   ├── data/                # Data loading
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── train.py             # Training script
├── scripts/                 # SLURM job scripts
│   ├── train_baseline.sh
│   └── hyperparameter_search.sh
├── configs/                 # Configuration files
│   ├── default.yaml
│   └── experiment1.yaml
├── notebooks/               # Jupyter notebooks for analysis
│   └── analysis.ipynb
├── tests/                   # Unit tests
│   └── test_model.py
├── logs/                    # Training logs
├── checkpoints/             # Model checkpoints
└── results/                 # Experiment results
```

### Data Organization

```
/fs/scratch/PAS1234/$USER/
├── datasets/               # Large datasets
│   ├── imagenet/
│   ├── cifar10/
│   └── custom_dataset/
└── my_ml_project/         # Project-specific data
    ├── processed_data/
    ├── checkpoints/       # Model checkpoints
    └── results/           # Experiment outputs
```

## Complete Workflow

### 1. Setup Phase

#### A. Create Project Structure

```bash
# Create project
mkdir -p ~/projects/my_ml_project
cd ~/projects/my_ml_project

# Create directory structure
mkdir -p src/{models,data,utils} scripts configs notebooks tests logs checkpoints results
touch README.md requirements.txt .gitignore
```

#### B. Initialize Git

```bash
git init
git add README.md .gitignore
git commit -m "Initial commit"

# If using remote repository
git remote add origin <your-repo-url>
git push -u origin main
```

#### C. Setup Environment

```bash
# Load modules
module load python/3.9-2022.05 cuda/11.8.0

# Create virtual environment
python -m venv ~/venvs/my_ml_project
source ~/venvs/my_ml_project/bin/activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib tensorboard
pip freeze > requirements.txt
```

#### D. Create .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/

# Data and models
data/raw/
data/processed/
*.pth
*.ckpt
checkpoints/
*.h5
*.hdf5

# Logs and results
logs/
results/
*.log

# Notebooks
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### 2. Data Preparation

#### Download and Prepare Data

Create `data/prepare_data.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_data_%j.out

# Use scratch space
SCRATCH_DIR=/fs/scratch/PAS1234/$USER/my_ml_project
mkdir -p $SCRATCH_DIR/datasets

# Download data (example)
cd $SCRATCH_DIR/datasets
wget https://example.com/dataset.tar.gz
tar -xzf dataset.tar.gz

# Preprocess
module load python/3.9-2022.05
source ~/venvs/my_ml_project/bin/activate
python ~/projects/my_ml_project/data/preprocess.py \
    --input $SCRATCH_DIR/datasets/raw \
    --output $SCRATCH_DIR/datasets/processed

echo "Data preparation complete"
```

#### Create Dataset Class

`src/data/dataset.py`:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load and organize your data
        samples = []
        # ... your data loading logic
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load and transform sample
        sample, label = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
```

### 3. Model Development

#### Define Model

`src/models/model.py`:

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ... more layers
        )
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### 4. Training Script

#### Basic Training Script

`src/train.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm

from models.model import MyModel
from data.dataset import CustomDataset

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100. * correct / total

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create datasets
    train_dataset = CustomDataset(args.train_data, transform=train_transform)
    val_dataset = CustomDataset(args.val_data, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = MyModel(num_classes=args.num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # TensorBoard
    writer = SummaryWriter(f'runs/{args.experiment_name}')
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, f'checkpoints/{args.experiment_name}_best.pth')
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/{args.experiment_name}_epoch_{epoch}.pth')
    
    writer.close()
    print(f'Training complete. Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, default='experiment')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    main(args)
```

### 5. Job Script

`scripts/train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ml_train
#SBATCH --account=PAS1234
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@osu.edu

echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load python/3.9-2022.05
module load cuda/11.8.0

# Activate environment
source ~/venvs/my_ml_project/bin/activate

# Verify GPU
nvidia-smi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Data paths
SCRATCH=/fs/scratch/PAS1234/$USER/my_ml_project
TRAIN_DATA=$SCRATCH/datasets/processed/train
VAL_DATA=$SCRATCH/datasets/processed/val

# Run training
cd ~/projects/my_ml_project
python src/train.py \
    --train-data $TRAIN_DATA \
    --val-data $VAL_DATA \
    --experiment-name baseline \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --num-workers 4

echo "Job completed at: $(date)"
```

### 6. Hyperparameter Search

`scripts/hyperparameter_search.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hp_search
#SBATCH --array=1-27              # 3x3x3 grid
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/hp_search_%A_%a.out

# Define hyperparameter grid
lrs=(0.001 0.01 0.1)
batch_sizes=(16 32 64)
dropouts=(0.1 0.3 0.5)

# Map array task ID to hyperparameters
idx=$SLURM_ARRAY_TASK_ID
lr_idx=$((idx % 3))
bs_idx=$(((idx / 3) % 3))
dropout_idx=$(((idx / 9) % 3))

lr=${lrs[$lr_idx]}
bs=${batch_sizes[$bs_idx]}
dropout=${dropouts[$dropout_idx]}

# Load environment
module load python/3.9-2022.05 cuda/11.8.0
source ~/venvs/my_ml_project/bin/activate

# Run training with these hyperparameters
SCRATCH=/fs/scratch/PAS1234/$USER/my_ml_project
python src/train.py \
    --train-data $SCRATCH/datasets/processed/train \
    --val-data $SCRATCH/datasets/processed/val \
    --experiment-name "hp_lr${lr}_bs${bs}_dropout${dropout}" \
    --lr $lr \
    --batch-size $bs \
    --dropout $dropout \
    --epochs 50
```

### 7. Experiment Tracking

#### Using TensorBoard

```bash
# On OSC
tensorboard --logdir=runs --port=6006 --bind_all

# On local machine (with port forwarding)
ssh -L 6006:localhost:6006 pitzer

# Open browser: http://localhost:6006
```

#### Using Weights & Biases (Optional)

```python
import wandb

# In train.py
wandb.init(
    project="my-ml-project",
    config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
)

# Log metrics
wandb.log({"train_loss": train_loss, "train_acc": train_acc})

# Log model
wandb.save('checkpoints/best_model.pth')
```

## Best Practices

### 1. Version Control
- Commit code regularly
- Use meaningful commit messages
- Tag important experiments
- Don't commit large files

### 2. Reproducibility
- Set random seeds
- Document environment (requirements.txt)
- Save hyperparameters with checkpoints
- Use configuration files

### 3. Resource Management
- Test with small dataset first
- Use debug queue for testing
- Request appropriate resources
- Monitor GPU usage

### 4. Data Management
- Use scratch space for large data
- Don't store data in home directory
- Clean up old checkpoints
- Archive completed experiments

### 5. Code Organization
- Modular code (separate data, models, training)
- Use configuration files
- Write unit tests
- Document your code

## Next Steps

- Learn about [GPU Computing](gpu-computing.md)
- Review [PyTorch Setup](pytorch-setup.md)
- Check [Best Practices](../working-on-osc/osc-best-practices.md)
- Read [Job Submission Guide](../working-on-osc/osc-job-submission.md)

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TensorBoard Tutorial](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)
- [Troubleshooting Guide](../resources/troubleshooting.md)
