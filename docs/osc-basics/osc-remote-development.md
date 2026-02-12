# Remote Development on OSC

Visual Studio Code's Remote-SSH extension allows you to develop on OSC as if you were working locally. This guide shows you how to set it up and use it effectively.

## Prerequisites

- [VS Code installed](../getting-started/vscode-setup.md)
- [Remote-SSH extension installed](../getting-started/vscode-extensions.md#1-remote-ssh)
- [OSC account and SSH configured](osc-ssh-connection.md)

## Initial Setup

### 1. Install Remote-SSH Extension

If not already installed:

```bash
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-ssh-edit
```

Or install via VS Code Extensions marketplace (`Ctrl+Shift+X`).

### 2. Configure SSH Connection

Ensure your `~/.ssh/config` file is set up (see [SSH Connection Guide](osc-ssh-connection.md)):

```ssh-config
Host pitzer
    HostName pitzer.osc.edu
    User your.osuusername
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Connecting to OSC

### Method 1: Command Palette

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
2. Type "Remote-SSH: Connect to Host"
3. Select `pitzer` (or `owens`) from your SSH config
4. A new VS Code window opens connected to OSC

### Method 2: Remote Explorer

1. Click the Remote Explorer icon in the sidebar (><)
2. Select "SSH Targets" from dropdown
3. Click the connect icon next to `pitzer`

### Method 3: Command Line

```bash
code --remote ssh-remote+pitzer /path/to/directory
```

### First Connection

On your first connection:
1. VS Code downloads the VS Code Server to OSC
2. This takes 1-2 minutes
3. A progress notification appears in the bottom-right
4. Future connections are much faster

## Working Remotely

### Opening Folders and Files

1. **Open Folder**: `File` → `Open Folder` or `Ctrl+K Ctrl+O`
   - Navigate to your home directory or project folder
   - Click "OK"

2. **Open File**: `File` → `Open File` or `Ctrl+O`

3. **Quick Open**: `Ctrl+P` to quickly open files by name

### Integrated Terminal

The integrated terminal runs on OSC:

1. Open terminal: `` Ctrl+` `` (backtick)
2. You're now in a bash shell on OSC
3. Run any command as if you SSH'd in directly

Example workflow:
```bash
# Load modules
module load python/3.9

# Check resources
squeue -u $USER

# Navigate your project
cd ~/projects/my-research

# Run Python script
python train_model.py
```

### File Browser

The VS Code file browser shows files on OSC:
- Browse your home directory
- Search files with `Ctrl+P`
- Use the file browser to navigate
- Right-click for file operations

### Editing Files

Edit files on OSC exactly like local files:
- IntelliSense works
- Linting works
- Auto-complete works
- Format on save works

Everything runs on the remote server!

## Installing Extensions Remotely

Extensions you use locally should also be installed on the remote server. Open the Extensions view (`Ctrl+Shift+X`) and click "Install in SSH: pitzer" for each extension you need. For recommended extensions, see [VS Code Extensions](../getting-started/vscode-extensions.md).

## File Transfer

VS Code supports drag-and-drop uploads and right-click downloads for small files. For bulk transfers, see the [File Transfer Guide](osc-file-transfer.md).

## Port Forwarding

Access web services running on OSC (Jupyter, TensorBoard, etc.):

### Automatic Port Forwarding

VS Code automatically detects and forwards ports:

1. Start a service on OSC (e.g., Jupyter on port 8888)
2. VS Code detects it and shows a notification
3. Click "Open in Browser"

### Manual Port Forwarding

1. Click "PORTS" tab in bottom panel (next to Terminal)
2. Click "Forward a Port"
3. Enter port number (e.g., 8888)
4. Access at `http://localhost:8888`

### Example: Jupyter Notebook

```bash
# In VS Code terminal on OSC
module load python/3.9
jupyter notebook --no-browser --port=8888
```

VS Code forwards port 8888 automatically. Click the link or open `localhost:8888` in your browser.

## Running Code

### Python Scripts

#### Method 1: Terminal
```bash
python script.py
```

#### Method 2: Run Button
- Open a Python file
- Click the "Run" button (▶️) in the top-right
- Or press `Ctrl+Alt+N` (if Code Runner extension installed)

### Jupyter Notebooks

1. Open `.ipynb` file in VS Code
2. Select Python kernel (from OSC)
3. Run cells with `Shift+Enter`

All execution happens on OSC!

### Debugging

Set breakpoints and debug remotely:

1. Set breakpoints by clicking left of line numbers
2. Press `F5` to start debugging
3. Use Debug Console to inspect variables
4. All debugging runs on OSC

## Best Practices

### 1. Use Login Nodes Appropriately

**Login nodes** (where Remote-SSH connects) are for:
- Editing code
- Light testing
- Job submission

**DO NOT** run intensive computations on login nodes.

### 2. Use Compute Nodes for Heavy Work

For intensive tasks:

```bash
# Request interactive session
srun -p debug -c 4 --time=01:00:00 --pty bash

# Or use batch jobs
sbatch job_script.sh
```

### 3. Save Regularly

Although Remote-SSH is reliable:
- Enable auto-save: `"files.autoSave": "afterDelay"`
- Commit to Git frequently
- Keep backups of important work

### 4. Manage Extensions

- Only install necessary extensions remotely
- Some extensions can slow down remote connections
- Disable unused extensions

### 5. Handle Disconnections Gracefully

If connection drops:
- VS Code tries to reconnect automatically
- Unsaved changes are usually preserved
- Terminal sessions may be lost (use `tmux` or `screen` for persistence)

## Advanced Tips

### Using tmux for Persistent Sessions

```bash
# Start tmux session
tmux new -s work

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t work
```

This keeps your session alive even if VS Code disconnects.

??? note "Workspace Settings for Remote"

    Create `.vscode/settings.json` in your project:

    ```json
    {
      "python.defaultInterpreterPath": "/usr/local/python/3.9/bin/python3",
      "python.terminal.activateEnvironment": true,
      "terminal.integrated.env.linux": {
        "PATH": "/usr/local/python/3.9/bin:${env:PATH}"
      }
    }
    ```

### SSH Config for Multiple Login Nodes

```ssh-config
# Primary login
Host pitzer
    HostName pitzer.osc.edu
    User your.osuusername

# Specific login node (if needed)
Host pitzer01
    HostName pitzer-login01.osc.edu
    User your.osuusername
```

## Troubleshooting

### Connection Hangs

**Problem**: "Setting up SSH Host..." hangs indefinitely

**Solutions**:
1. Check OSC system status
2. Verify SSH connection works: `ssh pitzer`
3. Delete remote VS Code server:
   ```bash
   ssh pitzer
   rm -rf ~/.vscode-server
   ```
4. Reconnect in VS Code

### Extension Installation Fails

**Problem**: Extensions won't install on remote

**Solutions**:
1. Check disk quota: `quota -s` on OSC
2. Clear extension cache:
   ```bash
   rm -rf ~/.vscode-server/extensions
   ```
3. Reinstall extensions one by one

### Python Interpreter Not Found

**Problem**: VS Code can't find Python

**Solutions**:
1. Load Python module in terminal:
   ```bash
   module load python/3.9
   which python3
   ```
2. Set interpreter in VS Code:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Enter the path from `which python3`

### Terminal Not Loading

**Problem**: Integrated terminal fails to start

**Solutions**:
1. Check your shell configuration (~/.bashrc)
2. Simplify `.bashrc` (remove complex startup scripts)
3. Try different shell: `"terminal.integrated.shell.linux": "/bin/bash"`

### Port Forwarding Not Working

**Problem**: Can't access forwarded ports

**Solutions**:
1. Verify service is running: `netstat -tuln | grep <port>`
2. Manually add port in VS Code Ports panel
3. Check firewall settings on OSC

??? tip "Performance Tuning"

    **Reduce remote extension count** — install only essentials remotely:

    - ms-python.python
    - ms-python.vscode-pylance
    - ms-toolsai.jupyter

    **Exclude large directories from the file watcher** in `.vscode/settings.json`:

    ```json
    {
      "files.watcherExclude": {
        "**/node_modules": true,
        "**/.git": true,
        "**/data": true,
        "**/checkpoints": true
      }
    }
    ```

## Next Steps

- Learn about [OSC Best Practices](../working-on-osc/osc-best-practices.md)
- Set up [PyTorch on OSC](../ml-workflows/pytorch-setup.md)
- Explore [Job Submission Guide](../working-on-osc/osc-job-submission.md)
- Read [ML Workflow Guide](../ml-workflows/ml-workflow.md)

## Resources

- [VS Code Remote-SSH Documentation](https://code.visualstudio.com/docs/remote/ssh)
- [OSC Documentation](https://www.osc.edu/resources/technical_support/supercomputers)
- [Troubleshooting Guide](../resources/troubleshooting.md)
