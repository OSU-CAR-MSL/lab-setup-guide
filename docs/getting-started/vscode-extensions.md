# VS Code Extensions

Extensions enhance VS Code's functionality. This guide covers required extensions for lab work and recommended extensions for productivity.

## Installing Extensions

### Method 1: Extension Marketplace (GUI)
1. Click the Extensions icon in the sidebar (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
2. Search for the extension name
3. Click "Install"

### Method 2: Command Line
```bash
code --install-extension <extension-id>
```

### Method 3: Extensions View
Press `Ctrl+Shift+P` / `Cmd+Shift+P`, type "Install Extensions", and search for the extension.

## Required Extensions

These extensions are essential for lab work:

### 1. Remote - SSH
- **ID**: `ms-vscode-remote.remote-ssh`
- **Purpose**: Connect to OSC and other remote servers
- **Install**: 
  ```bash
  code --install-extension ms-vscode-remote.remote-ssh
  ```
- **Why Required**: Essential for working on OSC clusters
- **Documentation**: [Remote - SSH Guide](../osc-basics/osc-remote-development.md)

### 2. Remote - SSH: Editing Configuration Files
- **ID**: `ms-vscode-remote.remote-ssh-edit`
- **Purpose**: Edit SSH configuration files easily
- **Install**: 
  ```bash
  code --install-extension ms-vscode-remote.remote-ssh-edit
  ```

### 3. Python
- **ID**: `ms-python.python`
- **Purpose**: Python language support, debugging, linting
- **Install**: 
  ```bash
  code --install-extension ms-python.python
  ```
- **Why Required**: Most lab projects use Python

### 4. Pylance
- **ID**: `ms-python.vscode-pylance`
- **Purpose**: Fast, feature-rich Python language server
- **Install**: 
  ```bash
  code --install-extension ms-python.vscode-pylance
  ```
- **Features**: Type checking, auto-imports, better IntelliSense

### 5. Jupyter
- **ID**: `ms-toolsai.jupyter`
- **Purpose**: Run and edit Jupyter notebooks in VS Code
- **Install**: 
  ```bash
  code --install-extension ms-toolsai.jupyter
  ```
- **Why Required**: Many ML workflows use Jupyter notebooks

## Highly Recommended Extensions

### General Development

#### 1. GitLens
- **ID**: `eamodio.gitlens`
- **Purpose**: Supercharge Git capabilities
- **Features**: Blame annotations, commit history, repository insights
- **Install**: 
  ```bash
  code --install-extension eamodio.gitlens
  ```

#### 2. Git Graph
- **ID**: `mhutchie.git-graph`
- **Purpose**: View Git repository graph
- **Features**: Visual commit history, branch visualization
- **Install**: 
  ```bash
  code --install-extension mhutchie.git-graph
  ```

#### 3. Path Intellisense
- **ID**: `christian-kohler.path-intellisense`
- **Purpose**: Auto-complete file paths
- **Install**: 
  ```bash
  code --install-extension christian-kohler.path-intellisense
  ```

#### 4. Todo Tree
- **ID**: `gruntfuggly.todo-tree`
- **Purpose**: Highlight and organize TODO comments
- **Install**: 
  ```bash
  code --install-extension gruntfuggly.todo-tree
  ```

### Python Development

#### 5. autoDocstring
- **ID**: `njpwerner.autodocstring`
- **Purpose**: Generate Python docstrings automatically
- **Install**: 
  ```bash
  code --install-extension njpwerner.autodocstring
  ```

#### 6. Python Indent
- **ID**: `kevinrose.vsc-python-indent`
- **Purpose**: Correct Python indentation
- **Install**: 
  ```bash
  code --install-extension kevinrose.vsc-python-indent
  ```

#### 7. isort
- **ID**: `ms-python.isort`
- **Purpose**: Sort Python imports automatically
- **Install**: 
  ```bash
  code --install-extension ms-python.isort
  ```

#### 8. Black Formatter
- **ID**: `ms-python.black-formatter`
- **Purpose**: Format Python code using Black
- **Install**: 
  ```bash
  code --install-extension ms-python.black-formatter
  ```

### Machine Learning

#### 9. PyTorch Snippets
- **ID**: `SBSnippets.pytorch-snippets`
- **Purpose**: PyTorch code snippets
- **Install**: 
  ```bash
  code --install-extension SBSnippets.pytorch-snippets
  ```

#### 10. TensorBoard
- **ID**: `ms-toolsai.vscode-tensorboard`
- **Purpose**: View TensorBoard logs in VS Code
- **Install**: 
  ```bash
  code --install-extension ms-toolsai.vscode-tensorboard
  ```

### Productivity

#### 11. Markdown All in One
- **ID**: `yzhang.markdown-all-in-one`
- **Purpose**: Enhanced Markdown editing
- **Features**: Shortcuts, table of contents, preview
- **Install**: 
  ```bash
  code --install-extension yzhang.markdown-all-in-one
  ```

#### 12. Better Comments
- **ID**: `aaron-bond.better-comments`
- **Purpose**: Color-code comments (TODO, FIXME, etc.)
- **Install**: 
  ```bash
  code --install-extension aaron-bond.better-comments
  ```

#### 13. Error Lens
- **ID**: `usernamehw.errorlens`
- **Purpose**: Highlight errors inline
- **Install**: 
  ```bash
  code --install-extension usernamehw.errorlens
  ```

#### 14. Bracket Pair Colorizer 2 (or built-in)
- **Note**: VS Code now has built-in bracket pair colorization
- **Enable**: Search for "Bracket Pair Colorization" in settings and enable it

## Extension Configuration

### Python Extension Settings

Add to your `settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.languageServer": "Pylance",
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### Remote SSH Settings

```json
{
  "remote.SSH.remotePlatform": {
    "pitzer.osc.edu": "linux",
    "owens.osc.edu": "linux"
  },
  "remote.SSH.showLoginTerminal": true,
  "remote.SSH.useLocalServer": false
}
```

## Batch Installation

Install all required extensions at once:

```bash
# Required extensions
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-ssh-edit
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter

# Recommended extensions
code --install-extension eamodio.gitlens
code --install-extension mhutchie.git-graph
code --install-extension christian-kohler.path-intellisense
code --install-extension njpwerner.autodocstring
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort
code --install-extension yzhang.markdown-all-in-one
code --install-extension usernamehw.errorlens
```

## Managing Extensions

### Disable Extensions for Specific Workspaces
- Right-click an extension in the Extensions view
- Select "Disable (Workspace)" to disable it only for the current project

### Extension Recommendations for Projects
Create `.vscode/extensions.json` in your project:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter"
  ]
}
```

## Extension Syncing

Enable Settings Sync (see [VS Code Setup](vscode-setup.md#settings-sync)) to sync extensions across machines.

## Next Steps

- Configure [Remote Development on OSC](../osc-basics/osc-remote-development.md)
- Set up [SSH Connection to OSC](../osc-basics/osc-ssh-connection.md)
- Learn about [PyTorch Setup](../ml-workflows/pytorch-setup.md)

## Troubleshooting

### Extension won't activate on remote
- Make sure the extension supports remote development
- Try reloading the VS Code window: `Ctrl+Shift+P` → "Developer: Reload Window"
- Check the extension's output log for errors

### Extensions causing performance issues
- Disable extensions you don't actively use
- Use "Disable (Workspace)" for extensions not needed in specific projects
- Check extension ratings and reviews for known issues

For more help, see the [Troubleshooting Guide](../resources/troubleshooting.md).
