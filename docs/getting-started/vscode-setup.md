# VS Code Setup Guide

Visual Studio Code (VS Code) is a free, powerful, and extensible code editor that we use in the lab for development work. This guide will walk you through installing and configuring VS Code.

## Installation

### Windows

1. **Download VS Code**
   - Visit [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Click the "Download for Windows" button
   - Run the downloaded installer (`VSCodeUserSetup-{version}.exe`)

2. **Installation Options**
   - ✅ Check "Add to PATH" (important for command-line usage)
   - ✅ Check "Create a desktop icon" (optional but convenient)
   - ✅ Check "Register Code as an editor for supported file types"
   - ✅ Check "Add 'Open with Code' action to context menu"

3. **Complete Installation**
   - Click "Install" and wait for the installation to complete
   - Launch VS Code

### macOS

1. **Download VS Code**
   - Visit [https://code.visualstudio.com/](https://code.visualstudio.com/)
   - Click "Download for Mac"
   - Open the downloaded `.zip` file

2. **Install**
   - Drag `Visual Studio Code.app` to the Applications folder
   - Launch VS Code from Applications or Spotlight

3. **Add to PATH** (for command-line usage)
   - Open VS Code
   - Press `Cmd+Shift+P` to open the Command Palette
   - Type "shell command" and select "Shell Command: Install 'code' command in PATH"

### Linux (Ubuntu/Debian)

1. **Using APT Repository** (Recommended)
   ```bash
   # Update package index and install dependencies
   sudo apt-get update
   sudo apt-get install wget gpg apt-transport-https
   
   # Download and install Microsoft GPG key
   wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
   sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
   
   # Add VS Code repository
   sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
   
   # Install VS Code
   sudo apt-get update
   sudo apt-get install code
   ```

2. **Using Snap** (Alternative)
   ```bash
   sudo snap install --classic code
   ```

3. **Launch VS Code**
   ```bash
   code
   ```

## Initial Configuration

### Basic Settings

1. **Open Settings**
   - Go to `File` → `Preferences` → `Settings` (or `Code` → `Preferences` → `Settings` on macOS)
   - Or press `Ctrl+,` (Windows/Linux) or `Cmd+,` (macOS)

2. **Recommended Settings**
   
   Search for and configure these settings:
   
   - **Auto Save**: Search for "Auto Save" and set to `afterDelay`
   - **Tab Size**: Search for "Tab Size" and set to `4` (or `2` for web development)
   - **Format On Save**: Search for "Format On Save" and enable it
   - **Trim Trailing Whitespace**: Search for "Trim Trailing Whitespace" and enable it
   - **Files: Insert Final Newline**: Enable to ensure files end with a newline

### Font and Appearance

1. **Font Settings**
   - Search for "Font Family" and set your preferred monospace font
   - Popular options: `'Fira Code', 'Consolas', 'Monaco', 'Courier New'`
   - For ligatures with Fira Code, enable "Font Ligatures"

2. **Theme**
   - Press `Ctrl+K Ctrl+T` (Windows/Linux) or `Cmd+K Cmd+T` (macOS)
   - Choose a color theme (popular: Dark+, Monokai, Dracula)

### Terminal Configuration

1. **Integrated Terminal**
   - Open terminal: `` Ctrl+` `` (backtick)
   - Configure default shell in Settings under "Terminal > Integrated > Default Profile"

2. **Windows Users**: Consider setting up Windows Terminal or Git Bash for a better terminal experience

## Keyboard Shortcuts

Learn these essential keyboard shortcuts:

### Universal
- `Ctrl+P` / `Cmd+P`: Quick file open
- `Ctrl+Shift+P` / `Cmd+Shift+P`: Command palette
- `Ctrl+B` / `Cmd+B`: Toggle sidebar
- `` Ctrl+` `` / `` Cmd+` ``: Toggle terminal
- `Ctrl+/` / `Cmd+/`: Toggle line comment

### Editing
- `Alt+Up/Down` / `Opt+Up/Down`: Move line up/down
- `Shift+Alt+Up/Down` / `Shift+Opt+Up/Down`: Copy line up/down
- `Ctrl+D` / `Cmd+D`: Add selection to next find match
- `Ctrl+Shift+L` / `Cmd+Shift+L`: Select all occurrences of current selection

### Navigation
- `Ctrl+Tab`: Switch between open files
- `F12`: Go to definition
- `Alt+Left/Right` / `Cmd+Left/Right`: Navigate back/forward

## Settings Sync

Enable Settings Sync to keep your configuration across devices:

1. Click the gear icon (⚙️) in the bottom-left corner
2. Select "Turn on Settings Sync"
3. Sign in with GitHub or Microsoft account
4. Choose what to sync (settings, extensions, keybindings, etc.)

## Next Steps

- Install [Required Extensions](vscode-extensions.md)
- Set up [Remote Development on OSC](../osc-basics/osc-remote-development.md)
- Configure your [OSC SSH Connection](../osc-basics/osc-ssh-connection.md)

## Troubleshooting

### VS Code won't launch
- **Windows**: Try running the installer as administrator
- **macOS**: Check that the app is in the Applications folder
- **Linux**: Verify the installation with `code --version`

### Extensions not installing
- Check your internet connection
- Try disabling your firewall temporarily
- Reload VS Code: Press `Ctrl+Shift+P` and run "Developer: Reload Window"

### More Issues?
See the [Troubleshooting Guide](../resources/troubleshooting.md) or visit [VS Code Documentation](https://code.visualstudio.com/docs)
