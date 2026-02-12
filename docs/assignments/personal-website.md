---
title: "Build Your Personal Academic Website"
description: "Set up VS Code, Git, and Quarto, then build and deploy a personal academic website on GitHub Pages."
---

# Build Your Personal Academic Website with Quarto + GitHub Pages

| | |
|---|---|
| **Assigned by** | Robert Frenken |
| **Due** | 1 week from assignment date |
| **Estimated time** | 3--4 hours (plus troubleshooting) |
| **Difficulty** | Beginner |
| **Prerequisites** | A computer (Windows or Mac), internet access |
| **Last updated** | February 2026 |

---

## Purpose

This assignment will get you set up with the core tools we use in the lab while producing something immediately useful: your own professional academic website. Along the way, you'll practice the full Git workflow (clone, edit, commit, push, pull) that we use daily for research collaboration.

By the end, you will have:

- [x] VS Code installed and configured
- [x] Git installed and working with GitHub via SSH
- [x] A live personal website at `https://YOURUSERNAME.github.io`
- [x] Hands-on experience with the Git commit/push/pull cycle
- [x] Familiarity with Quarto (our lab's publishing tool for papers, slides, and websites)

---

## Which Terminal Should I Use?

Throughout this assignment, you'll run commands in a terminal. Here's which one to use:

| OS | Terminal to Use | How to Open |
|----|----------------|-------------|
| **Windows** | **Git Bash** (installed with Git) | Search "Git Bash" in Start menu, or right-click a folder → "Open Git Bash here" |
| **Mac** | **Terminal** | Spotlight (++cmd+space++) → type "Terminal" |

!!! warning "Windows users"
    Use **Git Bash** for all commands in this assignment, **not** PowerShell or Command Prompt. Git Bash gives you the same Linux-style commands (`cd`, `ls`, `ssh-keygen`, etc.) that you'll see in tutorials online. You can also use the VS Code integrated terminal — just make sure it's set to Git Bash (see Part 0.2).

---

## Part 0: Environment Setup

Complete these steps before starting the main assignment. If you get stuck, ask in the lab Slack/Teams channel — setup issues are normal and everyone hits them.

### 0.1 Install Git

=== "Windows"

    1. Download from [git-scm.com/downloads/win](https://git-scm.com/downloads/win)
    2. Run the installer. Use the defaults for most screens, but pay attention to these:
        - **"Adjusting your PATH environment"** → select **"Git from the command line and also from 3rd-party software"**
        - **"Choosing the default editor"** → select **"Use Visual Studio Code as Git's default editor"** (if you've already installed VS Code)
        - **"Adjusting the name of the initial branch"** → select **"Override the default branch name"** and type `main`
    3. After install, open **Git Bash** (search for it in the Start menu) and verify:
        ```bash
        git --version
        ```

=== "Mac"

    1. Open **Terminal** (Spotlight → "Terminal")
    2. Run:
        ```bash
        git --version
        ```
    3. If Git is not installed, macOS will prompt you to install **Xcode Command Line Tools**. Click "Install" and wait (this can take a few minutes).
    4. After installation finishes, verify:
        ```bash
        git --version
        ```

    !!! tip "Alternative"
        If you use [Homebrew](https://brew.sh/), you can also install via `brew install git`.

### 0.2 Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/) — pick the installer for your OS.

**Install these extensions** (click the Extensions icon in the sidebar, or press ++ctrl+shift+x++ on Windows / ++cmd+shift+x++ on Mac):

1. **Quarto** — search "quarto" by Quarto
2. **GitLens** — search "gitlens" (optional but very helpful for understanding Git history)

??? note "Windows: Set Git Bash as your default terminal in VS Code"
    This is important so that all the commands in this assignment work correctly inside VS Code.

    1. Open VS Code
    2. Press ++ctrl+shift+p++ to open the Command Palette
    3. Type "Terminal: Select Default Profile" and select it
    4. Choose **Git Bash** from the list
    5. Open a new terminal (++ctrl+grave++) — it should now say "bash" in the top-right of the terminal panel

### 0.3 Install Quarto CLI

Download the installer for your OS from [quarto.org/docs/get-started](https://quarto.org/docs/get-started/)

=== "Windows"

    Download the `.msi` installer and run it.

=== "Mac"

    Download the `.pkg` installer and run it. Alternatively: `brew install --cask quarto`

Verify (open a **new** terminal window after installing):

```bash
quarto --version
```

!!! tip "command not found?"
    Close and reopen your terminal (or restart VS Code). The installer adds Quarto to your PATH, but existing terminal sessions don't pick it up automatically.

### 0.4 Create a GitHub Account

If you don't have one: [github.com/signup](https://github.com/signup)

**Use your university email** (or add it as a secondary email) — this qualifies you for [GitHub Education](https://education.github.com/) benefits.

### 0.5 Set Up SSH Keys for GitHub

SSH keys let you push/pull without entering your password every time. The setup differs slightly by OS.

=== "Windows (Git Bash)"

    **Step 1 — Generate a key:**

    ```bash
    ssh-keygen -t ed25519 -C "your.email@osu.edu"
    ```
    Press ++enter++ for all prompts (default file location, no passphrase is fine for now).

    **Step 2 — Start the SSH agent:**

    ```bash
    eval "$(ssh-agent -s)"
    ```
    You should see something like `Agent pid 12345`.

    **Step 3 — Add your key to the agent:**

    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```

    **Step 4 — Copy the public key to your clipboard:**

    ```bash
    clip < ~/.ssh/id_ed25519.pub
    ```

    **Step 5 — Add the key to GitHub:**

    1. Go to [github.com/settings/keys](https://github.com/settings/keys)
    2. Click **"New SSH key"**
    3. Title: something like "My Laptop"
    4. Key type: **Authentication Key**
    5. Paste the key (it should start with `ssh-ed25519`)
    6. Click **"Add SSH key"**

    **Step 6 — Test the connection:**

    ```bash
    ssh -T git@github.com
    ```
    Type `yes` if asked about the fingerprint. You should see:
    ```
    Hi YOURUSERNAME! You've successfully authenticated, but GitHub does not provide shell access.
    ```

=== "Mac (Terminal)"

    **Step 1 — Generate a key:**

    ```bash
    ssh-keygen -t ed25519 -C "your.email@osu.edu"
    ```
    Press ++enter++ for all prompts (default file location, no passphrase is fine for now).

    **Step 2 — Start the SSH agent:**

    ```bash
    eval "$(ssh-agent -s)"
    ```

    **Step 3 — Create/update the SSH config file:**

    This tells macOS to automatically load your key and store the passphrase in Keychain:
    ```bash
    touch ~/.ssh/config
    echo 'Host github.com
      AddKeysToAgent yes
      UseKeychain yes
      IdentityFile ~/.ssh/id_ed25519' >> ~/.ssh/config
    ```

    **Step 4 — Add your key to the agent:**

    ```bash
    ssh-add --apple-use-keychain ~/.ssh/id_ed25519
    ```

    !!! note
        If you get an error about `--apple-use-keychain`, try `ssh-add -K ~/.ssh/id_ed25519` instead (older macOS versions use `-K`).

    **Step 5 — Copy the public key to your clipboard:**

    ```bash
    pbcopy < ~/.ssh/id_ed25519.pub
    ```

    **Step 6 — Add the key to GitHub:**

    1. Go to [github.com/settings/keys](https://github.com/settings/keys)
    2. Click **"New SSH key"**
    3. Title: something like "My MacBook"
    4. Key type: **Authentication Key**
    5. Paste the key (it should start with `ssh-ed25519`)
    6. Click **"Add SSH key"**

    **Step 7 — Test the connection:**

    ```bash
    ssh -T git@github.com
    ```
    Type `yes` if asked about the fingerprint. You should see:
    ```
    Hi YOURUSERNAME! You've successfully authenticated, but GitHub does not provide shell access.
    ```

### 0.6 Configure Git Identity

Run these in your terminal (Git Bash on Windows, Terminal on Mac):

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@osu.edu"
```

---

## Part 1: Create Your Website Repository

### 1.1 Use the Template

1. Go to: **[github.com/OSU-CAR-MSL/quarto-academic-site-template](https://github.com/OSU-CAR-MSL/quarto-academic-site-template)**

    !!! tip "Want to understand the template?"
        See the [template documentation](../contributing/github-pages-setup.md#lab-academic-website-template) for a full breakdown of every file and how the auto-listing system works.
2. Click the green **"Use this template"** button → **"Create a new repository"**
3. **Repository name:** `YOURUSERNAME.github.io` (replace with your actual GitHub username — case-sensitive!)
4. Set to **Public**
5. Click **"Create repository"**

### 1.2 Clone to Your Computer

Open your terminal (Git Bash on Windows, Terminal on Mac):

```bash
# Navigate to where you want the project
cd ~/Documents

# Clone (replace YOURUSERNAME with your GitHub username)
git clone git@github.com:YOURUSERNAME/YOURUSERNAME.github.io.git

# Enter the project folder
cd YOURUSERNAME.github.io

# Open in VS Code
code .
```

!!! note "If `code .` doesn't work"

    === "Windows"

        Open VS Code manually, then File → Open Folder → navigate to the cloned folder.

    === "Mac"

        Open VS Code, press ++cmd+shift+p++, type "Shell Command: Install 'code' command in PATH", run it, then try again.

### 1.3 Enable GitHub Pages

1. Go to your repo on GitHub: `github.com/YOURUSERNAME/YOURUSERNAME.github.io`
2. Click **Settings** → **Pages** (in the left sidebar)
3. Under "Build and deployment" → Source: select **GitHub Actions**
4. Go back to **Settings** → **Actions** → **General** → scroll to "Workflow permissions"
5. Select **"Read and write permissions"** → click **Save**

---

## Part 2: Personalize Your Site (Git Practice)

Each task below practices a specific Git operation. **Do them in order.**

!!! info "Where to run commands"
    All `git` and `quarto` commands should be run in the **VS Code integrated terminal** (++ctrl+grave++ on Windows, ++cmd+grave++ on Mac). Make sure you're in your project folder.

### Task 1: Edit Your Home Page

**Git skills practiced:** `git add`, `git commit`, `git push`

1. In VS Code, open `_quarto.yml`
    - Replace `"Your Name"` with your name
    - Replace the `site-url` with `https://YOURUSERNAME.github.io`
    - Update the GitHub and email links in the `navbar` section
    - Update the footer with your name

2. Open `index.qmd`
    - Replace the placeholder bio with your actual information
    - Update your research interests, education, etc.
    - Update the links in the YAML header (GitHub, email, LinkedIn)

3. Add a profile photo
    - Put a photo in the `images/` folder named `profile.jpg`
    - Square crop is best, at least 400x400 pixels
    - You can drag and drop the image into the `images/` folder in VS Code's file explorer sidebar

4. **Preview locally** (in the VS Code terminal):
    ```bash
    quarto preview
    ```
    This opens your site in a browser. Check that it looks right. Press ++ctrl+c++ in the terminal to stop the preview.

5. **Commit and push:**
    ```bash
    # See what files changed
    git status

    # Stage all changes
    git add .

    # Commit with a descriptive message
    git commit -m "Add personal info and profile photo"

    # Push to GitHub
    git push
    ```

6. **Wait 1--2 minutes**, then check `https://YOURUSERNAME.github.io` — your site should be live!

### Task 2: Edit via GitHub Web UI, Then Pull Locally

**Git skills practiced:** editing on GitHub, `git pull`, understanding remote vs local

1. Go to your repo on GitHub **in your browser**
2. Click on `cv.qmd` → click the **pencil icon** (Edit this file)
3. Fill in your CV details (education, skills, etc.)
4. Scroll down → write a commit message like "Update CV from GitHub" → click **"Commit changes"**
5. Now your remote (GitHub) is **ahead** of your local copy. Back in your VS Code terminal:
    ```bash
    # Check the status — Git will tell you your local is behind
    git fetch
    git status

    # Pull the changes
    git pull
    ```
6. Open `cv.qmd` in VS Code — you should see your edits from GitHub.

!!! tip "Why this matters"
    In research collaboration, your teammates will push changes you don't have locally. `git pull` is how you stay in sync.

### Task 3: Add a Blog Post

**Git skills practiced:** creating new files, multi-file commits, YAML frontmatter

1. In VS Code, right-click the `posts/` folder → "New File" → name it `my-first-post.qmd`
2. Add this YAML header at the top (customize it):
    ```yaml
    ---
    title: "Getting Started in the CAR Lab"
    description: "My first week in the Mobility Systems Lab."
    date: "2025-02-12"
    categories: [lab, onboarding]
    ---
    ```
3. Write 2--3 paragraphs below the YAML. Content ideas:
    - What you're excited to work on
    - What you learned setting up this website
    - A brief intro of yourself

4. Preview: `quarto preview`

5. Commit and push:
    ```bash
    git add posts/my-first-post.qmd
    git commit -m "Add first blog post"
    git push
    ```

### Task 4: Add a Research Project Page

**Git skills practiced:** working with Quarto listings, understanding how auto-generated pages work

1. Create a new file: `research/my-project.qmd`
2. Add a YAML header:
    ```yaml
    ---
    title: "Your Project Title"
    description: "Brief description of what you're working on."
    date: "2025-02-12"
    categories: [your-topic]
    ---
    ```
3. Write a short summary of the research project you'll be working on (even if you're just getting started, describe what you understand so far)

4. Commit and push:
    ```bash
    git add research/my-project.qmd
    git commit -m "Add research project page"
    git push
    ```

### Task 5 (Stretch): Resolve a Merge Conflict

**Git skills practiced:** merge conflicts — the single most important Git skill to learn early

!!! info "Don't worry"
    This task intentionally creates a problem so you can practice fixing it. You can't break anything permanently.

1. Open `index.qmd` **in VS Code**. Change one of your research interests to something different (e.g., change "Interest 1" to "Robotics"). **Do NOT commit yet.**

2. Go to GitHub **in your browser**. Click on `index.qmd` → pencil icon → edit the *same line* with a *different* value (e.g., change "Interest 1" to "Computer Vision"). Commit it on GitHub.

3. Now back in VS Code terminal, try to commit and push:
    ```bash
    git add index.qmd
    git commit -m "Update research interest locally"
    git push   # This will FAIL — that's expected!
    ```

4. Git will tell you to pull first:
    ```bash
    git pull
    ```

5. Git will report a **merge conflict**. Open `index.qmd` in VS Code — you'll see something like:
    ```
    <<<<<<< HEAD
    - Robotics
    =======
    - Computer Vision
    >>>>>>> origin/main
    ```

    VS Code will highlight the conflict and show buttons: "Accept Current Change", "Accept Incoming Change", "Accept Both Changes". Pick whichever you want (or manually edit to combine them). Delete the `<<<<<<<`, `=======`, `>>>>>>>` markers if they remain.

6. Save the file, then commit the resolution:
    ```bash
    git add index.qmd
    git commit -m "Resolve merge conflict in research interests"
    git push
    ```

!!! success "Congratulations"
    You've handled your first merge conflict! This happens regularly in collaborative work, and now you know how to fix it.

---

## Deliverables

Send Robert the following when complete:

1. **Your live website URL:** `https://YOURUSERNAME.github.io`
2. **Your GitHub repo URL:** `https://github.com/YOURUSERNAME/YOURUSERNAME.github.io`

Your site should have:

- [ ] Your name and photo on the home page
- [ ] At least partially filled-in CV
- [ ] One blog post
- [ ] One research project entry
- [ ] At least 4 commits in your Git history (visible on GitHub under the "commits" link)

---

## Troubleshooting

### Common Issues (Both OS)

| Problem | Solution |
|---------|----------|
| `quarto: command not found` | Close and reopen your terminal (or restart VS Code) after installing Quarto |
| SSH key not working | Make sure you added the `.pub` (public) key to GitHub, not the private one |
| `git push` says "permission denied (publickey)" | Your SSH key isn't set up correctly — redo Part 0.5 |
| `git push` rejected (non-fast-forward) | Run `git pull` first, resolve any conflicts, then push again |
| Site not deploying after push | Check Settings → Pages → Source is set to "GitHub Actions" |
| Site shows README instead of website | Make sure Source is "GitHub Actions", NOT "Deploy from branch" |
| GitHub Actions failing | Go to the Actions tab on your repo, click the failed run, read the error log |
| Images not showing | Check that the file path in your `.qmd` matches the actual file name (case-sensitive!) |
| `code .` doesn't open VS Code | See the tips in Part 1.2 for your OS |

### Windows-Specific Issues

| Problem | Solution |
|---------|----------|
| `ssh-keygen` not recognized | Make sure you're using **Git Bash**, not PowerShell or Command Prompt |
| `eval` command not working | You're in PowerShell — switch to Git Bash |
| `clip` command not working | Try: `cat ~/.ssh/id_ed25519.pub` and copy the output manually |
| VS Code terminal shows PowerShell | ++ctrl+shift+p++ → "Terminal: Select Default Profile" → choose Git Bash |
| Long file paths cause errors | Run `git config --global core.longpaths true` |
| Line ending warnings (`LF will be replaced by CRLF`) | This is harmless. To silence it: `git config --global core.autocrlf true` |

### Mac-Specific Issues

| Problem | Solution |
|---------|----------|
| `ssh-add --apple-use-keychain` error | Try `ssh-add -K ~/.ssh/id_ed25519` (older macOS syntax) |
| Xcode tools install hangs | Run `xcode-select --install` directly in Terminal |
| "developer tools" popup keeps appearing | Install Xcode Command Line Tools fully: `xcode-select --install` |
| Homebrew not found | Install from [brew.sh](https://brew.sh/) — not required but helpful |
| `.DS_Store` files showing in `git status` | Already handled by the `.gitignore` in the template |

---

## Resources

- [Quarto documentation](https://quarto.org/docs/guide/)
- [Quarto website tutorial](https://quarto.org/docs/websites/)
- [Git basics (Atlassian)](https://www.atlassian.com/git/tutorials)
- [GitHub SSH setup — full guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [Markdown cheat sheet](https://www.markdownguide.org/cheat-sheet/)
- [VS Code keyboard shortcuts — Windows](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf)
- [VS Code keyboard shortcuts — Mac](https://code.visualstudio.com/shortcuts/keyboard-shortcuts-macos.pdf)

---

## What Comes Next

Once you're comfortable with this workflow, we'll use the same tools for:

- **Lab documentation** — contributing to our shared lab guide
- **Research reproducibility** — running experiments on OSC and tracking results
- **Paper writing** — Quarto can render the same source to PDF (for ICML/ITSC) and HTML (for websites)
- **Presentations** — Quarto slides with `reveal.js`

The Git + Quarto skills you build here transfer directly to everything we do in the lab.
