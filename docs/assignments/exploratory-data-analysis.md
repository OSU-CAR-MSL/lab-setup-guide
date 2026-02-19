<!-- last-reviewed: 2026-02-19 -->
# Assignment 2a: Exploratory Data Analysis

|                    |                                                        |
| ------------------ | ------------------------------------------------------ |
| **Author**         | Robert Frenken                                         |
| **Estimated time** | 6--8 hours                                             |
| **Prerequisites**  | Assignment 1 completed, Python basics, GitHub account  |

---

## What You'll Build

An exploratory data analysis (EDA) of a real automotive intrusion detection dataset, published as a blog post on your Quarto website. You'll load and explore the data with pandas, create visualizations with matplotlib and seaborn, train two simple ML models with scikit-learn, and recreate two plots from the Python Graph Gallery.

---

## Part 0: Project Setup

### 0.1 Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `eda-assignment` (or similar)
3. Set to **Public**, check **"Add a README"**, add a **Python `.gitignore`**
4. Clone it to your machine:

```bash
git clone git@github.com:YOURUSERNAME/eda-assignment.git
cd eda-assignment
```

### 0.2 Set Up a Python Environment

=== "uv (recommended)"

    ```bash
    # Install uv if you haven't already
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a virtual environment and install packages
    uv venv
    source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
    uv pip install pandas matplotlib seaborn scikit-learn jupyter
    ```

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
    pip install pandas matplotlib seaborn scikit-learn jupyter
    ```

=== "conda"

    ```bash
    conda create -n eda python=3.11 pandas matplotlib seaborn scikit-learn jupyter -y
    conda activate eda
    ```

For more details on Python environments, see the [Python Environment Setup](../getting-started/python-environment-setup.md) guide.

### 0.3 Project Directory Layout

Organize your repository like this:

```
eda-assignment/
├── data/               # Raw and processed data (add to .gitignore if large)
├── figures/            # Saved plot images
├── notebooks/
│   └── eda.ipynb       # Your main analysis notebook
├── .gitignore
├── README.md
└── requirements.txt    # Pin your dependencies
```

Create the directories and save your dependencies:

```bash
mkdir -p data figures notebooks
pip freeze > requirements.txt   # or: uv pip freeze > requirements.txt
```

!!! warning "Don't commit large data files"
    Add `data/` to your `.gitignore` if the dataset exceeds a few MB. Git is not designed for large binary files.

---

## Part 1: Get & Explore the Data

### 1.1 Download the Dataset

Download the **HCRL Survival IDS** dataset from [ocslab.hksecurity.net/Datasets/survival-ids](https://ocslab.hksecurity.net/Datasets/survival-ids).

1. Visit the link and download the dataset files
2. Place the CSV file(s) in your `data/` directory
3. Open `notebooks/eda.ipynb` (create it in VS Code or with `jupyter notebook`)

### 1.2 Load and Inspect

```python
import pandas as pd

df = pd.read_csv("../data/survival_ids.csv")  # adjust filename as needed

# Basic inspection
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()
```

### 1.3 Summary Statistics

Run these in separate notebook cells and **read the output** — don't just run them blindly:

```python
# Data types and non-null counts
df.info()

# Descriptive statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Value counts for categorical columns (if any)
# df["column_name"].value_counts()
```

- [ ] Data loaded successfully with `pd.read_csv`
- [ ] `df.info()` output reviewed — you understand the column types
- [ ] `df.describe()` output reviewed — you can identify reasonable ranges
- [ ] Missing values checked

---

## Part 2: Visualizations

Create **at least 3** EDA plots. Use `matplotlib` and `seaborn`. Save each figure to `figures/`.

### 2.1 Distribution Plot

Pick a numeric column and plot its distribution:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["your_column"], bins=50, kde=True, ax=ax)
ax.set_title("Distribution of Your Column")
ax.set_xlabel("Value")
ax.set_ylabel("Count")
fig.savefig("../figures/distribution.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 2.2 Correlation Heatmap

Visualize relationships between numeric features:

```python
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = df.select_dtypes(include="number")
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.savefig("../figures/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 2.3 Categorical Breakdown

If the dataset has a label or class column, visualize its distribution:

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="label_column", ax=ax)
ax.set_title("Class Distribution")
ax.set_xlabel("Class")
ax.set_ylabel("Count")
fig.savefig("../figures/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
```

!!! tip "Make your plots readable"
    Always include a title, axis labels, and a legend (if applicable). Use `bbox_inches="tight"` when saving to avoid clipped labels.

- [ ] Distribution plot created and saved
- [ ] Correlation heatmap created and saved
- [ ] Categorical breakdown (or third plot of your choice) created and saved

---

## Part 3: Toy ML Models

Train two simple classifiers and evaluate them. This is a first exposure to the sklearn API — the goal is to learn the workflow, not to achieve state-of-the-art accuracy.

### 3.1 Prepare the Data

```python
from sklearn.model_selection import train_test_split

# Adjust column names to match your dataset
X = df.drop("label_column", axis=1)
y = df["label_column"]

# Handle non-numeric columns if needed
X = X.select_dtypes(include="number")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### 3.2 Train Two Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

### 3.3 Evaluate with Confusion Matrix

```python
from sklearn.metrics import classification_report, confusion_matrix

for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
    y_pred = model.predict(X_test)
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.savefig(f"../figures/confusion_matrix_{name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
```

- [ ] Train/test split created
- [ ] Two models trained (Logistic Regression + Random Forest)
- [ ] `classification_report` printed for both models
- [ ] Confusion matrix heatmaps created and saved

---

## Part 4: Graph Gallery Picks

Visit the [Python Graph Gallery](https://python-graph-gallery.com/) and pick **2 plots** that look interesting. Recreate them using the Survival IDS dataset (or a subset of it).

### Requirements

1. Choose 2 different chart types (e.g., violin plot, radar chart, hexbin, pair plot, bubble chart)
2. Adapt the gallery code to use columns from your dataset
3. Customize each plot: change the color palette, add proper titles/labels, add annotations if relevant
4. Save both figures to `figures/`

!!! tip "Picking good chart types"
    Don't just pick the simplest plots. Try something you haven't used before — the point is to expand your visualization toolkit. Good picks: violin plots, pair plots, radar charts, parallel coordinates, ridgeline plots.

- [ ] Graph Gallery plot 1 created, customized, and saved
- [ ] Graph Gallery plot 2 created, customized, and saved

---

## Part 5: Publish as a Blog Post

Turn your analysis into a blog post on your Quarto website from Assignment 1.

### 5.1 Clean Your Notebook

1. Restart the kernel and run all cells top-to-bottom (++ctrl+shift+f5++ in VS Code)
2. Remove any scratch/debug cells
3. Add markdown cells that explain what you're doing and what the results mean — a reader should understand the analysis without reading the code

### 5.2 Add a YAML Header

Add this to the **first cell** of your notebook (as a Raw cell) or convert the notebook to `.qmd`:

```yaml
---
title: "Exploratory Data Analysis: Survival IDS Dataset"
description: "EDA and baseline ML models on the HCRL Survival IDS dataset."
date: "2026-02-19"
categories: [eda, python, machine-learning]
---
```

### 5.3 Publish on Your Quarto Site

1. Copy the notebook (or `.qmd` file) to your Quarto site's `posts/` folder
2. Copy any required figures to the post directory
3. Preview locally: `quarto preview`
4. Commit and push:

```bash
git add posts/eda-post/
git commit -m "Add EDA blog post"
git push
```

- [ ] Notebook runs cleanly top-to-bottom
- [ ] Markdown explanations added between code cells
- [ ] Blog post published and live on your Quarto site

---

## Final Deliverables

Submit the following:

- [ ] **GitHub repo URL** for your EDA project (e.g., `github.com/YOURUSERNAME/eda-assignment`)
- [ ] **5+ visualizations** in `figures/` (3 EDA + 2 Graph Gallery picks)
- [ ] **Confusion matrix** heatmaps for both models
- [ ] **Blog post live** on your Quarto site
- [ ] **`requirements.txt`** in the repo root

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'pandas'` | Make sure your virtual environment is activated and you've installed the packages |
| Jupyter kernel doesn't see installed packages | Select the correct kernel — click the kernel name in the top-right of VS Code and pick your `.venv` |
| `quarto render` fails on notebook | Restart kernel, run all cells, fix any errors, then try again |
| Data file too large for Git | Add `data/` to `.gitignore` — don't commit large files to Git |
| Heatmap annotations overlap | Reduce the number of features or use `annot=False` for large matrices |
| `SettingWithCopyWarning` | Use `.copy()` when creating subsets: `X = df.drop(...).copy()` |
