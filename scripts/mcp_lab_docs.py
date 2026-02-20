#!/usr/bin/env python3
"""MCP server exposing lab-setup-guide documentation as queryable tools.

Run with:
    uv run --with "mcp[cli]" --with pyyaml scripts/mcp_lab_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml
from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"

mcp = FastMCP(name="lab-setup-docs")


@mcp.tool()
async def search_docs(query: str, limit: int = 5) -> str:
    """Search lab documentation for a query string.

    Returns matching lines with surrounding context from all documentation pages.
    Useful for finding commands, configuration snippets, or topic references.

    Args:
        query: Case-insensitive search string (treated as literal text).
        limit: Maximum number of matches to return (default 5).
    """
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    matches: list[dict[str, object]] = []

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        if "includes" in md_file.parts:
            continue
        rel = md_file.relative_to(DOCS_DIR)
        lines = md_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if pattern.search(line):
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                context = "\n".join(
                    f"{'>' if j == i else ' '} {j + 1}: {lines[j]}"
                    for j in range(start, end)
                )
                matches.append({"file": str(rel), "line": i + 1, "context": context})
                if len(matches) >= limit:
                    break
        if len(matches) >= limit:
            break

    if not matches:
        return f"No results found for '{query}'."

    parts: list[str] = [f"Found {len(matches)} match(es) for '{query}':\n"]
    for m in matches:
        parts.append(f"### {m['file']}  (line {m['line']})\n```\n{m['context']}\n```\n")
    return "\n".join(parts)


@mcp.tool()
async def read_page(page_path: str) -> str:
    """Read the full markdown content of a documentation page.

    Args:
        page_path: Path relative to docs/, e.g. 'ml-workflows/pytorch-setup.md'.
    """
    target = (DOCS_DIR / page_path).resolve()

    # Validate the path stays within docs/
    if not str(target).startswith(str(DOCS_DIR)):
        return f"Error: path '{page_path}' resolves outside the docs directory."
    if not target.exists():
        return f"Error: page '{page_path}' not found. Use list_pages() to see available pages."
    if not target.suffix == ".md":
        return "Error: only .md files can be read."

    content = target.read_text(encoding="utf-8")
    # Extract title from first markdown heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else page_path

    return f"# {title}\n**Source:** `{page_path}`\n\n{content}"


@mcp.tool()
async def list_pages() -> str:
    """List all documentation pages organized by section.

    Returns the full navigation tree parsed from mkdocs.yml so you can
    discover which pages exist and find the right page_path for read_page().
    """
    config = yaml.safe_load(MKDOCS_YML.read_text(encoding="utf-8"))
    nav = config.get("nav", [])

    lines: list[str] = ["# Lab Setup Guide — Page Index\n"]
    _format_nav(nav, lines, depth=0)
    return "\n".join(lines)


def _format_nav(items: list, lines: list[str], depth: int) -> None:
    indent = "  " * depth
    for item in items:
        if isinstance(item, str):
            # Bare path without label
            lines.append(f"{indent}- `{item}`")
        elif isinstance(item, dict):
            for label, value in item.items():
                if isinstance(value, str):
                    lines.append(f"{indent}- **{label}**: `{value}`")
                elif isinstance(value, list):
                    lines.append(f"{indent}- **{label}**")
                    _format_nav(value, lines, depth + 1)


if __name__ == "__main__":
    print("lab-setup-docs MCP server starting…", file=sys.stderr)
    mcp.run(transport="stdio")
