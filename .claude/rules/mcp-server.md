# Lab-Docs MCP Server

`scripts/mcp_lab_docs.py` is a FastMCP server exposing documentation as queryable tools for any Claude Code session.

## Tools

| Tool | Purpose |
|------|---------|
| `search_docs(query, limit=5)` | Case-insensitive search across all pages; returns matching lines with context |
| `read_page(page_path)` | Full markdown content (e.g. `ml-workflows/pytorch-setup.md`) |
| `list_pages()` | Navigation tree from `mkdocs.yml` |

## Configuration

Registered in `~/.claude.json` under the `lab-docs` key. Uses `uv run --with` for `mcp[cli]` and `pyyaml` at runtime.

```bash
# Test
uv run --with "mcp[cli]" --with pyyaml scripts/mcp_lab_docs.py
# Verify in Claude Code: /mcp → "lab-docs" with 3 tools
```
