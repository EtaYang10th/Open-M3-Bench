---
allowed-tools: Bash, Read, Edit, Glob, Grep, Write, TodoWrite
description: Perform a version release with automated PyPI publishing and Docker image builds
---

# Release

Automate the release process: version bump, changelog, tag creation, and trigger CI/CD for PyPI and Docker deployments.

## Workflow

1. Review commits since last release
2. Determine version bump (patch/minor/major)
3. Update `pyproject.toml` and `RELEASE_NOTES.md`
4. Commit, tag, and create GitHub release
5. Verify PyPI and Docker deployments

## Key Files

- `pyproject.toml` - Package version
- `RELEASE_NOTES.md` - Release changelog
- `.github/workflows/publish.yml` - PyPI & Docker publishing (triggered by GitHub release)

## Execute

### 1. Review Changes

```bash
# Get current version and recent tags
grep '^version = ' pyproject.toml
git tag --list 'v*' --sort=-version:refname | head -5

# Review commits since last release (replace with actual last tag)
git log v1.0.2..HEAD --oneline
```

### 2. Update Version

Version bump types:
- **Patch** (x.y.Z): Bug fixes, CI/CD, docs
- **Minor** (x.Y.0): New features, backward-compatible
- **Major** (X.0.0): Breaking changes

Edit `pyproject.toml`:
```toml
version = "X.Y.Z"
```

### 3. Update Release Notes

Add new section at top of `RELEASE_NOTES.md` following existing format:

```markdown
# MCP-NixOS: vX.Y.Z Release Notes - [Title]

## Overview
Brief description (1-2 sentences).

## Changes in vX.Y.Z
### 🚀 [Category]
- **Feature**: Description
### 📦 Dependencies
- Changes or "No changes from previous version"

## Installation
[Standard installation commands]

## Migration Notes
Breaking changes or "Drop-in replacement with no user-facing changes."

---
```

### 4. Commit and Tag

```bash
# Commit changes
git add pyproject.toml RELEASE_NOTES.md
git commit -m "chore: Bump version to X.Y.Z"
git commit -m "docs: Update RELEASE_NOTES.md for vX.Y.Z"
git push

# Create and push tag
git tag -a vX.Y.Z -m "Release vX.Y.Z: [description]"
git push origin vX.Y.Z
```

### 5. Create GitHub Release

```bash
gh release create vX.Y.Z \
  --title "vX.Y.Z: [Title]" \
  --notes "## Overview

[Brief description]

## Highlights
- 🚀 [Key feature]
- 🔒 [Important fix]

## Installation
```bash
pip install mcp-nixos==X.Y.Z
```

See [RELEASE_NOTES.md](https://github.com/utensils/mcp-nixos/blob/main/RELEASE_NOTES.md) for details."
```

### 6. Monitor Pipeline

```bash
# Watch workflow execution
gh run list --workflow=publish.yml --limit 3
gh run watch <RUN_ID>
```

## Verify

### PyPI
```bash
uvx mcp-nixos@X.Y.Z --help
```

### Docker Hub & GHCR
```bash
docker pull utensils/mcp-nixos:X.Y.Z
docker pull ghcr.io/utensils/mcp-nixos:X.Y.Z
docker run --rm utensils/mcp-nixos:X.Y.Z --help
```

## Report

Provide release summary:

```
✅ Release vX.Y.Z Complete!

**Version:** vX.Y.Z
**Release URL:** https://github.com/utensils/mcp-nixos/releases/tag/vX.Y.Z

### Verified Deployments
- ✅ PyPI: https://pypi.org/project/mcp-nixos/X.Y.Z/
- ✅ Docker Hub: utensils/mcp-nixos:X.Y.Z
- ✅ GHCR: ghcr.io/utensils/mcp-nixos:X.Y.Z
```

## Troubleshooting

**Workflow fails**: `gh run view <RUN_ID> --log-failed`
**PyPI unavailable**: Wait 2-5 min for CDN, check Test PyPI first
**Docker unavailable**: Wait 5-10 min for multi-arch builds
**Tag exists**: Delete with `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z`
