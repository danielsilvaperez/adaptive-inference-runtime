# Creating GitHub Issues for Parallel Tasks

This directory contains scripts and documentation for creating GitHub issues for all parallelizable tasks identified in ROADMAP.md.

## Quick Start

```bash
# 1. Authenticate with GitHub CLI
gh auth login

# 2. Create labels first (one-time setup)
./create_labels.sh

# 3. Create all 22 issues
./create_issues_with_gh.sh
```

## What Gets Created

- **22 GitHub Issues** - One for each parallel task from ROADMAP.md
- **27 Labels** - Organized by phase, priority, and technical area
- All issues include:
  - Detailed descriptions
  - Subtask checklists
  - Acceptance criteria
  - Phase dependencies
  - Proper labels

## Files

- `create_labels.sh` - Creates all necessary labels in the repository
- `create_issues_with_gh.sh` - Creates all 22 issues using GitHub CLI
- `create_parallel_issues.py` - Alternative Python script using GitHub API
- `ISSUES_TO_CREATE.md` - Complete documentation with all issue templates

## Task Distribution

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 0 | 2 | Foundations (repo setup, docs) |
| Phase 1 | 4 | Confidence scoring metrics |
| Phase 2 | 2 | Speculative decoding |
| Phase 3 | 5 | KV cache compression |
| Phase 4 | 1 | Memory management |
| Phase 5 | 4 | Benchmarking |
| Phase 6 | 4 | Developer UX |
| **Total** | **22** | All parallelizable tasks |

## Priority Breakdown

- **P0 (Critical):** 15 tasks - Critical path blockers
- **P1 (High):** 5 tasks - Important features
- **P2 (Medium):** 2 tasks - Nice to have

## Labels Created

### Phase Labels
- `phase-0` through `phase-6` - One for each development phase

### Priority Labels
- `P0`, `P1`, `P2`, `P3` - Priority levels

### Type Labels
- `parallel` - Can be worked on concurrently
- `sequential` - Must follow dependencies

### Technical Area Labels
- Infrastructure, documentation, routing, confidence-scoring
- Speculative-decoding, benchmarking, kv-compression
- Safety, memory-management, macos, visualization
- Developer-ux, configuration, packaging, publication

## Troubleshooting

### "Label not found" errors
Run `./create_labels.sh` first to create all labels.

### "Authentication required" errors
Run `gh auth login` and follow the prompts.

### Using Python script instead
```bash
export GITHUB_TOKEN='your_github_token'
python3 create_parallel_issues.py
```

Get a token at: https://github.com/settings/tokens (requires `public_repo` or `repo` scope)

## See Also

- `ROADMAP.md` - Complete project roadmap with all tasks
- `ISSUES_TO_CREATE.md` - Detailed issue templates
