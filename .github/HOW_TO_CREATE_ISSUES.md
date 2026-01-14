# How to Create GitHub Issues

This guide explains how to create the 22 parallel task issues defined in `ISSUES_TO_CREATE.md`.

## Prerequisites

You need to create GitHub issues for this repository. There are three methods available, listed in order of recommendation.

## Method 1: Using Python Script (Recommended)

The Python script (`create_parallel_issues.py`) uses the GitHub REST API to create issues.

### Steps:

1. **Create a GitHub Personal Access Token:**
   - Go to https://github.com/settings/tokens
   - Click "Generate new token" → "Generate new token (classic)"
   - Give it a descriptive name (e.g., "AIR Issue Creator")
   - Select scopes:
     - For public repositories: `public_repo`
     - For private repositories: `repo` (full control)
   - Click "Generate token" and copy the token

2. **Set the token as an environment variable:**
   ```bash
   export GITHUB_TOKEN='your_token_here'
   ```

3. **Run the script:**
   ```bash
   python3 create_parallel_issues.py
   ```

The script will create all 22 issues with proper labels and formatting.

## Method 2: Using GitHub CLI

The bash script (`create_issues_with_gh.sh`) uses the GitHub CLI tool.

### Steps:

1. **Authenticate with GitHub CLI:**
   ```bash
   gh auth login
   ```
   Follow the prompts to authenticate.

2. **Make the script executable:**
   ```bash
   chmod +x create_issues_with_gh.sh
   ```

3. **Run the script:**
   ```bash
   ./create_issues_with_gh.sh
   ```

The script will create all 22 issues sequentially.

## Method 3: Manual Creation via GitHub Web Interface

If automated methods don't work, you can manually create issues:

1. **Run the Python script without authentication to see issue details:**
   ```bash
   python3 create_parallel_issues.py
   ```
   This will print all issue titles, labels, and bodies without creating them.

2. **For each issue:**
   - Go to https://github.com/danielsilva010/Adaptive-Inference-Runtime/issues/new
   - Copy the title and body from the script output
   - Add the labels manually
   - Click "Submit new issue"

Alternatively, refer to `ISSUES_TO_CREATE.md` which contains all issue templates formatted for easy copying.

## Verification

After running the scripts, verify that:
- 22 issues were created
- Each issue has the correct labels (phase-X, parallel, priority, technical area)
- Issue titles match the format: `[Phase X] Task X.Y: Description`

## Issue Summary

The 22 issues cover:
- **Phase 0** (2 issues): Repository structure and documentation
- **Phase 1** (4 issues): Confidence scoring metrics
- **Phase 2** (2 issues): Speculative decoding
- **Phase 3** (5 issues): KV cache compression
- **Phase 4** (1 issue): MacBook memory management
- **Phase 5** (4 issues): Benchmarking and publication
- **Phase 6** (4 issues): Developer UX improvements

## Troubleshooting

### Authentication Errors
- Ensure your GitHub token has the correct scopes
- For private repos, you need `repo` scope
- For public repos, `public_repo` scope is sufficient

### Rate Limiting
- GitHub API has rate limits (5,000 requests/hour for authenticated users)
- Creating 22 issues should be well within this limit
- If you hit rate limits, wait an hour and try again

### Script Errors
- Verify Python 3 is installed: `python3 --version`
- Verify required packages: `pip3 install requests`
- Verify gh CLI is installed: `gh --version`

## Support

For issues with the scripts or questions about the tasks:
1. Check the `ISSUES_TO_CREATE.md` for detailed task descriptions
2. Refer to `ROADMAP.md` for full project context
3. Open a discussion in the repository if you need help
