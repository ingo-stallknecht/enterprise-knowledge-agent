# Contributing

Thank you for your interest in contributing to **Enterprise Knowledge Agent**.
This guide is intentionally short and compact.

## Setup

To install the project locally:
```bash
    git clone https://github.com/ingo-stallknecht/enterprise-knowledge-agent.git
    cd enterprise-knowledge-agent
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
```
## Before you commit

Run tests and formatting checks:
```bash
    pytest
    pre-commit run --all-files
```
## Pull Requests

- Keep changes focused and minimal.
- If behavior changes, update the README or documentation.
- Describe clearly **what changed** and **how to test it**.

Thank you for helping improve the project.
