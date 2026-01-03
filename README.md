# Reputation-Safe Agent Blueprint

A production-ready implementation of the reputation-safe agent architecture for customer-facing chatbots.This repository is the code implementation of the architecture discussed in Tolga Ayan's article. For more details, you can refer to the article here: [https://tolga-ayan.medium.com/a-reputation-safe-agent-blueprint-for-customer-facing-chatbots-529cc99da294]

## Architecture Overview

```
User Input
    │
    ▼
┌─────────────────────┐
│   Input Guardrail   │  ← Deterministic: blocks injection, flags abuse
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Supervisor      │  ← Routes to specialized sub-agents
│       Agent         │
└──────────┬──────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
┌───────┐ ┌───────┐ ┌───────┐
│ Order │ │Product│ │Support│  ← Domain-specific sub-agents
│ Agent │ │ Agent │ │ Agent │
└───┬───┘ └───┬───┘ └───┬───┘
    │         │         │
    │    ┌────┴────┐    │
    │    ▼         ▼    │
    │ ┌─────┐ ┌──────┐  │
    │ │Qdrant│ │Tavily│  │      ← RAG & Web Search
    │ │Cloud │ │Search│  │
    │ └─────┘ └──────┘  │
    └─────────┬─────────┘
              │
              ▼
┌─────────────────────┐
│   Auditor Agent     │  ← Reviews output (NO user context!)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Output Guardrail   │  ← Deterministic: PII redaction, disclaimers
└──────────┬──────────┘
           │
           ▼
      Response
```

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup
cd reputation-safe-agent
uv sync
```

## Configuration

```bash
# Required: Gemini API
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: Qdrant Cloud (for RAG)
export QDRANT_URL="https://your-cluster.cloud.qdrant.io"
export QDRANT_API_KEY="your-qdrant-api-key"

# Optional: Tavily (for web search)
export TAVILY_API_KEY="your-tavily-api-key"
```

### Getting API Keys

1. **Gemini API**: https://makersuite.google.com/app/apikey
2. **Qdrant Cloud**: https://cloud.qdrant.io (Free tier available)
3. **Tavily**: https://tavily.com (Free tier: 1000 searches/month)

## Usage

### Interactive Mode
```bash
uv run python main.py
```

### Demo Mode
```bash
uv run python main.py --demo
```

### Test Components
```bash
uv run python test_components.py
```

## Project Structure

```
reputation-safe-agent/
├── pyproject.toml
├── main.py
├── test_components.py
└── src/
    ├── config.py           # Configuration
    ├── models.py           # Data models
    ├── tools.py            # Tools for sub-agents
    ├── guardrails.py       # Input/Output guardrails
    ├── llm_client.py       # Gemini API wrapper
    ├── sub_agents.py       # Order, Product, Support agents
    ├── supervisor.py       # Supervisor/orchestrator
    ├── auditor.py          # Auditor agent
    ├── pipeline.py         # End-to-end pipeline
    ├── rag.py              # Qdrant Cloud RAG
    └── web_search.py       # Tavily web search
```

## License

MIT