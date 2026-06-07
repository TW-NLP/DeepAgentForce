# DeepAgentForce

<p align="center">
  <img src="images/logo.png" alt="DeepAgentForce Logo" width="180"/>
</p>

<p align="center">
  <strong>A Production-Grade Multi-Tenant Agent Harness</strong><br>
  <em>Progressive disclosure of Skills, Tools and MCP — built on LangGraph + deepagents</em>
</p>

<p align="center">
  <a href="README_CN.md">🇨🇳 中文文档</a> &nbsp;|&nbsp;
  <a href="https://github.com/TW-NLP/DeepAgentForce">GitHub</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.128%2B-009688?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/LangGraph-Latest-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MCP-Supported-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

---

## What is DeepAgentForce?

DeepAgentForce is a **production-grade Agent Harness** — a runtime that gives agents a place to live, run, and scale across real multi-user environments.

It is not just another chatbot interface. The focus is on the operating layer:

- How an agent selects and invokes skills at scale (progressive disclosure, not a flat list)
- How dozens or hundreds of tools/MCP servers stay out of the context window until needed
- How multiple users share the same platform while keeping sessions, knowledge, configs, and skills fully isolated
- How RAG, long-term user memory, tools, and custom logic connect into one coherent runtime

---

## Key Differentiators

### 1. Progressive Disclosure — Skills, Tools & MCP in Three Tiers

Inspired by the [hermes-agent](https://github.com) architecture but rebuilt natively on LangGraph + LangChain:

| Layer | Skills | Built-in Tools | Extra / MCP Tools |
|-------|--------|---------------|-------------------|
| Always in context | Category overview only | Full list (15 tools, small schema) | Bridge tools only (≤4 stubs) |
| On demand — tier 1 | `skills_list(category)` → name+description | — | `tool_search` / `mcp_search` → hybrid retrieve + re-rank |
| On demand — tier 2 | `skill_view(name)` → full SKILL.md | — | `tool_describe(name)` → param schema |
| Execution | `shell` → run skill script | Direct call (already bound) | `tool_invoke(name, args)` → proxy |

**Why it matters:** A deployment with 19 skills, 50 MCP tools, and 10 custom tools keeps the same constant context overhead — only a handful of bridge stubs (≤4) are ever in the context window regardless of how many tools you add.

```
Threshold gate (tool_disclosure.py):
  extra_tools schema tokens < 10% context  →  bind directly (no overhead)
  extra_tools schema tokens ≥ 10% context  →  switch to Hi-RAG bridges
```

#### Hi-RAG — Hierarchical Tool Selection (Type → Service → Tool)

For large tool / MCP repositories, the disclosure layer uses **Hi-RAG**, a structure-aware, coarse-to-fine retrieval (inspired by *Hi-RAG: A Hierarchical Framework for Scalable and Generalizable Tool Selection*) that mirrors the natural `Type → Service → Tool` hierarchy of MCP:

<p align="center">
  <img src="images/hi-rag-framework.png" alt="Hi-RAG Framework" width="100%"/>
</p>

- **Stage 1 · Coarse-grained retrieval (hybrid):** BM25 (lexical) + embedding (semantic) fused with weighted RRF (`k=60, α=0.1`) over **tool** descriptions — *Tool-as-Proxy*: retrieve tools first, then roll up to their parent services.
- **Stage 2 · Fine-grained re-ranking (type-aware):** candidates re-ranked by embedding cosine over the combined `Type + Service + Tool` description; only the top results are surfaced to the LLM.
- **Two entry points:** `tool_search` for custom tools (`Type → Tool`, 2-tier) and `mcp_search` for MCP (`Type → Service → Tool`, 3-tier); both share `tool_describe` / `tool_invoke` and **gracefully fall back to pure BM25** when no embedding endpoint is configured.
- Each MCP server / custom tool carries a **Type** from a fixed 8-class taxonomy, used as the coarsest re-ranking signal.

**Results.** On the MCP tool-selection benchmark, Hi-RAG consistently beats Flat RAG on accuracy while keeping token cost close to Flat RAG — and far below feeding the whole *Full Service* list — across five backbones, in both single- and multi-turn settings:

<sub>**S** = Single-Turn (Top-1 Acc), **M** = Multi-Turn (Top-3 Acc). Acc ↑ higher is better; Avg Tokens ↓ lower is better. **∆Acc** is vs. the Flat RAG baseline.</sub>

| Model | Method | S·Acc (%) ↑ | S·Tokens ↓ | S·∆Acc | M·Acc (%) ↑ | M·Tokens ↓ | M·∆Acc |
|-------|--------|----------:|---------:|-------:|----------:|---------:|-------:|
| **Qwen3-8B** | Full Service | 38.8 | 423.5 | – | 11.7 | 739.0 | – |
| | Flat RAG | 72.6 | 170.8 | base | 36.7 | 198.7 | base |
| | **Hi-RAG (Ours)** | **78.1** | 176.5 | **+5.5** | **43.3** | 288.6 | **+6.6** |
| **Qwen3-32B** | Full Service | 47.8 | 610.0 | – | 21.7 | 1306.4 | – |
| | Flat RAG | 75.6 | 82.7 | base | 38.3 | 134.4 | base |
| | **Hi-RAG (Ours)** | **82.6** | 87.4 | **+7.0** | **48.3** | 132.9 | **+10.0** |
| **QwQ-32B** | Full Service | 45.3 | 597.5 | – | 20.0 | 1416.3 | – |
| | Flat RAG | 75.6 | 682.8 | base | 28.3 | 1353.3 | base |
| | **Hi-RAG (Ours)** | **82.6** | 621.4 | **+7.0** | **36.7** | 1573.6 | **+8.4** |
| **DeepSeek-V3** | Full Service | 41.3 | 167.2 | – | 6.7 | 271.2 | – |
| | Flat RAG | 75.1 | 111.3 | base | 33.3 | 236.6 | base |
| | **Hi-RAG (Ours)** | **82.6** | 111.4 | **+7.5** | **40.0** | 239.3 | **+6.7** |
| **GPT-4o-mini** | Full Service | 40.3 | 108.6 | – | 6.7 | 199.6 | – |
| | Flat RAG | 77.1 | 77.3 | base | 28.3 | 159.2 | base |
| | **Hi-RAG (Ours)** | **83.1** | 76.4 | **+6.0** | **31.7** | 168.6 | **+3.4** |

### 2. MCP Integration (Model Context Protocol)

Connect any MCP server — the same config format as Claude Desktop:

```json
{
  "mcpServers": {
    "slack":  { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-slack"] },
    "github": { "url": "https://my-mcp-server/github", "headers": { "Authorization": "Bearer ..." } }
  }
}
```

- Supports `stdio`, `streamable_http`, and `sse` transports
- Multi-tenant: global shared config + per-tenant override file
- Tools auto-prefixed `mcp__<server>__<tool>` to avoid conflicts
- Full async support — `tool_invoke` uses `ainvoke` for MCP tools
- Managed via Web UI (add, test connection, enable/disable)

### 3. Multi-Tenant by Design

Every resource is scoped to `tenant_uuid`:

| Resource | Isolation |
|----------|-----------|
| Chat sessions | `thread_id = tenant_uuid + session_id` |
| RAG knowledge base | Separate ChromaDB collection per tenant |
| Skills | Shared built-in + private `data/skill/<uuid>/` directory |
| MCP configs | Global read-only + `data/mcp_servers_<uuid>.json` |
| Custom tools | `data/agent_tools_custom/<uuid>/` |
| User profile | PageRank-based preference graph per user |
| File outputs | `data/outputs/<uuid>/` |

### 4. Custom Tool Sandbox

End users can upload Python tools via the Web UI. They run in an **isolated subprocess** — the main process never imports user code:

- `subprocess` + `rlimit`: CPU 10s, file 16 MB, memory 1 GB (Linux)
- Wall-clock timeout: 20s
- Environment variable filtering: strips any variable containing `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`
- New session + temp cwd per invocation
- Proxy tools via `pydantic.create_model` preserve the original arg schema

### 5. Standard Open-Source Stack

Built on LangGraph + LangChain + deepagents — not a custom runtime:

| Capability | Implementation |
|------------|---------------|
| Agent graph | LangGraph `create_deep_agent` + `MemorySaver` |
| Multi-turn memory | LangGraph `MemorySaver` (per thread_id) |
| Tool binding | LangChain `BaseTool` / `StructuredTool` |
| LLM access | `init_chat_model` — any OpenAI-compatible endpoint |
| Streaming | `astream(stream_mode="messages")` |
| Observability | WebSocket event callbacks (tool start/end, answer phase) |

### 6. Built-in RAG Pipeline

Not a plugin — a first-class runtime capability:

- ChromaDB local persistence, per-tenant collection isolation
- PDF / DOCX / TXT / CSV / Markdown ingestion
- Optional BM25 hybrid retrieval, reranking, and query rewrite
- Accessible via the `rag-query` skill from any conversation

### 7. Long-Term User Memory

`person_like_service.py` extracts entities and relationships from conversations, builds a NetworkX knowledge graph per user, and scores topics via PageRank + connection weight + mention frequency. The agent gets a personalized summary injected at every session start.

---

## Architecture Overview


```
┌─────────────────────────────────────────────────────────┐
│  Frontend (static/)  Chat · Knowledge · Skills · Config  │
└────────────────────────┬────────────────────────────────┘
                         │ WebSocket / REST
┌────────────────────────▼────────────────────────────────┐
│  API Layer (FastAPI)                                     │
│  routes · auth_routes · skills_routes                    │
│  tools_routes · mcp_routes · websocket                   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Agent Runtime (ConversationalAgent)                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Skills     │  │ Common Tools │  │ Extra/MCP Tools│  │
│  │ Disclosure  │  │  (15 tools)  │  │  Disclosure    │  │
│  │skills_list  │  │  utils·web   │  │  tool_search   │  │
│  │skill_view   │  │  memory      │  │  tool_invoke   │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  LangGraph  create_deep_agent + MemorySaver         │ │
│  └─────────────────────────────────────────────────────┘ │
└──────┬─────────────────────────┬───────────────────┬────┘
       │                         │                   │
┌──────▼──────┐  ┌───────────────▼──────┐  ┌────────▼────┐
│  RAG        │  │  Skill Manager       │  │  Sandbox    │
│  ChromaDB   │  │  built-in + custom   │  │  rlimit +   │
│  per-tenant │  │  skills/<cat>/<name> │  │  subprocess │
└─────────────┘  └──────────────────────┘  └─────────────┘
```

---

## Built-in Skills (19 skills across 5 categories)

| Category | Skills |
|----------|--------|
| `design` | algorithmic-art, brand-guidelines, canvas-design, frontend-design, slack-gif-creator, theme-factory, web-artifacts-builder |
| `development` | claude-api, mcp-builder, skill-creator, webapp-testing |
| `document` | docx, pdf, pptx, xlsx |
| `research` | rag-query, web-search |
| `writing` | doc-coauthoring, internal-comms |

Each skill lives in `skills/<category>/<name>/SKILL.md` + `scripts/`. The agent never loads them all — it uses `skills_list` → `skill_view` on demand.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/TW-NLP/DeepAgentForce
cd DeepAgentForce
docker compose up -d
```

Visit `http://localhost:8000` — then go to **Settings** to configure your LLM and Embedding API keys.

### Local

```bash
git clone https://github.com/TW-NLP/DeepAgentForce
cd DeepAgentForce

conda create -n agent python=3.12 -y
conda activate agent
pip install -r requirements.txt

python main.py
```

Mirror for users in China:
```bash
pip install -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host=mirrors.aliyun.com
```

`.env` minimum config:
```bash
SQLITE_DB_PATH=data/deepagentforce.db
JWT_SECRET_KEY=your-secret-key-change-in-production
HOST=127.0.0.1
PORT=8000
```

---

## User Journey

### 1. Register & Login
`http://localhost:8000/login.html` — every user gets an isolated workspace.

### 2. Configure Models
Go to **Settings** and fill in:

| Field | Example |
|-------|---------|
| `LLM_URL` | `https://api.openai.com/v1` |
| `LLM_API_KEY` | `sk-xxxxxxxx` |
| `LLM_MODEL` | `gpt-4o` |
| `EMBEDDING_URL` | `https://api.openai.com/v1` |
| `EMBEDDING_API_KEY` | `sk-xxxxxxxx` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |

### 3. Upload Knowledge Documents
Supports PDF, DOCX, TXT, CSV, Markdown — all indexed into a per-tenant ChromaDB collection.

### 4. Configure MCP Servers *(optional)*
Go to **Skills → MCP tab** → Add Server. Use the same JSON format as Claude Desktop. Click **Test Connection** before saving.

### 5. Upload Custom Tools *(optional)*
Go to **Skills → Tools tab** → Add Tool. Upload a `.py` file — any top-level function with a docstring becomes a callable agent tool, sandboxed automatically.

### 6. Start Chatting
The agent auto-selects skills, searches the knowledge base, calls tools, and synthesizes answers in natural language.

---

## Project Structure

```
DeepAgentForce/
├── main.py
├── config/settings.py
├── src/
│   ├── api/
│   │   ├── routes.py           # core chat + file routes
│   │   ├── skills_routes.py    # skill CRUD
│   │   ├── tools_routes.py     # custom tool CRUD
│   │   ├── mcp_routes.py       # MCP server CRUD
│   │   ├── auth_routes.py
│   │   └── websocket.py
│   └── services/
│       ├── conversational_agent.py   # agent assembly
│       ├── skill_disclosure.py       # skills progressive disclosure
│       ├── tool_disclosure.py        # tools BM25 progressive disclosure
│       ├── mcp_integration.py        # MCP connector + config store
│       ├── custom_tool_manager.py    # user-uploaded Python tools
│       ├── sandbox/                  # subprocess isolation
│       │   ├── runner.py
│       │   ├── loader.py
│       │   └── tool_worker.py
│       ├── agent_tools/              # 15 built-in common tools
│       │   ├── utils.py
│       │   ├── web.py
│       │   └── memory.py
│       ├── skill_manager.py
│       ├── rag.py
│       └── person_like_service.py
├── src/services/skills/              # 19 built-in skills
│   ├── design/
│   ├── development/
│   ├── document/
│   ├── research/
│   └── writing/
├── static/                           # Web UI
│   ├── js/i18n.js                    # EN/ZH language switcher
│   ├── index.html
│   ├── login.html
│   └── register.html
├── scripts/                          # test suites
│   ├── test_sandbox.py               # 15/15
│   ├── test_mcp_integration.py       # 14/14
│   ├── test_tools_mcp_mgmt.py        # 20/20
│   ├── test_routes_http.py           # 19/19
│   └── test_optimizations.py         # 30/30
└── data/                             # runtime data (gitignored)
```

---

## API Reference

Swagger: `http://localhost:8000/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ws/stream` | WebSocket | Streaming conversation |
| `/api/chat` | POST | Single-turn chat |
| `/api/chat/upload` | POST | Chat with file attachment |
| `/api/auth/register` | POST | Register |
| `/api/auth/login` | POST | Login |
| `/api/skills` | GET | List skills |
| `/api/skills/install` | POST | Install skill |
| `/api/tools` | GET | List tools (built-in + MCP + custom) |
| `/api/tools/custom` | POST | Upload custom tool |
| `/api/mcp/servers` | GET | List MCP servers |
| `/api/mcp/servers` | POST | Add/update MCP server |
| `/api/mcp/servers/test` | POST | Test MCP connection |
| `/api/rag/documents/upload` | POST | Upload knowledge document |
| `/api/rag/query` | POST | RAG query |

---

## 📰 Changelog

- **2026-06-07** — `v2.1.0` Hi-RAG Edition
  - Hi-RAG hierarchical tool selection (`Type → Service → Tool`), inspired by *Hi-RAG: A Hierarchical Framework for Scalable and Generalizable Tool Selection*
  - Two entry points: `tool_search` (custom tools, 2-tier) and `mcp_search` (MCP, 3-tier), sharing `tool_describe` / `tool_invoke`
  - Hybrid coarse retrieval: BM25 + embedding fused via weighted RRF, with type-aware fine re-ranking; graceful fallback to pure BM25 when no embedding endpoint is set
  - Fixed 8-class Type taxonomy for MCP servers / custom tools (`tool_taxonomy.py`)
  - ≤4 bridge stubs in context regardless of repository size

- **2026-06-02** — `v2.0.0` Progressive Disclosure Edition
  - Skills progressive disclosure: `skills_list` / `skill_view` two-tier system
  - BM25 tool search: `tool_search` / `tool_describe` / `tool_invoke` bridge
  - Full MCP integration (stdio + HTTP, multi-tenant config)
  - Custom Python tool sandbox (subprocess + rlimit)
  - Web UI: Skills / Tools / MCP management tabs
  - 15 built-in common tools (utils + web + memory)
  - 19 skills reorganized into 5 categories
  - EN/ZH frontend language switcher

- **2026-04-23** — `v1.4.0`
  - Docker build optimization + macOS DMG / Windows EXE packaging
  - SQLite as default database

- **2026-04-22** — `v1.3.0` — Skill zip upload, dialogue improvements

- **2026-04-21** — `v1.2.0` — 20 Claude built-in skills, regenerate/edit in chat

---

## Use Cases

- Agent platform thesis / research prototype
- Enterprise internal knowledge assistant
- Multi-user AI workbench with isolated data
- Extensible tool-calling agent with MCP ecosystem access
- Chinese NLP + RAG + proofreading pipeline

---

## FAQ

**Why can't I chat right after Docker starts?**
The LLM keys are not pre-configured. Go to Settings and fill in `LLM_*` and `EMBEDDING_*` fields first.

**How do I add a new skill?**
Create a directory with `SKILL.md` + `scripts/`, zip it, and upload via Skills → Add Skill. Or drop it directly into `src/services/skills/<category>/`.

**How do I clear all Docker data?**
```bash
docker compose down -v
```

---

## License

MIT — free to use, modify, and distribute commercially.

---

## Acknowledgements

- [LangChain / LangGraph](https://github.com/langchain-ai/langchain) — Agent framework
- [deepagents](https://pypi.org/project/deepagents/) — Skill-aware agent builder
- [FastAPI](https://github.com/tiangolo/fastapi) — Web framework
- [ChromaDB](https://github.com/chroma-core/chroma) — Local vector store

---

## Star History
<p align="center">
  <a href="https://github.com/TW-NLP/DeepAgentForce">
    <img src="https://img.shields.io/github/stars/TW-NLP/DeepAgentForce?style=social"/>
  </a>
</p>
