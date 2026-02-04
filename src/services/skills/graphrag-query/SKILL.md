---
name: graphrag-query
description: Answer questions using internal documents and enterprise knowledge via GraphRAG.
version: 1.0.0
---

# GraphRAG Internal Knowledge Query Skill

## 🎯 Purpose & Scope

This skill is designed **exclusively for local / internal knowledge-based question answering**.

Use this skill when the question:
- Refers to **company documents**, internal papers, design docs, reports
- Mentions **uploaded files**, PDFs, Word, Markdown, CSV, etc.
- Requires **long-term organizational knowledge**
- Needs answers grounded in a **private knowledge graph**
- Should NOT rely on real-time internet information

❌ **Do NOT use this skill for**:
- Real-time news
- Public web search
- Up-to-date external facts
- Open-domain exploration

For those cases, use `web-search` instead.

---

## 🧠 Knowledge Source

- Internal documents uploaded to GraphRAG
- Automatically constructed **entity–relationship knowledge graph**
- Community-based summaries and semantic retrieval
- Answers are generated via **GraphRAG Map-Reduce reasoning**

---

## 🚀 Standard Execution Command

**IMPORTANT**: Always run from the project root using this exact path structure:

```bash
python src/services/skills/graphrag-query/scripts/graphrag_query.py "<question>"
