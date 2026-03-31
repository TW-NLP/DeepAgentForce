---
name: rag-query
description: >
  Use this skill to query the internal corporate knowledge base via RAG.
  Trigger when the user asks about company policies, HR regulations, reimbursement rules,
  internal project specs, attendance systems, organizational procedures, or any proprietary
  enterprise knowledge. Do NOT use for general public knowledge or web searches.
version: 2.0.0
---

# Skill: Internal Knowledge Query (RAG)

Query the company's private knowledge graph and document repository for accurate,
tenant-scoped answers on internal topics.

**Applicable topics:**
- HR & admin: attendance, reimbursement, leave policies, onboarding procedures
- Projects: internal specs, architecture docs, meeting summaries
- Organization: department structures, role definitions, workflows

---

## Execution

**Command format (strictly follow — do not modify):**
```bash
<DEEPAGENTFORCE_ROOT>/src/services/skills/rag-query/scripts/query.py "<问题>" --tenant-uuid <UUID>
```

**How to determine `<DEEPAGENTFORCE_ROOT>`:**

`<DEEPAGENTFORCE_ROOT>` is the **absolute path to the DeepAgentForce project root** on the current host.
Resolve it at runtime before executing — do NOT hardcode or guess:
```bash
# Find the project root dynamically
find / -type d -name "DeepAgentForce" 2>/dev/null | head -1
# Example result: /home/user/projects/DeepAgentForce
```

Then substitute into the command:
```bash
python /home/user/projects/DeepAgentForce/src/services/skills/rag-query/scripts/query.py "<问题>" --tenant-uuid <UUID>
```

**Rules:**
1. Always resolve and use the **full absolute path** to `DeepAgentForce/` — never use relative paths
2. The question is a **positional argument** — wrap it in double quotes, no `--question` flag
3. Always include `--tenant-uuid` — the UUID is provided in the system prompt

**Examples:**

✅ `python /home/user/projects/DeepAgentForce/src/services/skills/rag-query/scripts/query.py "公司考勤制度是什么" --tenant-uuid 6d3b9398-ca84-4db5-a38b-e88ce427918b`

❌ Relative path (`src/services/...`) → execution will fail  
❌ Missing `--tenant-uuid` → query will hit wrong index  
❌ Using `--question "..."` instead of positional arg → argument ignored