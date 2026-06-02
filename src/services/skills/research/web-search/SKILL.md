---
name: web-search
description: >
  Use this skill to search the internet for real-time or up-to-date information.
  Trigger when the user asks about current events, recent news, live data (weather, prices,
  scores), unfamiliar topics requiring web verification, or anything that may have changed
  since the model's knowledge cutoff. Do NOT use for questions answerable from internal
  corporate knowledge — use rag-query instead.
version: 2.0.0
---

# Skill: Web Search (Tavily)

Search the public internet via Tavily API and return structured results for synthesis.

**Applicable scenarios:**
- Current events, breaking news, recent announcements
- Real-time data: weather, stock prices, sports scores, exchange rates
- Product research, technical documentation, third-party references
- Any fact that may be outdated in the model's training data

---

## Execution

**Command format (strictly follow — do not modify):**
```bash
python <DEEPAGENTFORCE_ROOT>/src/services/skills/web-search/scripts/web_search.py "<搜索词>" [--max-results N] [--output file.json]
```

**How to determine `<DEEPAGENTFORCE_ROOT>`:**

`<DEEPAGENTFORCE_ROOT>` is the **absolute path to the DeepAgentForce project root** on the current host.
Resolve it at runtime before executing:
```bash
find / -type d -name "DeepAgentForce" 2>/dev/null | head -1
# Example result: /home/user/projects/DeepAgentForce
```

**Parameters:**

| Parameter | Required | Default | Description |
|---|---|---|---|
| `query` | ✅ | — | Search query, wrapped in double quotes |
| `--max-results` | ❌ | 5 | Number of results to return (suggested: 5–10) |
| `--output` | ❌ | — | Save full JSON response to a file |

**Examples:**

✅ Basic search:
```bash
python /home/user/projects/DeepAgentForce/src/services/skills/web-search/scripts/web_search.py "2024年诺贝尔奖得主"
```

✅ More results:
```bash
python /home/user/projects/DeepAgentForce/src/services/skills/web-search/scripts/web_search.py "latest React 19 features" --max-results 10
```

✅ Save to file:
```bash
python /home/user/projects/DeepAgentForce/src/services/skills/web-search/scripts/web_search.py "machine learning trends 2025" --output results.json
```

❌ Relative path (`src/services/...`) → execution will fail  
❌ Missing quotes around query with spaces → argument parsing error  
❌ Using this skill for internal HR/policy questions → use `rag-query` instead

---

## Output Format

The script outputs structured JSON to stdout:
```json
{
  "query": "搜索词",
  "total_results": 5,
  "results": [
    {
      "title": "页面标题",
      "url": "https://...",
      "snippet": "摘要内容（最多500字符）"
    }
  ]
}
```

Parse this output to synthesize a final answer for the user. Do not return raw JSON directly.