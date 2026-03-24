---
name: web-search
description: Retrieve real-time information from the internet.
version: 2.0.0
---

# Web Search Skill

## When to Use
- User asks about current events, news, or real-time information
- Need to search the web for specific topics
- Finding facts that require up-to-date data

## 🚀 Execution

**⚠️ 重要：必须严格遵循以下命令格式！**

执行网络搜索的命令格式：
```bash
python /***/DeepAgentForce/src/services/skills/web-search/scripts/web_search.py "<搜索关键词>"
```

**示例：**
- ✅ 正确：`python /***DeepAgentForce/src/services/skills/web-search/scripts/web_search.py "今天的天气"`
- ❌ 错误：`python src/services/skills/web-search/scripts/web_search.py "今天的天气"`

**注意：**
- 必须使用**完整绝对路径**
- 将搜索关键词用**双引号**包裹
- 支持中文和英文搜索
