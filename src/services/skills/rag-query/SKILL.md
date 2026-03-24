---
name: rag-query
description: Access internal corporate regulations, policies, and proprietary project data.
version: 1.0.0
---

# Skill: Internal Knowledge Query (RAG)

## 🏢 Scope: Corporate Intelligence
Use this tool exclusively to retrieve information regarding **company policies**, **internal procedures**, and **proprietary organizational data**.

**Core Use Cases:**
* **Corporate Regulations:** HR policies, reimbursement workflows, and administrative guidelines.
* **Company Context:** Internal project specifications, architecture designs, and meeting summaries.
* **Knowledge Assets:** Information residing within the private knowledge graph or enterprise document repository.

---

## 🛠 Knowledge Sources
* **Official Handbooks:** Integrated corporate policy documents.
* **Project Repositories:** Internal documentation and technical specs.
* **Entity Graph:** Semantic relationships between internal departments, projects, and roles.

---

## 🚀 Execution

**⚠️ 重要：必须严格遵循以下命令格式，不得自行添加、删除或修改参数！**

执行 RAG 查询的命令格式：
```bash
python /***/DeepAgentForce/src/services/skills/rag-query/scripts/query.py "<要查询的问题>"
```

**示例：**
- ✅ 正确：`python /***/DeepAgentForce/src/services/skills/rag-query/scripts/query.py "公司的考勤制度是什么"`
- ❌ 错误：`python src/services/skills/rag-query/scripts/query.py "公司的考勤制度"`
- ❌ 错误：`python rag-query/scripts/query.py "公司的考勤制度"`

**注意：**
- 必须使用**完整绝对路径**（以 `/Users/tianwei/paper/DeepAgentForce/` 开头）
- `question` 是**位置参数**（positional argument），不是 `--question` 或 `--query`
- 必须用**双引号**将问题包裹