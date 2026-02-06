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
Execute the query via the following command:

```bash
python src/services/skills/rag-query/scripts/query.py "<Internal_Policy_or_Company_Query>"