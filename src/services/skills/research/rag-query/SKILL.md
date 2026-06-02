---
name: rag-query
description: >
  Use this skill to query the internal corporate knowledge base via RAG.
  Trigger when the user asks about company policies, HR regulations, reimbursement rules,
  internal project specs, attendance systems, organizational procedures, or any proprietary
  enterprise knowledge. Do NOT use for general public knowledge or web searches.
version: 3.0.0
---

# Skill: Internal Knowledge Query (RAG)

Query the company's private knowledge graph and document repository for accurate,
tenant-scoped answers on internal topics.

**Applicable topics:**
- HR & admin: attendance, reimbursement, leave policies, onboarding procedures
- Projects: internal specs, architecture docs, meeting summaries
- Organization: department structures, role definitions, workflows

---

## ⚠️ Pre-execution Checklist（执行前必读）

在执行任何命令前，确认以下三项均已就绪：

| 项目 | 来源 | 示例 |
|------|------|------|
| 项目根路径 | system prompt 中的 `项目根路径` 字段 | `/Users/tianwei/paper/DeepAgentForce` |
| 租户 UUID | system prompt 中的 `当前租户 UUID` 字段 | `6d3b9398-ca84-4db5-a38b-e88ce427918b` |
| 用户问题 | 用户输入的原始问题 | `公司桥梁工程相关资料` |

> 🚫 **严禁**执行 `find /`、`ls`、`cd` 来查找路径 —— 路径已由系统注入 system prompt，直接读取使用。

---

## Execution

### 命令格式（严格遵循，不得修改任何参数）
```bash
python <PROJECT_ROOT>/src/services/skills/rag-query/scripts/query.py "<用户问题>" --tenant-uuid <UUID>
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `<PROJECT_ROOT>` | 路径 | 从 system prompt `项目根路径` 字段读取，**禁止**自行推断或 find |
| `"<用户问题>"` | 位置参数（positional） | 用户的原始问题，**必须用双引号包裹** |
| `--tenant-uuid <UUID>` | 选项参数（flag） | 从 system prompt `当前租户 UUID` 字段读取，**不可省略** |

### 格式规则
```
✅ 正确格式：
python /Users/tianwei/paper/DeepAgentForce/src/services/skills/rag-query/scripts/query.py "公司考勤制度是什么" --tenant-uuid 6d3b9398-ca84-4db5-a38b-e88ce427918b

❌ 错误 - JSON 数组格式：
["python /path/query.py \"公司考勤制度\" --tenant-uuid xxx"]

❌ 错误 - 相对路径：
python src/services/skills/rag-query/scripts/query.py "问题" --tenant-uuid xxx

❌ 错误 - 使用 --question flag：
python /path/query.py --question "公司考勤制度" --tenant-uuid xxx

❌ 错误 - 缺少 --tenant-uuid：
python /path/query.py "公司考勤制度"

❌ 错误 - 先执行 find 查路径：
find / -type d -name "DeepAgentForce"
```

---

## Step-by-step 执行流程
```
Step 1. 从 system prompt 读取 项目根路径，记为 PROJECT_ROOT
Step 2. 从 system prompt 读取 当前租户 UUID，记为 UUID  
Step 3. 将用户问题用双引号包裹，记为 QUESTION
Step 4. 拼接完整命令：
        python {PROJECT_ROOT}/src/services/skills/rag-query/scripts/query.py "{QUESTION}" --tenant-uuid {UUID}
Step 5. 通过 shell 工具以纯文本字符串形式执行（不加任何 JSON 包装）
Step 6. 将脚本输出结果整理后返回给用户
```

---

## Output Handling

脚本返回 JSON 格式结果，包含以下字段：
```json
{
  "answer": "RAG 检索到的答案文本",
  "sources": ["来源文档1", "来源文档2"],
  "confidence": 0.95
}
```

**处理规则：**
- 优先使用 `answer` 字段内容作为回复主体
- 如存在 `sources`，以"参考来源"形式附在回复末尾
- 如 `answer` 为空或置信度过低，告知用户知识库中暂无相关内容，建议联系管理员补充文档

---

## Error Handling

| 错误信息 | 原因 | 处理方式 |
|----------|------|----------|
| `No such file or directory` | 路径错误 | 重新从 system prompt 读取 PROJECT_ROOT，确认路径正确 |
| `ModuleNotFoundError` | Python 环境问题 | 尝试用 `python3` 替代 `python` |
| `tenant not found` | UUID 错误或为空 | 检查 system prompt 中的 UUID 是否完整 |
| 返回空结果 | 知识库无相关内容 | 告知用户并建议补充文档，不要重复查询 |
| 脚本超时 | 网络或向量库问题 | 提示用户稍后重试，不要无限重试 |

> 🚫 遇到任何错误，**禁止**通过修改路径格式、添加 find 命令来自行"修复"——应直接报告错误信息。