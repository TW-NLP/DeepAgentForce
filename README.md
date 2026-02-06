# DeepAgentForce

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128%2B-009688)](https://fastapi.tiangolo.com/)
[![DeepAgents](https://img.shields.io/badge/DeepAgents-latest-orange)](https://github.com/deepagents/deepagents)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

**DeepAgentForce** 是一个基于 **DeepAgents 框架** 构建的下一代智能对话平台，深度融合了 **GraphRAG（图谱增强检索）** 认知能力与 **Agent Skills（技能系统）** 执行能力。

不同于传统 RAG 系统的片段检索，DeepAgentForce 通过知识图谱构建实现全局"理解"，并借助 DeepAgents 的 Skill 系统与自主规划能力"执行"复杂任务。系统采用现代化前后端分离架构，提供开箱即用的深度问答与知识库构建解决方案。

---

## ✨ 核心特性

### 1. 🧠 Agent Skills - 即插即用的技能系统

本项目在 `/src/services/skills` 目录下提供了两个示例 Skills，抛砖引玉：
- **PDF Skills** - PDF 文档处理能力
- **Web Search** - 联网搜索能力

#### 🔌 零配置扩展机制
- **即放即用**: 根据 Agent Skills 规范编写新的 Skill 后，直接放入 `skills/` 目录即可
- **自动融合**: 框架会自动发现并加载新增的 Skills，无需修改任何代码
- **热加载**: 支持运行时动态加载，无需重启服务

#### 📝 Skill 开发规范
[Agent Skills](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)

每个 Skill 只需包含：
- `SKILL.md` - 技能说明文档，定义能力和使用方法
- （可选）Python 脚本 - 实现复杂逻辑

Agent 会自动读取 `SKILL.md` 并理解如何使用该技能，实现真正的模块化扩展。

### 2. 🕸️ GraphRAG 知识图谱
从碎片化信息中重构知识网络：
- **深度理解**: 自动提取实体 (Entities)、构建关系 (Relationships) 并生成社区摘要 (Community Summaries)
- **全局查询**: 回答 "总结这几份文档的主要冲突点" 等宏观问题，突破传统向量检索局限
- **多模态支持**: 支持 PDF、DOCX、TXT、MD、CSV 等多种格式的即时索引与可视化状态监控
- **社区检测**: 基于图算法自动发现文档中的主题社区和知识聚类

### 3. 🎛️ 动态配置中枢
全热更新配置，无需重启服务：
- **模型热切换**: 随时在 GPT-4o、Claude-3.5、Qwen-Plus 或本地模型间切换
- **工具链集成**: 一键配置 Tavily（联网搜索）和 Firecrawl（网页爬取）等外部工具
- **持久化配置**: 所有系统参数自动保存至 `data/saved_config.json`，确保配置不丢失
- **实时生效**: 配置变更自动触发服务重建，确保最新参数立即应用

### 4. 👤 用户画像系统
沉淀对话记忆，构建越用越懂你的专属知识图谱：
- **深度偏好挖掘**: 结合 LLM 语义分析与 NetworkX 图算法（PageRank），量化提取用户关注的核心实体与关系
- **无感异步更新**: 采用 Fire-and-Forget 机制，在会话结束时通过后台线程池静默更新画像，零延迟不阻塞对话
- **动态侧写生成**: 实时将挖掘结果转化为结构化标签与自然语言摘要，持久化至 `data/person_like.json` 供前端可视化展示
- **个性化响应**: Agent 自动结合用户画像调整回答风格和内容侧重点

### 5. 💬 WebSocket 实时交互
现代化的流式对话体验：
- **双向实时通信**: 基于 WebSocket 的低延迟消息传输
- **思考过程可视化**: 实时展示 Agent 的推理步骤和工具调用状态
  - 🚀 开始思考
  - 🛠️ 调用工具
  - ✨ 执行完成
  - 🎉 流程结束
- **历史会话管理**: 侧边栏自动加载和恢复历史对话，支持会话切换
- **会话隔离**: 每个 WebSocket 连接独立的 Session，支持多用户并发

---

## 🚀 快速启动

### 1. 环境准备

确保您的环境已安装 **Python 3.12+**。

```bash
# 1. 克隆仓库
git clone https://github.com/TW-NLP/DeepAgentForce
cd AgentForce

# 2. 环境的准备

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n agent -y python=3.12
conda activate agent
pip install -r requirements.txt
# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

```

### 2. 启动 DeepAgentForce

系统由后端 API 和前端 UI 两部分组成。

#### 🟢 启动后端引擎 (Backend)

```bash
# 在项目根目录下运行
python main.py

```

> *后端 API 服务将启动在: http://localhost:8000*

#### 🔵 启动可视化终端 (Frontend)

```bash
# 进入静态资源目录
cd static

# 启动轻量级 Web 服务
python -m http.server 8080

```

### 3. 开始探索

打开浏览器访问可视化终端：
👉 **http://localhost:8080**

---

## 📖 操作指南

### 🔧 模型配置 (Model Config)

首次启动后，点击界面左侧的 **"模型配置"** 图标。
下面的 LLM、Embedding 参数配置需符合 OpenAI 规范。
Tavily 搜索配置请访问：[app.tavily.com](https://app.tavily.com/home) 获取 API Key。

1. **LLM Config**: 填入您的模型服务商信息 (API Key, URL, Model Name)。
2. **GraphRAG Config**: 配置 Embedding 模型参数（RAG服务）。
3. 点击 **保存配置**。系统会自动测试连接并持久化保存配置。
<div align="center">
  <img src="images/model_config.jpg" alt="模型配置" width="80%">
  <br>
  <em>模型配置</em>
</div>

### 📚 构建知识库 (Knowledge Base)

让 AgentForce 学习您的私有数据：

1. 进入 **"知识库" (Knowledge)** 标签页。
2. 拖拽上传文档 (PDF/Markdown/TXT)。
3. 观察控制台，AgentForce 会自动执行 **ETL 流程**：`文本分块` -> `实体提取` -> `关系构建` -> `社区检出`。
<div align="center">
  <img src="images/rag.jpg" alt="知识库上传" width="80%">
  <br>
  <em>知识库上传</em>
</div>

### 💬 智能交互 (Chat)

回到首页对话框：

* **提问**: "根据上传的财报，分析该公司去年的主要风险点。"
* **Agent 响应**: Agent 将自动识别意图，调用不同的agent进行回复，并综合生成深度回答。

<div align="center">
  <img src="images/chat.jpg" alt="智能交互" width="80%">
  <br>
  <em>智能交互</em>
</div>

### 👤 用户画像 (User Persona)

回到首页对话框：

* **提问**: "你好 你知道 我喜欢什么吗"
* **Agent 响应**: 你好！😊 根据我们的对话历史，我注意到你是一位深耕自然语言处理（NLP）领域的核心AI实践者——对NLP兴趣浓厚

<div align="center">
  <img src="images/chat_example.jpg" alt="效果展示" width="80%">
  <br>
  <em>问答效果展示</em>
</div>


<div align="center">
  <img src="images/person_like.jpg" alt="用户画像的展示" width="80%">
  <br>
  <em>用户画像</em>
</div>




## 🔌 API 开发文档

后端提供标准的 OpenAPI/Swagger 文档。服务启动后，访问 `http://localhost:8000/docs` 即可查看和调试接口。

| 模块 | Endpoint | 描述 |
| --- | --- | --- |
| **Chat** | `POST /chat` | 核心对话接口，触发 Agent 工作流 |
| **History** | `GET /history/saved` | 拉取持久化的历史会话 |
| **GraphRAG** | `POST /graphrag/upload` | 上传文件并触发异步索引任务 |
| **GraphRAG** | `POST /graphrag/query` | 执行图谱增强的深度查询 |
| **System** | `GET /config` | 获取当前系统运行参数 |

---

## 🛠 技术栈

* **Core**: Python 3.12+, LangChain (Optional), NetworkX
* **Web Framework**: FastAPI, Uvicorn
* **RAG Engine**: GraphRAG, Community Detection Algorithms
* **Frontend**: Vanilla JS (ES6+), HTML5, CSS3
* **Async Task**: Python `asyncio` & FastAPI `BackgroundTasks`

## Contact

**微信：** NLP技术交流群。

<img src="https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/images/chat.jpg" width="200" />

