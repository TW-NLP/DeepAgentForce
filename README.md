# AgenticRAG

**AgenticRAG** 是一个结合了 **GraphRAG（基于图的检索增强生成）** 技术与 **Agentic Workflow（代理工作流）** 的智能对话系统。该项目旨在提供一个高度可配置、支持多模态文档知识库构建以及深度问答的解决方案。

系统采用前后端分离架构：
- **Backend**: 基于 FastAPI 的高性能 Python 服务，负责逻辑处理、GraphRAG 管道和 LLM 交互。
- **Frontend**: 轻量级静态 Web 界面，提供直观的聊天窗口和配置管理面板。

---

## ✨ 核心功能

### 1. 🧠 智能对话 (Chat Agent)
- **多轮对话管理**: 支持具有上下文记忆的连续对话。
- **会话持久化**: 聊天记录自动保存至 `config/saved_history.json`，支持历史回溯。
- **会话隔离**: 独立的 Session ID 管理，支持创建新会话、清理历史和删除会话。

### 2. 🕸️ GraphRAG 知识库
不仅仅是向量检索，而是基于图谱的深度理解：
- **文档管理**: 支持上传 PDF, DOCX, TXT, MD, CSV 等多种格式文件。
- **自动化索引**: 文件上传后，后台自动触发实体提取、关系构建和社区检测（Community Detection）。
- **全局查询 (Global Query)**: 基于生成的社区摘要回答概括性问题，而非仅限于局部片段匹配。
- **状态监控**: 实时查看文档数、实体数、关系数及索引状态。

### 3. ⚙️ 动态配置系统
无需重启服务即可调整核心参数：
- **LLM 设置**: 动态切换模型（如 GPT-4o, Claude-3.5）、API Key 和 API URL。
- **搜索增强**: 集成 Tavily Search 配置。
- **爬虫配置**: 集成 Firecrawl 用于网页知识获取。
- **配置持久化**: 所有设置保存至 `config/saved_config.json`。

---

## 📂 项目结构

```text
AgenticRAG/
├── config/                  # [配置层]
│   ├── __init__.py
│   ├── prompts.py           # 提示词模版
│   ├── saved_config.json    # [自动生成] 用户保存的配置
│   ├── saved_history.json   # [自动生成] 历史对话记录
│   └── settings.py          # 全局设置加载器
├── src/                     # [源码层]
│   ├── api/                 # API 接口路由
│   │   ├── main.py          # FastAPI 应用入口
│   │   ├── routes.py        # 核心路由 (Chat, GraphRAG, Config)
│   │   └── websocket.py     # 实时通讯处理
│   ├── models/              # Pydantic 数据模型
│   ├── services/            # 业务服务逻辑
│   │   ├── llm_service.py   # LLM 交互封装
│   │   ├── rag_graph.py     # GraphRAG 核心管道实现
│   │   └── search_service.py # 联网搜索服务
│   ├── workflow/            # Agent 工作流引擎
│   │   ├── agent.py         # 对话代理实现
│   │   └── nodes.py         # 工作流节点定义
│   └── __init__.py
├── static/                  # [前端层] 纯静态文件
│   ├── chat.js              # 聊天逻辑
│   ├── config.js            # 设置页面逻辑
│   ├── index.html           # 主界面
│   └── knowledge.js         # 知识库管理逻辑
├── uploads/                 # [数据层] 上传文件临时存储
├── requirements.txt         # 项目依赖
├── run.py                   # 后端启动脚本
└── README.md                # 项目说明文档

```

---

## 🚀 快速开始

### 1. 环境准备

确保已安装 **Python 3.10+**。

```bash
# 1. 克隆项目
git clone <repository_url>
cd AgenticRAG

# 2. 创建并激活虚拟环境 (推荐)
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

```

### 2. 启动服务

本项目需要分别启动后端 API 和前端服务。

#### 🟢 启动后端 (Backend)

后端服务负责处理所有 API 请求。

```bash
# 在项目根目录下运行
python run.py

```

> *后端服务默认运行在 http://localhost:8000 (具体视 run.py 配置而定)*

#### 🔵 启动前端 (Frontend)

前端为静态文件，使用 Python 内置 HTTP 服务器运行。

```bash
# 进入 static 目录
cd static

# 启动 HTTP 服务 (端口 8080)
python -m http.server 8080

```

### 3. 访问应用

打开浏览器访问：
👉 **http://localhost:8080**

---

## 📖 使用指南

### 模型配置 (Model Configuration)

首次运行时，请点击左侧界面的 **"模型配置" (Settings)**：

1. **LLM Config**: 输入您的 LLM API Key (如 OpenAI Key)、LLM URL 和模型名称。
2. **GraphRAG Config**: 配置 Embedding 模型的 Key 和 URL（用于RAG服务）。
3. 点击 **Save** 保存。配置将写入 `config/saved_config.json`，无需重启。

### 构建知识库 (Knowledge Base)

1. 切换到 **"知识库"** 标签页。
2. 上传您的文档（PDF/TXT/Markdown）。
3. 系统会在后台自动进行 **GraphRAG 索引构建**（提取实体->构建关系->生成社区）。
4. 待索引状态变为 `Indexed` 后，即可进行基于图谱的问答。

### 对话 (Chat)

在首页对话框输入问题。

* 如果涉及知识库内容，Agent 会进行自主规划，进行问答。
* 对话历史会被自动保存。

---

## 🔌 API 接口概览

后端提供完整的 RESTful API，启动后可访问 `http://localhost:8000/docs` 查看 Swagger 文档。

| 模块 | 方法 | 路径 | 描述 |
| --- | --- | --- | --- |
| **对话** | POST | `/chat` | 发送消息并获取智能回复 |
| **对话** | GET | `/history/saved` | 获取持久化的历史记录 |
| **GraphRAG** | POST | `/graphrag/documents/upload` | 上传文档并触发索引 |
| **GraphRAG** | POST | `/graphrag/query` | 基于知识图谱的问答查询 |
| **GraphRAG** | GET | `/graphrag/index/status` | 查看索引构建进度与统计 |
| **配置** | GET | `/config` | 获取当前系统配置 (已脱敏) |
| **配置** | POST | `/config` | 更新系统配置 |

---

## 🛠️ 技术栈

* **Language**: Python 3.10+
* **Web Framework**: FastAPI
* **Data Validation**: Pydantic
* **RAG Engine**: GraphRAG (NetworkX, Community Detection)
* **Frontend**: Native HTML5/JS (No framework required)
* **Task Queue**: FastAPI BackgroundTasks (用于异步索引构建)

