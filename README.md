# DeepAgentForce

<p align="center">
  <img src="images/logo.png" alt="DeepAgentForce Logo" width="180"/>
</p>

<p align="center">
  <strong>一个面向真实场景的 Agent Harness</strong>
  <br>
  <em>An extensible harness for multi-tenant agents, skills, memory and RAG</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/FastAPI-0.128%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-Latest-blueviolet?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ed?style=for-the-badge&logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

---

## 项目简介

 **DeepAgentForce**，它的重点不是单次问答效果，而是为智能体提供一套可持续运行、扩展、隔离和观测的底座。  
相比“聊天页面 + 一个模型接口”的常见项目，这个项目更关注下面这些运行层问题：

- Agent 如何按规则选择并调用技能,支持无限扩展的skills
- 多用户环境下如何做会话、知识、配置和技能隔离
- 如何把 RAG、工具、用户记忆接进同一个运行时
- 如何在不重写主流程的情况下继续扩展能力

围绕这些问题，DeepAgentForce 做成了一个前后端一体化的智能体运行框架与工作台。

如果这个项目对您有帮助，欢迎 Star ✨ 。

---

## News

- 2026-04-21 发布 `V1.2.0`
- 新增 Claude 官网 20 个 Skills
- 优化智能对话交互，新增重新生成与编辑功能
- 2026-04-20 发布 `V1.1.0`
- 本次版本集成了文本校对能力
- 优化了知识库管理 UI
- 优化了 Skill 管理 UI

---

## 为什么说它是 Harness

<details>
<summary>展开查看 DeepAgentForce 作为 Agent Harness 的设计逻辑</summary>

### 1. 它提供的是智能体运行底座，不只是对话入口

项目已经把一个 Agent 在真实系统里常见的几类能力接到了同一个运行时中：

- 对话入口
- WebSocket 流式交互
- Skills 调度
- RAG 检索
- 用户画像与偏好沉淀
- 文件输入输出
- 用户认证与租户隔离

这就是 Harness 的价值所在：它不是替你定义单个 Agent 的能力，而是给 Agent 提供一个可运行、可接入、可扩展的宿主环境。

### 2. Skill-first 设计让它像 Harness，而不是写死逻辑的应用

项目内置了 `web-search`、`rag-query`、`pdf-processing` 等技能，Agent 会先读取技能目录中的 `SKILL.md`，再按规范执行脚本。  
这意味着主流程和技能实现是解耦的，你可以把业务能力作为 Skill 挂上来，而不必不断侵入核心对话逻辑。

如果把项目当成 Harness 来看，这一层就是它的“能力接插件接口”。

### 3. 多租户隔离让 Harness 可以真正服务多人场景

系统的会话、配置、RAG 索引、用户技能、用户画像都带有租户隔离能力：

- 会话按 `tenant_uuid + session_id` 区分
- RAG 按租户维护独立 collection / 元数据
- 用户上传的 Skill 存放在租户专属目录
- 用户画像持久化到租户独立文件
- API / WebSocket 会从 JWT 或 Header 中解析租户信息

这很关键，因为很多 Agent Demo 能跑，但一进入多用户环境就开始混数据、混配置、混知识。  
DeepAgentForce 把这些问题前置处理掉了，所以更像平台底座，而不是个人脚本。

### 4. Harness 不只接知识，还接“长期用户记忆”

`person_like_service.py` 中实现了基于对话的用户偏好挖掘：

- 用 LLM 从对话中抽取实体和关系
- 用 NetworkX 构建用户知识图谱
- 用 PageRank、连接权重、提及频次综合估计用户偏好

相比只保留最近几轮上下文，这种方式更适合长期使用场景。

从 Harness 视角看，这代表系统不只是执行一次任务，而是在持续积累用户状态。

### 5. RAG 在这里不是外挂，而是运行时内的标准能力

RAG 模块支持分阶段增强：

- 向量召回
- 可选 BM25 关键词召回
- 可选 Rerank 重排
- 可选 Query Rewrite 多路检索投票

也就是说，这里的 RAG 不是外挂脚本，而是 Harness 内的一等能力，可以自然接入 Agent 的任务流程。

### 6. 对开发者和用户来说，它都是一个可观测的工作台

项目直接由 FastAPI 托管静态页面，开箱即有：

- 登录 / 注册
- 聊天页面
- 配置页面
- 知识库页面
- Skills 管理页面
- 输出文件浏览
- WebSocket 流式交互与状态展示
- 工具调用与中间状态传递

这意味着 Harness 不只是“在后端偷偷跑”，而是可以被直接演示、调试和交付。

</details>

---

## Harness 能力一览

| Harness 维度 | DeepAgentForce 提供什么 |
|:-------------|:------------------------|
| Agent 接入 | 对话 Agent、流式会话、状态回调 |
| Skill 编排 | 基于 `SKILL.md` 的技能发现与执行 |
| Knowledge 接入 | 多租户 RAG、文档上传、检索增强 |
| Memory | 用户偏好图谱与长期画像 |
| Isolation | 会话、配置、技能、知识的租户隔离 |
| Observability | WebSocket 事件、思考步骤、输出展示 |
| Extensibility | 内置 Skill + 用户上传 Skill |
| Delivery | FastAPI + 静态前端一体化部署 |

---

## 系统架构

<div align="center">
  <img src="images/frame.png" alt="系统架构" width="90%"/>
</div>

整体架构可以概括为 5 层：

1. 展示层：`static/` 下的聊天、配置、知识库、技能管理等页面
2. 接口层：FastAPI 路由、认证、WebSocket、文件上传下载
3. 智能体层：`ConversationalAgent` 负责模型接入、技能工作空间和对话流程
4. 能力层：Skill Manager、RAG Pipeline、用户画像、校对服务
5. 存储层：MySQL、Milvus Lite、本地 data 目录

如果从 Harness 的角度看，这 5 层分别对应：

1. 用户与开发者交互界面
2. 统一接入层
3. 智能体运行时
4. 可插拔能力模块
5. 状态与数据持久化

---

## 快速开始

### 方式一：Docker 启动

```bash
git clone https://github.com/TW-NLP/DeepAgentForce
cd DeepAgentForce
docker compose up -d
```

启动后访问：

- 首页：http://localhost:8000
- Swagger：http://localhost:8000/docs

`docker-compose.yml` 会启动：

- `mysql`：MySQL 8.4
- `app`：DeepAgentForce 应用

注意：容器启动成功后，你仍需要在前端配置页补充模型与 embedding 参数，系统才能正常调用大模型。

### 方式二：本地运行

环境要求：

- Python 3.12+
- MySQL 8.0+

安装步骤：

```bash
git clone https://github.com/TW-NLP/DeepAgentForce
cd DeepAgentForce

conda create -n agent python=3.12 -y
conda activate agent

pip install -r requirements.txt
```

国内网络可使用镜像：

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

创建数据库：

```sql
CREATE DATABASE deepagentforce CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'agent'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepagentforce.* TO 'agent'@'%';
FLUSH PRIVILEGES;
```

然后准备 `.env`：

```bash
DB_HOST=localhost
DB_PORT=3306
DB_NAME=deepagentforce
DB_USERNAME=agent
DB_PASSWORD=your_password
JWT_SECRET_KEY=your-secret-key-change-in-production
```

启动：

```bash
python main.py
```

---

## 上手路径

### 1. 注册并登录

访问 `http://localhost:8000/login.html`，注册后系统会为当前用户分配独立工作空间。

<div align="center">
  <img src="images/login.png" alt="模型配置" width="82%"/>
</div>


### 2. 配置模型

进入“配置”页面，至少填写以下字段：

| 配置项 | 说明 |
|--------|------|
| `LLM_URL` | 对话模型 API 地址 |
| `LLM_API_KEY` | 对话模型 Key |
| `LLM_MODEL` | 对话模型名称 |
| `EMBEDDING_URL` | 向量模型 API 地址 |
| `EMBEDDING_API_KEY` | 向量模型 Key |
| `EMBEDDING_MODEL` | 向量模型名称 |

说明：

- `LLM_URL` 和 `EMBEDDING_URL` 填 base URL 即可
- 系统会自动处理 `/chat/completions`、`/embeddings` 等后缀
- 也支持给校对服务单独配置模型

<div align="center">
  <img src="images/config.png" alt="模型配置" width="82%"/>
</div>

### 3. 上传知识文档

知识库模块支持常见格式：

- PDF
- DOC / DOCX
- TXT
- CSV
- Markdown

文档会被解析、切分、向量化，并写入租户专属索引。

<div align="center">
  <img src="images/rag.png" alt="知识库管理" width="88%"/>
</div>

### 4. 开始对话

在聊天页中，系统会根据问题自动决定是否：

- 直接回答
- 调用 Skill
- 查询知识库
- 结合历史与用户偏好生成回复

<div align="center">
  <img src="images/chat.png" alt="智能对话界面" width="88%"/>
</div>

---

## 内置模块

### Agent Skills

内置 Skill：

| Skill | 作用 |
|-------|------|
| `web-search` | 联网搜索与网页信息获取 |
| `rag-query` | 面向私有知识库的问答 |
| `pdf-processing` | PDF 文本、表格与处理任务 |

你也可以通过前端安装自定义 Skill。对每个租户来说：

- 内置 Skill 全员可见
- 用户 Skill 仅当前租户可见
- Skill 内容由 `SKILL.md + scripts/*.py` 组成

示例结构：

<div align="center">
  <img src="images/skill.png" alt="Skill 管理界面" width="88%"/>
</div>

### 用户画像

用户画像不是简单的标签存储，而是一个逐步演化的图结构。系统会从长期对话中抽取：

- 关注主题
- 技术偏好
- 常见任务类型
- 潜在表达习惯

这部分能力尤其适合做“越用越像你的助手”。


### 校对服务

项目还内置了中文校对能力，支持：

- 通用 LLM 校对
- 独立校对模型接入
- 分句 / 分块并发处理

如果你的场景不仅要“回答”，还要“润色、改错、校验”，这一块可以直接复用。

<div align="center">
  <img src="images/ChineseErrorCorrect4.png" alt="ChineseErrorCorrect4 校对" width="88%"/>
</div>

---

## API 概览

启动后可访问 Swagger：

- http://localhost:8000/docs

常用接口：

| Endpoint | 方法 | 说明 |
|----------|:----:|------|
| `/api/chat` | POST | 同步对话 |
| `/api/chat/upload` | POST | 携带附件对话 |
| `/ws/stream` | WebSocket | 流式对话 |
| `/api/auth/register` | POST | 注册 |
| `/api/auth/login` | POST | 登录 |
| `/api/skills` | GET | 获取 Skill 列表 |
| `/api/skills/install` | POST | 安装 Skill |
| `/api/rag/documents/upload` | POST | 上传知识文档 |
| `/api/rag/query` | POST | RAG 查询 |

---

## 项目结构

```text
DeepAgentForce/
├── main.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config/
│   ├── settings.py
│   └── prompts.py
├── src/
│   ├── api/
│   │   ├── routes.py
│   │   ├── auth_routes.py
│   │   ├── skills_routes.py
│   │   └── websocket.py
│   ├── database/
│   ├── models/
│   ├── services/
│   │   ├── conversational_agent.py
│   │   ├── skill_manager.py
│   │   ├── rag.py
│   │   ├── person_like_service.py
│   │   ├── proofread_service.py
│   │   └── skills/
│   └── workflow/
├── static/
├── images/
└── data/
```

---

## 适合什么场景

如果你正在做下面这些方向，这个项目会比较有参考价值：

- 智能体平台课程 / 毕设 / 实验项目
- Agent Harness / Agent Platform 原型
- 企业内部知识助手
- 多用户共享的 AI 工作台
- 可扩展工具型 Agent 原型
- 面向中文场景的对话 + 校对 + 知识库系统

---

## 常见问题

### Docker 启动后为什么还不能直接聊天？

因为数据库和应用虽然已经启动，但模型配置默认是空的。  
需要先到配置页补充 `LLM_*` 与 `EMBEDDING_*` 参数。

### 为什么这个项目适合做二次开发？

因为它的能力边界比较清楚：

- 对话由 `ConversationalAgent` 统一编排
- 知识库能力集中在 `rag.py`
- Skill 扩展集中在 `skill_manager.py + skills/`
- 多租户逻辑主要在 API 和服务层传递 `tenant_uuid`

### 如何清空 Docker 数据？

```bash
docker compose down -v
```

这会删除数据库卷，请谨慎使用。

---

## License

本项目采用 **MIT License**，可自由使用、修改和分发，商用无忧。

---

## Contact

**微信：** NLP技术交流群

<img src="https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/images/chat_.jpg" width="200" />

---

## 致谢

本项目基于以下优秀的开源项目构建：

- [LangChain / LangGraph](https://github.com/langchain-ai/langchain) — Agent 开发框架
- [FastAPI](https://github.com/tiangolo/fastapi) — 高性能 Web 框架
- [Milvus](https://github.com/milvus-io/milvus) — 向量数据库

---

<p align="center">
  <br><br>
  <a href="https://github.com/TW-NLP/DeepAgentForce">
    <img src="https://img.shields.io/github/stars/TW-NLP/DeepAgentForce?style=social" alt="GitHub Stars"/>
  </a>
</p>
