# DeepAgentForce

<p align="center">
  <img src="images/logo.png" alt="DeepAgentForce Logo" width="180"/>
</p>

<p align="center">
  <strong>新一代智能体协同系统 — 让 AI 真正成为你的智能助手</strong>
  <br>
  <em>Self-Evolving AI Agent with Modular Skills & Knowledge Intelligence</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/FastAPI-0.128%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangGraph-Latest-blueviolet?style=for-the-badge" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Stars-Welcome-orange?style=for-the-badge" alt="Stars"/>
</p>

---

## 🎯 项目简介

**DeepAgentForce** 是一款基于大语言模型的**智能体协同系统**，它不仅仅是传统的 RAG 问答系统，更是一个具有**自我进化能力**的 AI 助手平台。

- 通过模块化的 **Agent Skills** 架构，系统支持无限扩展工具能力
- 通过**用户画像挖掘**技术，AI 会越用越懂你
- 通过**思考过程可视化**，你可以清晰看到 AI 是如何分析和解决问题的
- 通过**多租户架构**，支持团队协作和企业级部署

---

## ✨ 核心特性

| 特性 | 传统 AI | DeepAgentForce |
|:-----|:-------:|:--------------:|
| 知识库问答 | ✅ | ✅ |
| 工具自动调用 | ❌ | ✅ |
| 模块化技能扩展 | ❌ | ✅ |
| 用户偏好学习 | ❌ | ✅ |
| 思考过程可视化 | ❌ | ✅ |
| 多会话管理 | ❌ | ✅ |
| 多租户支持 | ❌ | ✅ |

---

## 🏗️ 系统架构

<div align="center">
  <img src="images/frame.png" alt="系统架构" width="90%"/>
</div>

**前后端一体化部署**：后端 FastAPI 直接托管所有静态资源（HTML / JS / CSS），无需独立前端服务器，一条命令即可启动完整服务。

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | 3.12+ |
| MySQL | 8.0+ |
| pip | 最新版 |

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/TW-NLP/DeepAgentForce
cd DeepAgentForce

# 2. 创建虚拟环境
conda create -n agent python=3.12 -y
conda activate agent

# 3. 安装依赖
pip install -r requirements.txt

# 中国用户可使用镜像加速
pip install -r requirements.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host=mirrors.aliyun.com
```

### 数据库配置

```bash
# 1. 安装并启动 MySQL 8.0+

# 2. 创建数据库
mysql -u root -p
CREATE DATABASE deepagentforce CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'agent'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON deepagentforce.* TO 'agent'@'%';
FLUSH PRIVILEGES;
EXIT;

# 3. 配置环境变量（可选，创建 .env 文件）
cat > .env << EOF
DB_HOST=localhost
DB_PORT=3306
DB_NAME=deepagentforce
DB_USERNAME=agent
DB_PASSWORD=your_password
JWT_SECRET_KEY=your-secret-key-change-in-production
EOF
```

### 启动服务

```bash
# 仅需一条命令，前后端同时启动（默认端口 8000）
python main.py
```

访问 [http://localhost:8000](http://localhost:8000) 开始使用！


---

## 🔐 多租户认证系统

DeepAgentForce 支持完整的多租户用户认证系统，满足团队协作和企业级部署需求。

### 主要功能

- **👤 用户注册与登录** — 支持用户名/邮箱注册和登录
- **🏢 多租户架构** — 每个团队/公司拥有独立的工作空间
- **🔑 JWT Token 认证** — 安全的无状态认证机制
- **👥 团队协作** — 支持邀请成员加入团队
- **🔄 Token 刷新** — 安全的 Token 自动续期机制

### 技术实现

| 组件 | 技术 |
|------|------|
| 数据库 | MySQL |
| ORM | SQLAlchemy |
| 密码加密 | Bcrypt |
| 认证协议 | JWT (HS256) |
| 后端框架 | FastAPI |

---

## 📰 更新日志 (News)



## 📸 功能展示

### 💬 智能对话

<div align="center">
  <img src="images/chat.jpg" alt="智能对话界面" width="90%"/>
</div>

### 🛠️ 可视化 Skill 管理

<div align="center">
  <img src="images/skill.png" alt="Skill 管理界面" width="90%"/>
</div>

### 📚 知识库管理

<div align="center">
  <img src="images/rag.png" alt="知识库管理界面" width="90%"/>
</div>

### 🧠 思考过程可视化

```
🤔 初始化          →  接收用户任务，开始分析
   ↓
🔧 调用工具        →  识别需要调用的 Skill / 工具
   ↓
✅ 工具执行完成     →  获取执行结果
   ↓
🎯 生成回答        →  综合分析，生成最终回复
```

---

## 🛠️ Agent Skills 系统

### 什么是 Agent Skills？

Agent Skills 是 DeepAgentForce 的**模块化扩展系统**，每个 Skill 都是一个独立的功能模块，可以被 AI Agent 自动发现和调用。

### 预置 Skills

| Skill | 功能 | 使用场景 |
|-------|------|----------|
| 📄 **pdf-processing** | PDF 文档处理 | 提取文本、表格、合并/拆分、OCR |
| 🔍 **rag-query** | 企业知识库问答 | 私有文档智能问答 |
| 🌐 **web-search** | 联网搜索 | 实时获取网络信息 |

### 自定义 Skills

只需创建符合规范的目录结构：

```
src/services/skills/
└── my-skill/
    ├── SKILL.md          # 技能描述文件
    └── scripts/
        └── main.py       # 执行脚本
```

**SKILL.md 规范示例：**

```yaml
---
name: my-awesome-skill
description: 技能描述，说明何时使用
version: 1.0.0
---

# Skill 使用说明

## 何时使用
描述该技能适用的场景

## 执行命令

python scripts/main.py "<参数>"

```

---

## 👤 动态用户画像

系统会自动从对话中学习用户的：

- 🎯 **职业背景** — 了解用户的专业领域
- 💻 **技术偏好** — 掌握用户常用的技术栈
- 📝 **交互风格** — 适配用户的回答偏好
- 🧠 **上下文记忆** — 持续学习，越用越聪明

---

## 📖 使用指南

### 1. 注册 / 登录

首次使用访问 [http://localhost:8000](http://localhost:8000)，完成注册并创建工作空间（或直接加入已有团队）。

### 2. 模型配置

进入左侧 **"配置"** 页面，填写以下信息：

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `LLM_URL` | 模型 API 地址（支持 OpenAI / OpenRouter 等兼容接口） | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | 模型 API Key | `sk-or-...` |
| `LLM_MODEL` | 模型名称 | `anthropic/claude-3.5-sonnet` |

> **注意**：`LLM_URL` 填入 API 的 base 地址即可（不含 `/chat/completions` 后缀），系统会自动处理。

<div align="center">
  <img src="images/config.png" alt="模型配置" width="80%"/>
</div>

### 3. 构建知识库（可选）

让 AI 学习你的私有知识：

1. 进入 **"知识库"** 页面
2. 拖拽或选择文档（PDF / Word / TXT / Markdown）
3. 系统自动向量化并建立索引

### 4. 开始对话

直接在对话框中提问，AI 会自动：

- 分析用户意图
- 判断是否需要调用工具
- 检索相关知识
- 生成最优回答

---

## 📡 API 文档

后端提供完整的 RESTful API，启动后访问 [http://localhost:8000/docs](http://localhost:8000/docs) 查看 Swagger 交互文档。

### 认证 API

| Endpoint | 方法 | 说明 |
|----------|:----:|------|
| `/api/auth/register` | POST | 用户注册 |
| `/api/auth/login` | POST | 用户登录 |
| `/api/auth/refresh` | POST | 刷新 Token |
| `/api/auth/me` | GET | 获取当前用户信息 |
| `/api/auth/logout` | POST | 用户登出 |

### 对话与知识库 API

| Endpoint | 方法 | 说明 |
|----------|:----:|------|
| `/api/chat` | POST | 发送对话消息 |
| `/api/ws/stream` | WebSocket | 流式对话 |
| `/api/rag/documents/upload` | POST | 上传文档 |
| `/api/rag/query` | POST | 知识库问答 |
| `/api/history/saved` | GET | 获取历史会话 |

### 文件管理 API

| Endpoint | 方法 | 说明 |
|----------|:----:|------|
| `/api/output/files` | GET | 获取输出文件列表 |
| `/api/output/files/preview` | GET | 预览文件内容 |
| `/api/output/files/download` | GET | 下载文件 |

---

## 🔧 配置说明

配置文件位于 `data/saved_config.json`，也可通过前端界面进行配置：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `LLM_URL` | 模型 API 地址（base URL） | — |
| `LLM_API_KEY` | 大模型 API Key | — |
| `LLM_MODEL` | 模型名称 | — |
| `EMBEDDING_URL` | Embedding API 地址 | — |
| `EMBEDDING_API_KEY` | Embedding API Key | — |
| `EMBEDDING_MODEL` | Embedding 模型 | — |

> **URL 填写说明**：在模型配置页面填入 API 的 base 地址（如 `https://openrouter.ai/api/v1`），系统内部会自动拼接正确的接口路径，无需手动添加 `/chat/completions` 或 `/embeddings` 后缀。

---

## 🗂️ 项目结构

```
DeepAgentForce/
├── main.py                          # 后端入口（同时托管前端静态资源）
├── requirements.txt                 # 依赖列表
├── README.md                        # 项目文档
├── .env                             # 环境变量配置
├── config/
│   └── settings.py                  # 配置管理
├── src/
│   ├── api/
│   │   ├── routes.py                # 主路由
│   │   ├── websocket.py             # WebSocket 处理
│   │   ├── skills_routes.py         # Skills 路由
│   │   └── auth_routes.py           # 认证路由
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py            # 数据库连接
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py                  # 用户和租户模型
│   ├── services/
│   │   ├── conversational_agent.py  # 对话 Agent
│   │   ├── rag.py                   # 知识库 / RAG
│   │   ├── person_like_service.py   # 用户画像
│   │   ├── auth_service.py          # 认证服务
│   │   └── skills/                  # Agent Skills
│   │       ├── rag-query/
│   │       ├── web-search/
│   │       └── pdf-processing/
│   └── utils/                       # 工具函数
├── static/
│   ├── index.html                   # 主页面
│   ├── login.html                   # 登录页面
│   ├── register.html                # 注册页面
│   ├── skills.html                  # Skills 管理页面
│   ├── chat.js                      # 对话逻辑
│   ├── auth.js                      # 认证管理
│   ├── config.js                    # 配置管理
│   ├── knowledge.js                 # 知识库管理
│   ├── output.js                    # 输出文件管理
│   └── skills.js                    # Skills 管理
├── images/                          # README 图片资源
└── data/
    ├── sessions/                    # 会话历史
    └── saved_config.json            # 用户配置
```

---

## 📄 License

本项目采用 **MIT License**，可自由使用、修改和分发，商用无忧。

---

## Contact

**微信：** NLP技术交流群。

<img src="https://github.com/TW-NLP/ChineseErrorCorrector/blob/main/images/wechat.jpg" width="200" />

---

## 致谢

本项目基于以下优秀的开源项目构建：

- [LangChain / LangGraph](https://github.com/langchain-ai/langchain) — Agent 开发框架
- [FastAPI](https://github.com/tiangolo/fastapi) — 高性能 Web 框架
- [Milvus](https://github.com/milvus-io/milvus) — 向量数据库

---

<p align="center">
  如果这个项目对你有帮助，欢迎 Star ⭐
  <br><br>
  <a href="https://github.com/TW-NLP/DeepAgentForce">
    <img src="https://img.shields.io/github/stars/TW-NLP/DeepAgentForce?style=social" alt="GitHub Stars"/>
  </a>
</p>
