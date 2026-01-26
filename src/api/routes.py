"""
REST API 路由 - 完整版
包含：对话接口 + GraphRAG 知识库接口 + 配置管理接口

路径: src/api/routes.py
"""

import logging
import uuid
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import json
from config.settings import settings
from src.workflow.agent import ConversationalAgent

# GraphRAG 导入
try:
    from src.services.rag_graph import GraphRAGPipeline
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG 模块未找到，相关功能将不可用")

logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 配置文件路径
CONFIG_FILE = Path("config/saved_config.json")
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = Path("config/saved_history.json")


class SavedHistoryResponse(BaseModel):
    success: bool
    history: List[Dict[str, Any]]

# ==================== 数据模型（对话） ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    message: str
    session_id: str
    timestamp: str


class HistoryResponse(BaseModel):
    """历史记录响应"""
    history: List[Dict[str, str]]
    session_id: str


class StatusResponse(BaseModel):
    """状态响应"""
    status: str
    message: str


# ==================== 数据模型（GraphRAG） ====================

class DocumentMetadata(BaseModel):
    """文档元数据"""
    title: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class UploadResponse(BaseModel):
    """上传响应"""
    success: bool
    message: str
    document_id: str
    document_name: str
    chunks_count: int
    uploaded_at: str


class DeleteResponse(BaseModel):
    """删除响应"""
    success: bool
    message: str
    document_id: str


class QueryRequest(BaseModel):
    """查询请求"""
    question: str
    top_k_communities: Optional[int] = 10


class QueryResponse(BaseModel):
    """查询响应"""
    success: bool
    question: str
    answer: str
    processing_time: float


class DocumentInfo(BaseModel):
    """文档信息"""
    document_id: str
    name: str
    path: str
    chunks: int
    uploaded_at: str
    metadata: Dict


class ListDocumentsResponse(BaseModel):
    """文档列表响应"""
    success: bool
    total: int
    documents: List[DocumentInfo]


class IndexStatusResponse(BaseModel):
    """索引状态响应"""
    success: bool
    is_indexed: bool
    total_documents: int
    total_entities: int
    total_relationships: int
    total_communities: int
    message: str


# ==================== 数据模型（配置） ====================

class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool
    message: str
    config: Optional[Dict[str, str]] = None


# ==================== Session 管理器 ====================

class SessionManager:
    """Session 管理器"""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationalAgent] = {}
        self.session_timestamps: Dict[str, datetime] = {}
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, ConversationalAgent]:
        """获取或创建 Session"""
        if session_id and session_id in self.sessions:
            self.session_timestamps[session_id] = datetime.now()
            return session_id, self.sessions[session_id]
        
        new_session_id = str(uuid.uuid4())
        self.sessions[new_session_id] = ConversationalAgent()
        self.session_timestamps[new_session_id] = datetime.now()
        
        logger.info(f"创建新会话: {new_session_id}")
        return new_session_id, self.sessions[new_session_id]
    
    def clear_session(self, session_id: str):
        """清空 Session 历史"""
        if session_id in self.sessions:
            self.sessions[session_id].clear_history()
            logger.info(f"清空会话历史: {session_id}")
    
    def delete_session(self, session_id: str):
        """删除 Session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            del self.session_timestamps[session_id]
            logger.info(f"删除会话: {session_id}")
    
    def cleanup_old_sessions(self, timeout_seconds: int = 3600):
        """清理过期 Session"""
        now = datetime.now()
        expired_sessions = [
            sid for sid, ts in self.session_timestamps.items()
            if (now - ts).total_seconds() > timeout_seconds
        ]
        
        for sid in expired_sessions:
            self.delete_session(sid)
        
        if expired_sessions:
            logger.info(f"清理了 {len(expired_sessions)} 个过期会话")
    
    def get_session_count(self) -> int:
        """获取当前 Session 数量"""
        return len(self.sessions)


# ==================== GraphRAG 管理器 ====================

class GraphRAGManager:
    """GraphRAG 知识库管理器"""
    
    def __init__(self):
        self.pipeline: Optional[GraphRAGPipeline] = None
        self.initialized = False
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        if GRAPHRAG_AVAILABLE:
            self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """初始化 GraphRAG Pipeline"""
        try:
            self.pipeline = GraphRAGPipeline(
                llm_api_key=getattr(settings, 'LLM_API_KEY'),
                embedding_api_key=getattr(settings, 'EMBEDDING_API_KEY'),
                llm_url=getattr(settings, 'LLM_URL'),
                embedding_url=getattr(settings, 'EMBEDDING_URL'),
                embedding_name=getattr(settings, 'EMBEDDING_MODEL'),
                embedding_dim=getattr(settings, 'EMBEDDING_DIM', 1024),
                llm_name=getattr(settings, 'LLM_MODEL'),
                storage_dir=getattr(settings, 'GRAPHRAG_STORAGE_DIR', './graphrag_storage')
            )
            
            try:
                self.pipeline.load("default")
                logger.info("✅ GraphRAG: 加载已有知识库")
            except FileNotFoundError:
                logger.info("📝 GraphRAG: 创建新知识库")
            
            self.initialized = True
            logger.info("✅ GraphRAG Pipeline 初始化完成")
            
        except Exception as e:
            logger.error(f"❌ GraphRAG 初始化失败: {e}", exc_info=True)
            self.initialized = False
    
    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self.initialized and self.pipeline is not None
    
    def save_upload_file(self, upload_file: UploadFile) -> Path:
        """保存上传文件"""
        file_path = self.upload_dir / f"{uuid.uuid4()}_{upload_file.filename}"
        
        try:
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            return file_path
        finally:
            upload_file.file.close()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.is_ready():
            return {
                'enabled': False,
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'total_communities': 0,
                'index_status': 'Not initialized'
            }
        
        total_communities = sum(len(comms) for comms in self.pipeline.communities.values())
        is_indexed = self.pipeline.community_summary_index is not None
        
        return {
            'enabled': True,
            'total_documents': len(self.pipeline.documents),
            'total_entities': len(self.pipeline.entities),
            'total_relationships': len(self.pipeline.relationships),
            'total_communities': total_communities,
            'index_status': 'Indexed' if is_indexed else 'Not indexed'
        }
    
    def save(self):
        """保存知识库"""
        if self.is_ready():
            try:
                self.pipeline.save("default")
                logger.info("✅ GraphRAG 知识库已保存")
            except Exception as e:
                logger.error(f"❌ 保存知识库失败: {e}")


# 全局管理器实例
session_manager = SessionManager()
graphrag_manager = GraphRAGManager()


# ==================== 辅助函数 ====================


def load_history_from_file() -> List[Dict[str, Any]]:
    if HISTORY_FILE.exists():
        try:
            with HISTORY_FILE.open('r', encoding='utf-8') as f:
                data = json.load(f)
                # 确保返回的是列表格式
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.error(f"❌ 读取历史记录失败: {e}")
            return []
    return []

def check_graphrag_ready():
    """检查 GraphRAG 是否就绪"""
    if not GRAPHRAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG 模块不可用，请检查安装"
        )
    
    if not graphrag_manager.is_ready():
        raise HTTPException(
            status_code=503,
            detail="GraphRAG 服务未就绪"
        )


async def rebuild_index_background():
    """后台任务：重建索引"""
    try:
        logger.info("📊 开始重建 GraphRAG 索引...")
        graphrag_manager.pipeline.rebuild_index()
        graphrag_manager.save()
        logger.info("✅ GraphRAG 索引重建完成")
    except Exception as e:
        logger.error(f"❌ 重建索引失败: {e}", exc_info=True)


def load_config_from_file() -> Dict[str, Any]:
    """
    从文件加载配置 (修复版：支持读取 JSON 结构)
    返回嵌套字典结构，例如: {'llm_config': {'LLM_API_KEY': '...'}, ...}
    """
    config_data = {
        "llm_config": {},
        "search_config": {},
        "firecrawl_config": {},
        "embedding_config": {}
    }
    
    # 1. 尝试从 JSON 文件加载
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open('r', encoding='utf-8') as f:
                saved_data = json.load(f)
                # 合并保存的数据
                for group, values in saved_data.items():
                    if group in config_data:
                        config_data[group].update(values)
            logger.info(f"✅ 从文件加载配置成功")
        except json.JSONDecodeError:
            logger.warning("⚠️ 配置文件格式错误，将使用默认值")
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")

    # 2. (可选) 从环境变量补充/覆盖关键配置
    # 如果你想让环境变量优先级最高，可以在这里通过 os.getenv 覆盖 config_data
    # 这里为了简单，仅当文件不存在对应值时才读取环境变量
    env_mapping = {
        'LLM_API_KEY': ('llm_config', 'LLM_API_KEY'),
        'LLM_URL': ('llm_config', 'LLM_URL'),
        'LLM_MODEL': ('llm_config', 'LLM_MODEL'),
        'TAVILY_API_KEY': ('search_config', 'TAVILY_API_KEY'),
        'FIRECRAWL_API_KEY': ('firecrawl_config', 'FIRECRAWL_API_KEY'),
        'FIRECRAWL_URL': ('firecrawl_config', 'FIRECRAWL_URL'),
        'EMBEDDING_API_KEY': ('embedding_config', 'EMBEDDING_API_KEY'),
        'EMBEDDING_URL': ('embedding_config', 'EMBEDDING_URL'),
        'EMBEDDING_MODEL': ('embedding_config', 'EMBEDDING_MODEL'),
    }

    for env_key, (group, dict_key) in env_mapping.items():
        if dict_key not in config_data[group] or not config_data[group][dict_key]:
            env_val = os.getenv(env_key)
            if env_val:
                config_data[group][dict_key] = env_val

    return config_data


def save_config_to_file(config: Dict[str, str]):
    """保存配置到文件"""
    try:
        config_dict={"llm_config":{}, "search_config":{}, "firecrawl_config":{}, "embedding_config":{}}
        
        # LLM 配置
        for key in ['LLM_API_KEY', 'LLM_URL', 'LLM_MODEL']:
            if key in config:
                config_dict['llm_config'][key]=config[key]
        
        
        # 搜索配置
        if 'TAVILY_API_KEY' in config:
            config_dict['search_config']['TAVILY_API_KEY'] = config['TAVILY_API_KEY']
        
        # Firecrawl 配置
        for key in ['FIRECRAWL_API_KEY', 'FIRECRAWL_URL']:
            if key in config:
                config_dict['firecrawl_config'][key]=config[key]


        # Embedding 配置
        for key in ['EMBEDDING_API_KEY', 'EMBEDDING_URL', 'EMBEDDING_MODEL']:
            if key in config:
                config_dict['embedding_config'][key]=config[key]
        # 保存json格式
        with CONFIG_FILE.open('w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=4)
        
        logger.info(f"✅ 配置已保存到文件: {CONFIG_FILE}")

        return config_dict
        
    except Exception as e:
        logger.error(f"❌ 保存配置文件失败: {e}")
        raise
def save_conversation_to_file(user_msg: str, ai_msg: str):
    """将对话追加保存到 JSON 文件"""
    try:
        # 1. 读取现有记录
        history_data = []
        if HISTORY_FILE.exists():
            with HISTORY_FILE.open('r', encoding='utf-8') as f:
                try:
                    history_data = json.load(f)
                    if not isinstance(history_data, list):
                        history_data = []
                except json.JSONDecodeError:
                    history_data = []

        # 2. 获取时间
        timestamp = datetime.now().isoformat()

        # 3. 追加新记录 (保存为一组对话)
        # 这里我们设计结构为：一条记录包含 question 和 answer
        new_entry = {
            "id": str(uuid.uuid4()), # 给个ID方便前端索引
            "timestamp": timestamp,
            "user_content": user_msg,
            "ai_content": ai_msg
        }
        history_data.append(new_entry)

        # 4. 写入文件
        with HISTORY_FILE.open('w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=4)
            
    except Exception as e:
        logger.error(f"❌ 保存历史记录失败: {e}")

# ==================== 对话接口 ====================
@router.get("/history/saved", response_model=SavedHistoryResponse, tags=["对话"])
async def get_saved_history():
    """获取 config/saved_history.json 中的历史记录"""
    history_data = load_history_from_file()
    return SavedHistoryResponse(
        success=True,
        history=history_data
    )



@router.post("/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest):
    """同步对话接口"""
    try:
        session_manager.cleanup_old_sessions(settings.SESSION_TIMEOUT)
        session_id, agent = session_manager.get_or_create_session(request.session_id)
        
        logger.info(f"[{session_id}] 收到消息: {request.message[:50]}...")
        
        # 1. 获取 AI 回复
        response_content = await agent.chat(request.message)
        
        # 2. ★★★ 新增：保存到 saved_history.json ★★★
        # 注意：这里我们保存的是这一轮的对话
        save_conversation_to_file(request.message, response_content)
        
        return ChatResponse(
            message=response_content,
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"对话处理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}", response_model=HistoryResponse, tags=["对话"])
async def get_history(session_id: str):
    """获取对话历史"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = session_manager.sessions[session_id]
    
    return HistoryResponse(
        history=agent.get_history(),
        session_id=session_id
    )


@router.post("/clear/{session_id}", response_model=StatusResponse, tags=["对话"])
async def clear_history(session_id: str):
    """清空对话历史"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.clear_session(session_id)
    
    return StatusResponse(
        status="ok",
        message="History cleared"
    )


@router.delete("/session/{session_id}", response_model=StatusResponse, tags=["对话"])
async def delete_session(session_id: str):
    """删除 Session"""
    if session_id not in session_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.delete_session(session_id)
    
    return StatusResponse(
        status="ok",
        message="Session deleted"
    )


# ==================== GraphRAG 接口（省略部分代码以保持简洁）====================

@router.post("/graphrag/documents/upload", response_model=UploadResponse, tags=["GraphRAG"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    auto_rebuild: bool = True
):
    """上传文档到知识库"""
    check_graphrag_ready()
    
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file_extension}"
        )
    
    try:
        file_path = graphrag_manager.save_upload_file(file)
        logger.info(f"📄 文件已保存: {file_path}")
        
        metadata = {
            "title": title or file.filename,
            "author": author,
            "category": category,
            "original_filename": file.filename,
            "file_extension": file_extension,
            "uploaded_at": datetime.now().isoformat()
        }
        
        doc_uuid = graphrag_manager.pipeline.add_document(str(file_path), metadata)
        doc_info = graphrag_manager.pipeline.documents[
            graphrag_manager.pipeline.uuid_to_docid[doc_uuid]
        ]
        
        if auto_rebuild:
            background_tasks.add_task(rebuild_index_background)
        
        return UploadResponse(
            success=True,
            message="文档上传成功" + ("，正在后台重建索引" if auto_rebuild else ""),
            document_id=doc_uuid,
            document_name=file.filename,
            chunks_count=len(doc_info['chunks']),
            uploaded_at=doc_info['added_at']
        )
        
    except Exception as e:
        logger.error(f"❌ 上传文档失败: {e}", exc_info=True)
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@router.get("/graphrag/documents", response_model=ListDocumentsResponse, tags=["GraphRAG"])
async def list_documents():
    """获取所有文档列表"""
    check_graphrag_ready()
    
    try:
        docs = graphrag_manager.pipeline.list_documents()
        
        documents = [
            DocumentInfo(
                document_id=doc['uuid'],
                name=doc['name'],
                path=doc['path'],
                chunks=doc['chunks'],
                uploaded_at=doc['added_at'],
                metadata=doc['metadata']
            )
            for doc in docs
        ]
        
        return ListDocumentsResponse(
            success=True,
            total=len(documents),
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"❌ 获取文档列表失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.post("/graphrag/index/rebuild", tags=["GraphRAG"])
async def rebuild_index(background_tasks: BackgroundTasks):
    """手动触发索引重建"""
    check_graphrag_ready()
    
    stats = graphrag_manager.get_stats()
    if stats['total_documents'] == 0:
        raise HTTPException(status_code=400, detail="没有文档可以索引")
    
    try:
        background_tasks.add_task(rebuild_index_background)
        
        return {
            "success": True,
            "message": "索引重建任务已添加到后台队列",
            "total_documents": stats['total_documents']
        }
        
    except Exception as e:
        logger.error(f"❌ 触发索引重建失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"触发索引重建失败: {str(e)}")


# ==================== 配置管理接口 ====================
class ConfigResponse(BaseModel):
    """配置响应"""
    success: bool
    message: str
    # ⚠️ 关键修改：必须是 Any 或者是 dict，否则 Pydantic 会拦截嵌套 JSON 报错
    config: Optional[Dict[str, Any]] = None 

# ==================== 辅助函数（配置） ====================

def load_config_from_file() -> Dict[str, Any]:
    """从 JSON 文件加载配置（返回嵌套结构）"""
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open('r', encoding='utf-8') as f:
                # 直接读取 JSON 结构
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
    
    # 默认空结构，防止前端报错
    return {
        "llm_config": {},
        "search_config": {},
        "firecrawl_config": {},
        "embedding_config": {}
    }

def save_config_to_file(new_flat_config: Dict[str, str]) -> Dict[str, Any]:
    """
    保存配置：
    1. 前端传过来的是扁平的 {'LLM_API_KEY': '...'}
    2. 我们要把它合并进现有的嵌套 JSON 结构 {'llm_config': {'LLM_API_KEY': ...}}
    """
    # 1. 先读取旧的完整配置
    current_config = load_config_from_file()
    
    # 2. 定义 字段 -> 组 的映射关系 (必须与前端 config.js 一致)
    field_mapping = {
        # LLM
        'LLM_API_KEY': 'llm_config', 
        'LLM_URL': 'llm_config', 
        'LLM_MODEL': 'llm_config',
        # Search
        'TAVILY_API_KEY': 'search_config',
        # Firecrawl
        'FIRECRAWL_API_KEY': 'firecrawl_config', 
        'FIRECRAWL_URL': 'firecrawl_config',
        # Embedding
        'EMBEDDING_API_KEY': 'embedding_config', 
        'EMBEDDING_URL': 'embedding_config', 
        'EMBEDDING_MODEL': 'embedding_config'
    }

    # 3. 将扁平的新值更新到嵌套结构中
    for key, value in new_flat_config.items():
        # 跳过空值
        if not value: 
            continue
            
        # 跳过没变的脱敏数据 (如果前端传回，说明用户没改，不要存进去)
        if "..." in value and len(value) < 20:
            continue

        group = field_mapping.get(key)
        if group:
            if group not in current_config:
                current_config[group] = {}
            # 更新值
            current_config[group][key] = value

    # 4. 写入文件
    try:
        with CONFIG_FILE.open('w', encoding='utf-8') as f:
            json.dump(current_config, f, ensure_ascii=False, indent=4)
        logger.info(f"✅ 配置已保存到文件: {CONFIG_FILE}")
    except Exception as e:
        logger.error(f"❌ 保存配置文件失败: {e}")
        raise

    return current_config

# ==================== 配置管理接口 ====================

@router.get("/config", response_model=ConfigResponse, tags=["配置管理"])
async def get_config():
    """获取当前配置（返回嵌套结构 + 脱敏）"""
    try:
        # 1. 获取嵌套字典
        config_data = load_config_from_file()
        
        # 2. 深拷贝用于脱敏，不修改原数据
        safe_config = json.loads(json.dumps(config_data))
        
        # 3. 遍历嵌套结构进行脱敏
        for group, items in safe_config.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    # 如果包含 KEY 且长度较长，则脱敏
                    if value and isinstance(value, str) and 'KEY' in key.upper() and len(value) > 8:
                        items[key] = f"{value[:4]}...{value[-4:]}"
        
        # 4. 返回
        return ConfigResponse(
            success=True,
            message="配置获取成功",
            config=safe_config  # 这里是嵌套字典，ConfigResponse 现在可以接收了
        )
        
    except Exception as e:
        logger.error(f"❌ 获取配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取配置失败: {str(e)}")


@router.post("/config", response_model=ConfigResponse, tags=["配置管理"])
async def update_config(config: Dict[str, str]):
    """更新配置（接收扁平结构 -> 保存为嵌套结构）"""
    try:
        # 1. 保存配置 (会处理扁平转嵌套逻辑)
        # 注意：这里 config 是前端发来的扁平字典，save_config_to_file 会处理它
        updated_nested_config = save_config_to_file(config)
        
        # 2. 更新内存中的 settings (需要展平更新，或者让 settings 支持读取 dict)
        # 简单起见，我们把嵌套配置展平后更新给 settings
        flat_settings = {}
        for group in updated_nested_config.values():
            if isinstance(group, dict):
                flat_settings.update(group)
        settings.update(**flat_settings)
        
        logger.info(f"✅ 配置已更新并应用")
        
        return ConfigResponse(
            success=True,
            message="配置保存成功",
            config=None # 保存成功不需要返回 config
        )
        
    except Exception as e:
        logger.error(f"❌ 更新配置失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


# ==================== 服务信息接口 ====================

@router.get("/info", tags=["服务信息"])
async def get_info():
    """获取服务信息"""
    graphrag_stats = graphrag_manager.get_stats()
    
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "features": {
            "streaming": True,
            "graphrag": graphrag_stats['enabled'],
            "config_management": True
        },
        "endpoints": {
            "http": {
                "chat": "/api/chat",
                "config_get": "/api/config",
                "config_update": "/api/config"
            }
        },
        "graphrag_status": graphrag_stats
    }


@router.get("/stats", tags=["服务信息"])
async def get_stats():
    """获取服务统计信息"""
    graphrag_stats = graphrag_manager.get_stats()
    
    return {
        "active_sessions": session_manager.get_session_count(),
        "graphrag": graphrag_stats,
        "config_loaded": CONFIG_FILE.exists(),
        "timestamp": datetime.now().isoformat()
    }



@router.get("/graphrag/index/status", response_model=IndexStatusResponse, tags=["GraphRAG"])
async def get_index_status():
    """获取索引状态"""
    check_graphrag_ready()
    
    try:
        stats = graphrag_manager.get_stats()
        is_indexed = stats['index_status'] == 'Indexed'
        
        return IndexStatusResponse(
            success=True,
            is_indexed=is_indexed,
            total_documents=stats['total_documents'],
            total_entities=stats['total_entities'],
            total_relationships=stats['total_relationships'],
            total_communities=stats['total_communities'],
            message=stats['index_status']
        )
        
    except Exception as e:
        logger.error(f"❌ 获取索引状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取索引状态失败: {str(e)}")
@router.post("/graphrag/query", response_model=QueryResponse, tags=["GraphRAG"])
async def query_knowledge_base(request: QueryRequest):
    """
    查询知识库
    
    - **question**: 要查询的问题
    - **top_k_communities**: 检索的社区数量（默认 10）
    """
    check_graphrag_ready()
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    try:
        start_time = datetime.now()
        
        # 执行查询
        answer = graphrag_manager.pipeline.global_query(
            request.question,
            top_k_communities=request.top_k_communities
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ 查询完成: {request.question[:50]}... (耗时 {processing_time:.2f}s)")
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=answer,
            processing_time=processing_time
        )
        
    except RuntimeError as e:
        if "索引未构建" in str(e):
            raise HTTPException(
                status_code=400,
                detail="索引未构建，请先上传文档并等待索引构建完成"
            )
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")