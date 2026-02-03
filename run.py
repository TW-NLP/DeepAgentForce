"""
项目启动脚本
"""
import sys
import io
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "langchain",
        "tavily",
        "firecrawl",
        "pydantic_settings"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error("❌ 缺少以下依赖包:")
        for pkg in missing_packages:
            logger.error(f"  - {pkg}")
        logger.error("\n请运行: pip install -r requirements.txt")
        return False
    
    return True



def start_server():
    """启动服务器"""
    import uvicorn
    from config.settings import settings
    
    logger.info("=" * 70)
    logger.info("🚀 启动智能搜索助手服务")
    logger.info("=" * 70)
    logger.info(f"\n📋 配置信息:")
    logger.info(f"  - 应用名称: {settings.APP_NAME}")
    logger.info(f"  - 版本: {settings.APP_VERSION}")
    logger.info(f"  - LLM 模型: {settings.LLM_MODEL}")
    logger.info("\n" + "=" * 70 + "\n")
    
    try:
        uvicorn.run(
            "src.api.main:app",
            host=settings.HOST,
            port=settings.PORT,
            log_level=settings.LOG_LEVEL.lower(),
            reload=settings.DEBUG
        )
    except KeyboardInterrupt:
        logger.info("\n👋 服务已停止")
    except Exception as e:
        # exception() 会自动记录完整的堆栈跟踪信息
        logger.exception("❌ 服务启动失败")
        sys.exit(1)


def main():
    """主函数"""
    logger.info("🔧 初始化检查...")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    
    
    # 启动服务
    start_server()


if __name__ == "__main__":
    main()