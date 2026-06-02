"""子进程沙箱：在受限子进程中导入/执行用户上传的自定义 Python 工具。

主进程**永不导入**用户代码。所有「导入并抽取工具」(introspect) 和「调用某个工具」
(invoke) 都通过 :class:`SandboxRunner` 派生一个受限子进程完成，主进程只持有转发用的
代理工具 (proxy tool)。这样即便用户代码恶意/出错，也被 CPU/内存/文件大小/墙钟超时
等 rlimit 限制圈住，且拿不到主进程的密钥环境变量。
"""

from .runner import SandboxRunner, SandboxError

__all__ = ["SandboxRunner", "SandboxError"]
