/**
 * 知识库管理 JavaScript (UX 优化版)
 */

// 自动获取当前服务器的 API 地址 (使用全局函数)
const API_BASE_URL = window.getApiBase ? window.getApiBase() : `${window.location.protocol}//${window.location.hostname}:8000/api`;

// ============ UI 交互组件: Loading 管理 ============
const LoadingManager = {
    overlay: null,
    textElement: null,

    init() {
        if (this.overlay) return;
        
        // 动态创建 DOM，无需手动写在 HTML 里
        const div = document.createElement('div');
        div.className = 'loading-overlay';
        div.innerHTML = `
            <div class="loading-spinner"></div>
            <div class="loading-text">正在处理...</div>
        `;
        document.body.appendChild(div);
        
        this.overlay = div;
        this.textElement = div.querySelector('.loading-text');
    },

    show(message = '正在加载...') {
        this.init();
        this.textElement.textContent = message;
        this.overlay.classList.add('active');
    },

    updateText(message) {
        if (this.textElement) {
            this.textElement.textContent = message;
        }
    },

    hide() {
        if (this.overlay) {
            this.overlay.classList.remove('active');
        }
    }
};

// ============ 工具函数 (保持不变) ============

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`; // 请确保CSS里有 toast 的样式
    toast.innerHTML = `
        <span class="toast-icon">${type === 'success' ? '✅' : '❌'}</span>
        <span class="toast-message">${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse'; // 假设你有 slideIn 动画
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕', 'docx': '📘', 'doc': '📘',
        'txt': '📄', 'md': '📝', 'markdown': '📝', 'csv': '📊'
    };
    return icons[ext] || '📄';
}

function formatDate(dateString) {
    if (!dateString) return '未知时间';
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return `${minutes} 分钟前`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} 小时前`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days} 天前`;
    
    return date.toLocaleDateString('zh-CN');
}

// ============ API 调用 (文档列表与删除) ============

async function loadKnowledgeBaseStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/rag/index/status`);
        const data = await response.json();
        
        if (data.success) {
            const docCountEl = document.getElementById('statDocs');
            if (docCountEl) docCountEl.textContent = data.document_count;
            
            // 隐藏无关统计
            ['statEntities', 'statRelationships', 'statCommunities'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = '-';
            });
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/rag/documents`);
        const data = await response.json();
        const documentsList = document.getElementById('documentsList');
        
        if (data.success && data.documents && data.documents.length > 0) {
            documentsList.innerHTML = data.documents.map(doc => `
                <div class="document-item">
                    <div class="doc-icon">${getFileIcon(doc.name)}</div>
                    <div class="doc-info">
                        <div class="doc-name" title="${doc.name}">${doc.name}</div>
                        <div class="doc-meta">
                            <span>📦 ${doc.chunks} 块</span>
                            <span>🕐 ${formatDate(doc.uploaded_at)}</span>
                        </div>
                    </div>
                    <div class="doc-actions">
                        <button class="doc-action-btn" onclick="deleteDocument('${doc.document_id}')" title="删除">🗑️</button>
                    </div>
                </div>
            `).join('');
        } else {
            documentsList.innerHTML = `<div class="empty-state">📭 暂无文档</div>`;
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
    }
}

async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) return;
    
    // 删除操作通常很快，可以用轻量级Loading或不阻塞
    LoadingManager.show('正在删除...');
    try {
        const response = await fetch(`${API_BASE_URL}/rag/documents/${documentId}`, { method: 'DELETE' });
        const data = await response.json();
        
        if (data.success) {
            showToast('文档已删除');
            await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);
        } else {
            throw new Error(data.message || '删除失败');
        }
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        LoadingManager.hide();
    }
}

// ============ 核心优化：批量上传处理 ============

/**
 * 统一处理单个文件上传请求
 * @returns {Promise<boolean>} 是否成功
 */
async function uploadSingleFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${API_BASE_URL}/rag/documents/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!data.success) throw new Error(data.message);
        return true;
    } catch (error) {
        console.error(`上传 ${file.name} 失败:`, error);
        return false;
    }
}

/**
 * 批量上传逻辑控制器
 * 负责 UI 状态更新、循环上传和结果汇总
 */
async function handleBatchUpload(files) {
    if (files.length === 0) return;

    // 1. 开启 Loading，阻止用户操作
    LoadingManager.show('准备上传...');
    
    let successCount = 0;
    const total = files.length;
    const failedFiles = [];

    // 2. 循环上传
    for (let i = 0; i < total; i++) {
        const file = files[i];
        // 实时更新 Loading 文字，告知用户当前进度
        LoadingManager.updateText(`正在上传 (${i + 1}/${total}): ${file.name}`);
        
        // 执行上传
        const success = await uploadSingleFile(file);
        
        if (success) {
            successCount++;
        } else {
            failedFiles.push(file.name);
        }
    }

    // 3. 上传完成，刷新数据
    LoadingManager.updateText('正在刷新列表...');
    await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);

    // 4. 关闭 Loading
    LoadingManager.hide();

    // 5. 显示最终结果 Toast
    if (failedFiles.length === 0) {
        showToast(`✅ 全部成功！共上传 ${successCount} 个文件`);
    } else if (successCount === 0) {
        showToast(`❌ 全部失败，请检查网络或文件格式`, 'error');
    } else {
        showToast(`⚠️ 上传完成: ${successCount} 个成功，${failedFiles.length} 个失败`, 'warning');
    }
}

// ============ 事件监听 ============

const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');

// 点击上传
if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleBatchUpload(files);
        fileInput.value = ''; // 清空选择，允许重复选同一个文件
    });
}

// 拖拽上传
if (uploadSection) {
    ['dragover', 'dragleave'].forEach(eventName => {
        uploadSection.addEventListener(eventName, (e) => {
            e.preventDefault();
            if (eventName === 'dragover') uploadSection.classList.add('dragging');
            else uploadSection.classList.remove('dragging');
        });
    });

    uploadSection.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadSection.classList.remove('dragging');
        
        const files = Array.from(e.dataTransfer.files);
        const supportedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'];
        
        const validFiles = files.filter(file => {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            return supportedExtensions.includes(ext);
        });
        
        if (validFiles.length !== files.length) {
            showToast(`已过滤 ${files.length - validFiles.length} 个不支持的文件格式`, 'warning');
        }

        if (validFiles.length > 0) {
            handleBatchUpload(validFiles);
        }
    });
}

// ============ 初始化 ============
// 页面加载完成后初始化 Loading 组件（确保 DOM 存在）
document.addEventListener('DOMContentLoaded', () => {
    LoadingManager.init();
    
    // 如果在知识库页面，加载数据
    const knowledgePage = document.getElementById('knowledgePage');
    if (knowledgePage && knowledgePage.classList.contains('active')) {
        loadKnowledgeBaseStats();
        loadDocuments();
    }
});