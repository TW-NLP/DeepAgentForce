/**
 * 知识库管理 JavaScript (优化版)
 * 适配后端 routes.py:
 * 1. 移除 auto_rebuild 参数
 * 2. 移除重建索引功能 (后端未提供)
 * 3. 修正统计信息字段映射 (仅保留文档数)
 */

const API_BASE_URL = 'http://localhost:8000/api';

// ============ 工具函数 ============

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${type === 'success' ? '✅' : '❌'}</span>
        <span class="toast-message">${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕',
        'docx': '📘',
        'doc': '📘',
        'txt': '📄',
        'md': '📝',
        'markdown': '📝',
        'csv': '📊'
    };
    return icons[ext] || '📄';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function formatDate(dateString) {
    if (!dateString) return '未知时间';
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;
    
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (minutes < 1) return '刚刚';
    if (minutes < 60) return `${minutes} 分钟前`;
    if (hours < 24) return `${hours} 小时前`;
    if (days < 7) return `${days} 天前`;
    
    return date.toLocaleDateString('zh-CN');
}

// ============ API 调用 ============

async function loadKnowledgeBaseStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/rag/index/status`);
        const data = await response.json();
        
        if (data.success) {
            // 后端 IndexStatusResponse 仅返回 document_count
            const docCountEl = document.getElementById('statDocs');
            if (docCountEl) docCountEl.textContent = data.document_count;

            // 如果页面上还有实体/关系/社区的统计元素，建议隐藏或设为 "-"
            ['statEntities', 'statRelationships', 'statCommunities'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.textContent = '-'; // 或者 el.parentElement.style.display = 'none';
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
        
        // 后端返回 ListDocumentsResponse: { success, total, documents: [...] }
        if (data.success && data.documents && data.documents.length > 0) {
            documentsList.innerHTML = data.documents.map(doc => `
                <div class="document-item" data-id="${doc.document_id}">
                    <div class="doc-icon">${getFileIcon(doc.name)}</div>
                    <div class="doc-info">
                        <div class="doc-name" title="${doc.name}">${doc.name}</div>
                        <div class="doc-meta">
                            <span>📦 ${doc.chunks} 个文本块</span>
                            <span>🕐 ${formatDate(doc.uploaded_at)}</span>
                        </div>
                    </div>
                    <div class="doc-actions">
                        <button class="doc-action-btn" onclick="deleteDocument('${doc.document_id}')" title="删除">
                            🗑️
                        </button>
                    </div>
                </div>
            `).join('');
        } else {
            documentsList.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">📭</div>
                    <div class="empty-text">暂无文档，请上传文档到知识库</div>
                </div>
            `;
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
        showToast('加载文档列表失败', 'error');
    }
}

async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    // 注意：后端 routes.py 不再接受 auto_rebuild 参数，已移除
    
    try {
        const response = await fetch(`${API_BASE_URL}/rag/documents/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast(`✅ ${file.name} 上传成功`);
            return true;
        } else {
            throw new Error(data.message || '上传失败');
        }
    } catch (error) {
        console.error('上传失败:', error);
        showToast(`❌ ${file.name} 上传失败: ${error.message}`, 'error');
        return false;
    }
}

async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) {
        return;
    }
    
    try {
        // 注意：后端 routes.py 不再接受 auto_rebuild 参数，已移除
        const response = await fetch(
            `${API_BASE_URL}/rag/documents/${documentId}`,
            { method: 'DELETE' }
        );
        
        const data = await response.json();
        
        if (data.success) {
            showToast('文档已删除');
            // 删除后重新加载列表和统计
            await loadDocuments();
            await loadKnowledgeBaseStats();
        } else {
            throw new Error(data.message || '删除失败');
        }
    } catch (error) {
        console.error('删除文档失败:', error);
        showToast('删除文档失败: ' + error.message, 'error');
    }
}

// 注意：routes.py 中没有 '/rag/index/rebuild' 接口。
// 如果确实需要重建索引功能，需要在后端添加相应接口。
// 此处已移除 rebuildIndex 函数及其绑定。

// ============ 文件上传处理 ============

const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');
const rebuildButton = document.getElementById('rebuildButton');

// 如果页面上还有重建按钮，建议禁用或隐藏
if (rebuildButton) {
    rebuildButton.style.display = 'none'; // 后端无此接口，隐藏按钮
}

// 点击上传
if (fileInput) {
    fileInput.addEventListener('change', async (e) => {
        const files = Array.from(e.target.files);
        
        if (files.length === 0) return;
        
        showToast(`开始上传 ${files.length} 个文件...`);
        
        let successCount = 0;
        
        for (const file of files) {
            const success = await uploadDocument(file);
            if (success) successCount++;
        }
        
        // 清空文件选择
        fileInput.value = '';
        
        // 刷新列表
        await loadDocuments();
        await loadKnowledgeBaseStats();
        
        if (successCount > 0) {
            showToast(`✅ 成功上传 ${successCount} 个文件`);
        }
    });
}

// 拖拽上传
if (uploadSection) {
    uploadSection.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadSection.classList.add('dragging');
    });

    uploadSection.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadSection.classList.remove('dragging');
    });

    uploadSection.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadSection.classList.remove('dragging');
        
        const files = Array.from(e.dataTransfer.files);
        
        // 过滤支持的文件类型
        const supportedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'];
        const validFiles = files.filter(file => {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            return supportedExtensions.includes(ext);
        });
        
        if (validFiles.length === 0) {
            showToast('没有支持的文件格式', 'error');
            return;
        }
        
        showToast(`开始上传 ${validFiles.length} 个文件...`);
        
        let successCount = 0;
        
        for (const file of validFiles) {
            const success = await uploadDocument(file);
            if (success) successCount++;
        }
        
        // 刷新列表
        await loadDocuments();
        await loadKnowledgeBaseStats();
        
        if (successCount > 0) {
            showToast(`✅ 成功上传 ${successCount} 个文件`);
        }
    });
}

// ============ 初始化 ============

// 页面加载时，如果在知识库页面，加载数据
const knowledgePage = document.getElementById('knowledgePage');
if (knowledgePage && knowledgePage.classList.contains('active')) {
    loadKnowledgeBaseStats();
    loadDocuments();
}