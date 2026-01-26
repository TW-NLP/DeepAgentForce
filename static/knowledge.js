/**
 * 知识库管理 JavaScript
 * 文件: knowledge.js
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
        const response = await fetch(`${API_BASE_URL}/graphrag/index/status`);
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('statDocs').textContent = data.total_documents;
            document.getElementById('statEntities').textContent = data.total_entities;
            document.getElementById('statRelationships').textContent = data.total_relationships;
            document.getElementById('statCommunities').textContent = data.total_communities;
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE_URL}/graphrag/documents`);
        const data = await response.json();
        
        const documentsList = document.getElementById('documentsList');
        
        if (data.success && data.documents.length > 0) {
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
    formData.append('auto_rebuild', 'false'); // 批量上传时先不重建索引
    
    try {
        const response = await fetch(`${API_BASE_URL}/graphrag/documents/upload`, {
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
    if (!confirm('确定要删除这个文档吗？删除后需要重建索引。')) {
        return;
    }
    
    try {
        const response = await fetch(
            `${API_BASE_URL}/graphrag/documents/${documentId}?auto_rebuild=false`,
            { method: 'DELETE' }
        );
        
        const data = await response.json();
        
        if (data.success) {
            showToast('文档已删除');
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

async function rebuildIndex() {
    const button = document.getElementById('rebuildButton');
    button.disabled = true;
    button.textContent = '🔄 重建中...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/graphrag/index/rebuild`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('✅ 索引重建任务已提交，请稍候...');
            
            // 等待一段时间后刷新状态
            setTimeout(async () => {
                await loadKnowledgeBaseStats();
                button.disabled = false;
                button.textContent = '🔄 重建索引';
            }, 3000);
        } else {
            throw new Error(data.message || '重建失败');
        }
    } catch (error) {
        console.error('重建索引失败:', error);
        showToast('重建索引失败: ' + error.message, 'error');
        button.disabled = false;
        button.textContent = '🔄 重建索引';
    }
}

// ============ 文件上传处理 ============

const uploadSection = document.getElementById('uploadSection');
const fileInput = document.getElementById('fileInput');

// 点击上传
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
        showToast(`✅ 成功上传 ${successCount} 个文件，建议重建索引`);
    }
});

// 拖拽上传
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
        showToast(`✅ 成功上传 ${successCount} 个文件，建议重建索引`);
    }
});

// 重建索引按钮
document.getElementById('rebuildButton').addEventListener('click', rebuildIndex);

// ============ 初始化 ============

// 页面加载时，如果在知识库页面，加载数据
if (document.getElementById('knowledgePage').classList.contains('active')) {
    loadKnowledgeBaseStats();
    loadDocuments();
}