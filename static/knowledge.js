/**
 * 知识库管理 JavaScript - 增强版
 */

// ============ 全局状态 ============
let allDocuments = [];
let selectedDocuments = new Set();
let searchKeyword = '';
let currentFilter = 'all';

// ============ UI 组件 ============
const LoadingManager = {
    overlay: null,
    textElement: null,

    init() {
        if (this.overlay) return;
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

// ============ 工具函数 ============
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕', 'docx': '📘', 'doc': '📘',
        'txt': '📄', 'md': '📝', 'markdown': '📝', 'csv': '📊'
    };
    return icons[ext] || '📄';
}

function getFileTypeLabel(ext) {
    const labels = {
        'pdf': 'PDF文档', 'docx': 'Word文档', 'doc': 'Word文档',
        'txt': '文本文件', 'md': 'Markdown', 'csv': 'CSV表格'
    };
    return labels[ext] || '未知格式';
}

function formatDate(dateString) {
    if (!dateString) return '未知时间';
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============ API 调用 ============
async function loadKnowledgeBaseStats() {
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/index/status`, { headers });
        const data = await response.json();

        if (data.success) {
            document.getElementById('statDocs').textContent = data.document_count;
            document.getElementById('statChunks').textContent = data.chunks_count || 0;
            document.getElementById('statSize').textContent = data.total_size || '0 KB';
            document.getElementById('statLastUpdate').textContent = data.last_updated ? formatDate(data.last_updated) : '暂无';
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

async function loadDocuments() {
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents`, { headers });
        const data = await response.json();

        if (data.success) {
            allDocuments = data.documents || [];
            renderDocuments();
        }
    } catch (error) {
        console.error('加载文档列表失败:', error);
    }
}

function renderDocuments() {
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');

    // 过滤文档
    let filteredDocs = allDocuments;
    if (searchKeyword) {
        filteredDocs = filteredDocs.filter(doc =>
            doc.name.toLowerCase().includes(searchKeyword.toLowerCase())
        );
    }

    // 更新选中计数
    const selectedCount = selectedDocuments.size;
    const selectAllBtn = document.getElementById('selectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.textContent = selectedCount > 0 ? `已选 ${selectedCount} 项` : '全选';
    }

    if (filteredDocs.length === 0) {
        documentsList.innerHTML = '';
        emptyState.style.display = 'flex';
        return;
    }

    emptyState.style.display = 'none';

    documentsList.innerHTML = filteredDocs.map(doc => `
        <div class="document-item ${selectedDocuments.has(doc.document_id) ? 'selected' : ''}" data-id="${doc.document_id}">
            <div class="doc-checkbox" onclick="toggleDocumentSelect('${doc.document_id}', event)">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    ${selectedDocuments.has(doc.document_id)
                        ? '<polyline points="20 6 9 17 4 12"/>'
                        : '<rect x="3" y="3" width="18" height="18" rx="2"/>'}
                </svg>
            </div>
            <div class="doc-icon">${getFileIcon(doc.name)}</div>
            <div class="doc-info" onclick="showDocumentDetail('${doc.document_id}')">
                <div class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</div>
                <div class="doc-meta">
                    <span class="doc-type">${getFileTypeLabel(doc.name.split('.').pop())}</span>
                    <span class="doc-chunks">📦 ${doc.chunks || 0} 块</span>
                    <span class="doc-date">🕐 ${formatDate(doc.uploaded_at)}</span>
                </div>
            </div>
            <div class="doc-actions">
                <button class="doc-action-btn" onclick="previewDocument('${doc.document_id}')" title="预览">👁️</button>
                <button class="doc-action-btn" onclick="deleteDocument('${doc.document_id}')" title="删除">🗑️</button>
            </div>
        </div>
    `).join('');
}

function toggleDocumentSelect(docId, event) {
    event.stopPropagation();
    if (selectedDocuments.has(docId)) {
        selectedDocuments.delete(docId);
    } else {
        selectedDocuments.add(docId);
    }
    renderDocuments();
}

function selectAllDocuments() {
    if (selectedDocuments.size === allDocuments.length) {
        selectedDocuments.clear();
    } else {
        allDocuments.forEach(doc => selectedDocuments.add(doc.document_id));
    }
    renderDocuments();
}

function batchDeleteDocuments() {
    if (selectedDocuments.size === 0) {
        showToast('请先选择要删除的文档', 'error');
        return;
    }

    if (!confirm(`确定要删除选中的 ${selectedDocuments.size} 个文档吗？`)) return;

    LoadingManager.show('正在批量删除...');

    const deletePromises = Array.from(selectedDocuments).map(docId =>
        fetch(`${getApiUrl()}/rag/documents/${docId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
        }).then(res => res.json())
    );

    Promise.all(deletePromises).then(results => {
        const successCount = results.filter(r => r.success).length;
        selectedDocuments.clear();
        loadDocuments();
        loadKnowledgeBaseStats();
        showToast(`成功删除 ${successCount} 个文档`);
    }).catch(err => {
        showToast('删除失败: ' + err.message, 'error');
    }).finally(() => {
        LoadingManager.hide();
    });
}

async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) return;

    LoadingManager.show('正在删除...');
    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents/${documentId}`, {
            method: 'DELETE',
            headers
        });
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

function showDocumentDetail(docId) {
    const doc = allDocuments.find(d => d.document_id === docId);
    if (!doc) return;

    const modal = createModal('文档详情', `
        <div class="detail-modal">
            <div class="detail-header">
                <div class="detail-icon">${getFileIcon(doc.name)}</div>
                <div class="detail-title">
                    <h3>${escapeHtml(doc.name)}</h3>
                    <span class="detail-type">${getFileTypeLabel(doc.name.split('.').pop())}</span>
                </div>
            </div>
            <div class="detail-info">
                <div class="detail-row">
                    <span class="detail-label">文档ID</span>
                    <span class="detail-value">${doc.document_id}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">文件大小</span>
                    <span class="detail-value">${formatFileSize(doc.size || 0)}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">文本块数</span>
                    <span class="detail-value">${doc.chunks || 0}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">上传时间</span>
                    <span class="detail-value">${formatDate(doc.uploaded_at)}</span>
                </div>
                ${doc.metadata ? `
                <div class="detail-row">
                    <span class="detail-label">元数据</span>
                    <span class="detail-value">${JSON.stringify(doc.metadata, null, 2)}</span>
                </div>
                ` : ''}
            </div>
            <div class="detail-actions">
                <button class="btn btn-primary" onclick="previewDocument('${docId}'); closeModal();">
                    👁️ 预览内容
                </button>
                <button class="btn btn-danger" onclick="deleteDocument('${docId}'); closeModal();">
                    🗑️ 删除文档
                </button>
            </div>
        </div>
    `);
    document.body.appendChild(modal);
}

async function previewDocument(docId) {
    LoadingManager.show('正在加载预览...');
    try {
        const token = localStorage.getItem('access_token');
        const response = await fetch(`${getApiUrl()}/rag/documents/${docId}/content`, {
            headers: token ? { 'Authorization': `Bearer ${token}` } : {}
        });

        if (!response.ok) throw new Error('无法加载预览');

        const data = await response.json();
        const doc = allDocuments.find(d => d.document_id === docId);

        const modal = createModal(`预览: ${doc?.name || '文档'}`, `
            <div class="preview-content">
                <div class="preview-header">
                    <span class="preview-type">${getFileTypeLabel(doc?.name.split('.').pop() || '')}</span>
                    <span class="preview-chunks">${data.chunks?.length || 0} 个文本块</span>
                </div>
                <div class="preview-body">
                    ${(data.chunks || []).map((chunk, i) => `
                        <div class="preview-chunk">
                            <div class="chunk-header">块 ${i + 1}</div>
                            <div class="chunk-content">${escapeHtml(chunk.content || chunk.text || '')}</div>
                        </div>
                    `).join('')}
                    ${(!data.chunks || data.chunks.length === 0) ? '<div class="empty-preview">暂无内容</div>' : ''}
                </div>
            </div>
        `);
        document.body.appendChild(modal);
    } catch (error) {
        showToast('预览加载失败: ' + error.message, 'error');
    } finally {
        LoadingManager.hide();
    }
}

function createModal(title, content) {
    // 移除已有的模态框
    const existingModal = document.querySelector('.modal-overlay');
    if (existingModal) existingModal.remove();

    const modal = document.createElement('div');
    modal.className = 'modal-overlay active';
    modal.innerHTML = `
        <div class="modal detail-modal-container">
            <div class="modal-header">
                <h2>${title}</h2>
                <button class="modal-close" onclick="closeModal()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
            <div class="modal-body">${content}</div>
        </div>
    `;
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    window.currentModal = modal;
    return modal;
}

function closeModal() {
    const modal = document.querySelector('.modal-overlay');
    if (modal) {
        modal.classList.remove('active');
        setTimeout(() => modal.remove(), 300); // 等待动画完成后移除
    }
    window.currentModal = null;
}

// ============ 上传进度管理 ============
const UploadProgress = {
    container: null,

    init() {
        if (this.container) return;

        const div = document.createElement('div');
        div.className = 'upload-progress-container';
        div.innerHTML = `
            <div class="upload-progress-header">
                <span class="upload-title">📤 正在上传文件...</span>
                <span class="upload-count">0/0</span>
            </div>
            <div class="upload-progress-bar">
                <div class="upload-progress-fill"></div>
            </div>
            <div class="upload-file-list"></div>
        </div `;
        document.body.appendChild(div);
        this.container = div;
        this.fileList = div.querySelector('.upload-file-list');
        this.fill = div.querySelector('.upload-progress-fill');
        this.count = div.querySelector('.upload-count');
    },

    show(total) {
        this.init();
        this.total = total;
        this.completed = 0;
        this.success = 0;
        this.failed = 0;
        this.fileList.innerHTML = '';
        this.container.classList.add('active');
    },

    update(current, filename, success) {
        this.completed = current;
        this.count.textContent = `${current}/${this.total}`;

        const percent = (current / this.total) * 100;
        this.fill.style.width = `${percent}%`;

        // 添加文件项
        const fileItem = document.createElement('div');
        fileItem.className = `upload-file-item ${success ? 'success' : 'failed'}`;
        fileItem.innerHTML = `
            <span class="file-status">${success ? '✅' : '❌'}</span>
            <span class="file-name">${escapeHtml(filename)}</span>
        `;
        this.fileList.appendChild(fileItem);

        if (success) this.success++;
        else this.failed++;

        // 保持滚动到底部
        this.fileList.scrollTop = this.fileList.scrollHeight;
    },

    hide() {
        if (this.container) {
            setTimeout(() => {
                this.container.classList.remove('active');
                if (this.failed === 0) {
                    showToast(`✅ 上传成功！${this.success} 个文件`);
                } else if (this.success === 0) {
                    showToast(`❌ 上传失败，请检查文件格式`, 'error');
                } else {
                    showToast(`⚠️ ${this.success} 成功，${this.failed} 失败`, 'warning');
                }
            }, 500);
        }
    }
};

// ============ 文件上传 ============
async function uploadSingleFile(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const token = localStorage.getItem('access_token');
        const headers = {};
        if (token) headers['Authorization'] = `Bearer ${token}`;

        const response = await fetch(`${getApiUrl()}/rag/documents/upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        const data = await response.json();
        if (!data.success) throw new Error(data.message || '上传失败');
        if (onProgress) onProgress(true);
        return true;
    } catch (error) {
        console.error(`上传 ${file.name} 失败:`, error);
        if (onProgress) onProgress(false);
        return false;
    }
}

async function handleBatchUpload(files) {
    if (files.length === 0) return;

    UploadProgress.show(files.length);

    const total = files.length;

    for (let i = 0; i < total; i++) {
        await uploadSingleFile(files[i], (success) => {
            UploadProgress.update(i + 1, files[i].name, success);
        });
    }

    UploadProgress.hide();

    // 刷新列表
    await Promise.all([loadDocuments(), loadKnowledgeBaseStats()]);
}

// ============ 事件监听 ============
document.addEventListener('DOMContentLoaded', () => {
    LoadingManager.init();
    UploadProgress.init();

    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('documentInput');
    const searchInput = document.getElementById('docSearchInput');

    // 文件上传
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            handleBatchUpload(files);
            fileInput.value = '';
        });
    }

    if (uploadArea) {
        ['dragover', 'dragleave'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                if (eventName === 'dragover') uploadArea.classList.add('dragging');
                else uploadArea.classList.remove('dragging');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragging');

            const files = Array.from(e.dataTransfer.files);
            const supportedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.csv'];

            const validFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return supportedExtensions.includes(ext);
            });

            const invalidFiles = files.filter(file => {
                const ext = '.' + file.name.split('.').pop().toLowerCase();
                return !supportedExtensions.includes(ext);
            });

            if (invalidFiles.length > 0) {
                showToast(`已过滤 ${invalidFiles.length} 个不支持的文件`, 'warning');
            }

            if (validFiles.length > 0) {
                handleBatchUpload(validFiles);
            }
        });
    }

    // 搜索
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            searchKeyword = e.target.value;
            renderDocuments();
        });
    }

    // 批量操作按钮
    const selectAllBtn = document.getElementById('selectAllBtn');
    if (selectAllBtn) {
        selectAllBtn.addEventListener('click', selectAllDocuments);
    }

    const batchDeleteBtn = document.getElementById('batchDeleteBtn');
    if (batchDeleteBtn) {
        batchDeleteBtn.addEventListener('click', batchDeleteDocuments);
    }

    console.log('✅ knowledge.js 增强版已加载');
});

// 暴露给全局
window.deleteDocument = deleteDocument;
window.toggleDocumentSelect = toggleDocumentSelect;
window.showDocumentDetail = showDocumentDetail;
window.previewDocument = previewDocument;
window.closeModal = closeModal;
